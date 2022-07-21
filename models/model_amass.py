from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G_amass
from models.model_base import ModelBase
from models.loss import CharbonnierLoss

from utils.utils_regularizers import regularizer_orth, regularizer_clip
from human_body_prior.tools.angle_continuous_repres import geodesic_loss_R
from IPython import embed
from utils.utils_transform import bgs
from utils import utils_transform
from utils import utils_visualize as vis

from models.network import optimiser_pose
import time


class ModelAMASS(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelAMASS, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G_amass(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        self.hidden_dim = self.opt['netG']['hidden_dim']
        self.do_FK = True
        self.window_size = self.opt['datasets']['train']['window_size']


    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log


    def init_hidden(self):
#        embed()

        self.netG.hidden = torch.zeros(1, self.L.shape[0], self.hidden_dim).to(self.device)                    # initialize hidden state

#        embed()
    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'geodesic':
            self.G_lossfn = geodesic_loss_R(reduction='mean')
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)



    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True, test=False):
#        embed()
#        embed()
        self.L = data['L'].to(self.device)
#        self.H = data['H'].to(self.device)
        self.P = data['P']
        self.Head_trans_global = data['Head_trans_global'].to(self.device)
#        self.pos_pelvis_gt = data['pos_pelvis_gt'].to(self.device)
#        print(data['H'].shape, data['pos_pelvis_gt'].shape)
#        embed()
        self.H = torch.cat([data['H'], data['pos_pelvis_gt']],dim=-1).to(self.device)
#        self.H_v = torch.cat([data['H_v'], data['vel_pelvis_gt']],dim=-1).to(self.device)[:,[6:18,24:36,42:54,60:72,-3:]]
#        embed()
#        self.H_v = torch.cat([data['H_v'][...,6:18],
#                            data['H_v'][...,24:36],
#                            data['H_v'][...,42:54],
#                            data['H_v'][...,60:72],
#                            data['vel_pelvis_gt']],dim=-1).to(self.device)
    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
#        embed()
#        start = torch.cuda.Event(enable_timing=True)
#        end = torch.cuda.Event(enable_timing=True)
#        start.record()
#        self.E,self.E_v = self.netG(self.L)
        self.E = self.netG(self.L)
#        end.record()
#        torch.cuda.synchronize()
#        print(start.elapsed_time(end))

#        embed()
    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
#        embed()
        self.netG_forward()

        G_loss = 1 * self.G_lossfn(self.E, self.H.squeeze()[...,:132]) + 0.02* self.G_lossfn(self.E[...,:6], self.H.squeeze()[...,:6])
        loss = G_loss
        if self.do_FK:
            angle_pred = utils_transform.sixd2aa(self.E[...,:132].reshape(-1,6)).reshape(self.E[...,:132].shape[0],-1).float()
            body_pose_local_pred=vis.bm(**{'pose_body':angle_pred[..., 3:66], 'root_orient':angle_pred[..., :3]})
#            position_end_effector_pred = body_pose_local_pred.Jtr[:,[7,8,10,11,20,21],:]
            position_end_effector_pred = body_pose_local_pred.Jtr
            angle_gt = utils_transform.sixd2aa(self.H[...,:132].reshape(-1,6)).reshape(self.H[...,:132].shape[0],-1).float()
#                embed()
            body_pose_local_gt=vis.bm(**{'pose_body':angle_gt[..., 3:66], 'root_orient':angle_gt[..., :3]})
#            position_end_effector_gt = body_pose_local_gt.Jtr[:,[7,8,10,11,20,21],:]
            position_end_effector_gt = body_pose_local_gt.Jtr
            EF_loss = 1 * self.G_lossfn(position_end_effector_pred, position_end_effector_gt)

            loss = G_loss + 1*EF_loss
#            G_loss = EF_loss
        loss.backward()


        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['EF_loss'] = EF_loss.item()


        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
#            import os
#            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
#            embed.()
    #    self.netIK.eval()

    #    for param in self.netG.parameters():
    #        param.requires_grad = False
            self.netG.eval()

            self.L = self.L.squeeze()
            self.H = self.H.squeeze()
    #        self.H_v = self.H_v.squeeze()            
            self.Head_trans_global = self.Head_trans_global.squeeze()
            window_size = self.opt['datasets']['train']['window_size']
            input = []
            output = []
            output2 = []
            head = []
            pelvis = []
    #            embed()


            # ------------------------------------
            # Offline
            # ------------------------------------

    #            embed()
            '''
            for start_frame in range(self.L.shape[0]-window_size+1):
                input.append(self.L[start_frame:start_frame+window_size,:])
                output.append(self.H[start_frame + window_size-1:start_frame+window_size,:].squeeze())
                head.append(self.Head_trans_global[start_frame + window_size-1:start_frame+window_size,:].squeeze())
            self.L = torch.stack(input)
            self.H = torch.stack(output)
            self.Head_trans_global = torch.stack(head)
            self.netG_forward()
            '''

            # ------------------------------------
            # Online
            # ------------------------------------
    #            embed()
            states = None
            time_list = []
            time_cpu_list = []

            input_list_1 = []

            do_singleframe = False


            if self.L.shape[0] < window_size:

                if do_singleframe:
                    input_list = []
                    for frame_idx in range(self.L.shape[0]):
                        input_list.append(self.L[[frame_idx],...].unsqueeze(0))
                    input_tensor = torch.cat(input_list, dim = 0)
                    with torch.no_grad():
                        netG_outputs = self.netG(input_tensor)
                else:
                    output_list = []
                    with torch.no_grad():
                        output_list.append(self.netG(self.L[[0]].unsqueeze(0)))                     
                    for frame_idx in range(1,self.L.shape[0]):
                        with torch.no_grad():
                            output_list.append(self.netG(self.L[0: frame_idx].unsqueeze(0))) 
                    netG_outputs = torch.cat(output_list, dim=0)

            else:  
#                print("cat input tensor")

                if do_singleframe:
                    for frame_idx in range(window_size):
                        input_list_1.append(self.L[[frame_idx],...].unsqueeze(0))
                    input_tensor_1 = torch.cat(input_list_1, dim = 0)
                    with torch.no_grad():
                        output_tensor_1 = self.netG(input_tensor_1)

                else:
                    output_list = []
                    with torch.no_grad():
                        output_list.append(self.netG(self.L[[0]].unsqueeze(0)))                     

                    with torch.no_grad():
                        for frame_idx in range(1,window_size):
                                output_list.append(self.netG(self.L[0: frame_idx].unsqueeze(0))) 

                    output_tensor_1 = torch.cat(output_list, dim=0)                   

                input_list_2 = []
                for frame_idx in range(window_size,self.L.shape[0]):
                    input_list_2.append(self.L[frame_idx-window_size+1:frame_idx+1,...].unsqueeze(0))
                input_tensor_2 = torch.cat(input_list_2, dim = 0)

                with torch.no_grad():
                    output_tensor_2 = self.netG(input_tensor_2)

                netG_outputs = torch.cat([output_tensor_1, output_tensor_2], dim=0)


            self.E = netG_outputs

            self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float()#.cpu()
#        embed()
        out_dict['E'] = utils_transform.sixd2aa(self.E[:,:132].reshape(-1,6).detach()).reshape(self.E[:,:132].shape[0],-1).float()#.cpu()
#        out_dict['E_v'] = utils_transform.sixd2aa(self.E_v[:,:48].reshape(-1,6).detach()).reshape(self.E_v[:,:48].shape[0],-1).float()#.cpu()

        out_dict['P'] = self.P

        out_dict['H'] = utils_transform.sixd2aa(self.H[:,:132].reshape(-1,6).detach()).reshape(self.H[:,:132].shape[0],-1).float()#.cpu()
        out_dict['Head_trans_global'] = self.Head_trans_global
        out_dict['pos_pelvis_pred'] = self.E[:,:3]#self.E[:,132:]#self.E[:,:3]#self.E[:,132:]
#        out_dict['vel_pelvis_pred'] = self.E_v[:,48:]

        return out_dict


    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
