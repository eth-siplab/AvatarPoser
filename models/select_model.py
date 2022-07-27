import functools
import torch
from torch.nn import init
from human_body_prior.body_model.body_model import BodyModel
import os


"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'avatarposer':  # two inputs: L, C
        from models.model_avatarposer import ModelAvatarPoser as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m



"""
# --------------------------------------------
# select the network of G
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------


def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    support_dir = opt['support_dir']
    subject_gender = "male"
    bm_fname = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
    dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    body_model = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)


    if net_type == 'AvatarPoser':
        from models.network import AvatarPoser as net
        netG = net(input_dim=opt_net['input_dim'],
                   output_dim=opt_net['output_dim'],
                   num_layer=opt_net['num_layer'],
                   embed_dim=opt_net['embed_dim'],
                   nhead = opt_net['nhead'],
                   body_model = body_model,
                   device = device)

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG



"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                #init.orthogonal_(m.weight.data, gain=gain)
                pass
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

#            if m.bias is not None:
#                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')
