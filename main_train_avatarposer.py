import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader


from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset

from models.select_model import define_Model_AMASS
from IPython import embed
from scipy.spatial.transform import Rotation as R
import time
 
from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa
from utils import utils_transform

import pickle

save_animation = False


def main(json_path='options/train_avatarposer.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    border = 0
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



    from utils import utils_visualize as vis

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt['datasets']['train']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():
#        embed()

        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model_AMASS(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())
#    embed()
#    start = torch.cuda.Event(enable_timing=True)
#    end = torch.cuda.Event(enable_timing=True)
    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
#    embed()
    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1
#            print(i)
#            embed()
            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------
#            start = time.perf_counter()
#            start.record()
#            embed()
            model.feed_data(train_data)

            # -------------------------------
            # 1b) initialize hidden state for time-series network
            # -------------------------------
            model.init_hidden()



#            end.record()
#            torch.cuda.synchronize()
#            print(start.elapsed_time(end))

            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 3) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)


            # -------------------------------
            # merge bnorm
            # -------------------------------
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                model.print_network()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                idx = 0


                rot_error = []
                pos_error = []
                vel_error = []
                pos_error_pelvis_pos = []
                pos_error_pelvis_pos_hands = []
                pos_error_pelvis_pos_head = []


                pos_error_pelvis_pos_post = []
                pos_error_head_align = []
                pos_error_post = []
                pos_error_pseudo = []
                pos_error_pseudo_post = []
                pos_error_median = []
                pos_error_median_pelvis_pos = []
                pos_error_median_pelvis_pos_post = []
                pos_error_median_head_align = []
                pos_error_post_median = []
                pos_error_pseudo_median = []
                pos_error_pseudo_post_median = []



                save_results = []
                save_results_gt = []

                for index, test_data in enumerate(test_loader):
                    print("testing the sample {}/{}".format(index, len(test_loader)))
#                    embed()

                    save_dict = {}
                    save_dict_gt = {}
#                    image_name_ext = os.path.basename(test_data['L_path'][0])
#                    img_name, ext = os.path.splitext(image_name_ext)
#                    embed()
                    if index % 5 == 0:
                        video_dir = os.path.join(opt['path']['images'], str(idx))
                        util.mkdir(video_dir)
#                    print(idx)
                    model.feed_data(test_data, test=True)
#                    embed()
#                    if model.L.shape[1]< model.window_size:
#                        print("Sequence Too short!")
#                        continue

                    # -------------------------------
                    # initialize hidden state for time-series network
                    # -------------------------------
                    model.init_hidden()
#                    print('run model')
                    model.test()
#                    print('run model finished')

#                    embed()
                    results = model.current_results()
                    predicted_angle = results['E']#.cpu().numpy()
                    gt_angle = results['H']#.cpu().numpy()


                    body_parms = results['P']

                    pos_pelvis_pred = results['pos_pelvis_pred']

#                    embed()
                    for k,v in body_parms.items():
                        body_parms[k] = v.squeeze().cuda()
                        body_parms[k] = body_parms[k][-predicted_angle.shape[0]:,...]

#                    embed()
                    # -----------------------
                    # save groundtruth avatar video
                    # -----------------------
                    save_video_path_gt = os.path.join(video_dir, 'gt.avi')
                    body_pose_gt=vis.bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','trans', 'root_orient']})
#                    embed()
                    save_dict_gt['pose_body'] = body_parms['pose_body']
                    save_dict_gt['trans'] = body_parms['trans']
                    save_dict_gt['root_orient'] = body_parms['root_orient']
                    save_results_gt.append(save_dict_gt)

                    position_global_full_gt = body_pose_gt.Jtr[:,:22,:]
#                    embed()

                    if not os.path.exists(save_video_path_gt):
                        if save_animation:
                            vis.save_animation(body_pose=body_pose_gt, savepath=save_video_path_gt, fps=60)

  
#                    save_video_path_gt_input = os.path.join(video_dir, 'gt_input.avi')
#                    embed()
                    '''
                    if not os.path.exists(save_video_path_gt_input):
                        rotation_local_matrot_gt = aa2matrot(torch.cat([body_parms['root_orient'].cuda(),body_parms['pose_body']],dim=1).reshape(-1,3)).reshape(body_parms['root_orient'].shape[0],-1,9)
                        rotation_global_matrot_gt = local2global_pose(rotation_local_matrot_gt, vis.bm.kintree_table[0][:22].long())

                        head2root_rotation_gt = rotation_global_matrot_gt[:,15,:]
                        head_position_gt = body_pose_gt.Jtr[:,15,:]
                        head_transform_matrix = torch.eye(4).repeat(head_position_gt.shape[0],1,1)
                        head_transform_matrix[:,:3,:3] = head2root_rotation_gt
                        head_transform_matrix[:,:3,3] = head_position_gt

                        Lefthand2root_rotation_gt = rotation_global_matrot_gt[:,20,:]
                        Lefthand_posotion_gt = body_pose_gt.Jtr[:,20,:]
                        Lefthand_transform_matrix = torch.eye(4).repeat(head_position_gt.shape[0],1,1)
                        Lefthand_transform_matrix[:,:3,:3] = Lefthand2root_rotation_gt
                        Lefthand_transform_matrix[:,:3,3] = Lefthand_posotion_gt

                        Righthand2root_rotation_gt = rotation_global_matrot_gt[:,21,:]
                        Righthand_posotion_gt = body_pose_gt.Jtr[:,21,:]
                        Righthand_transform_matrix = torch.eye(4).repeat(head_position_gt.shape[0],1,1)
                        Righthand_transform_matrix[:,:3,:3] = Righthand2root_rotation_gt
                        Righthand_transform_matrix[:,:3,3] = Righthand_posotion_gt
                    '''


                    for k,v in body_parms.items():
#                        body_parms[k] = v.squeeze().cuda()
                        if k == 'pose_body':
                            body_parms[k] = predicted_angle[...,3:66]
#                        body_parms[k] = body_parms[k][-predicted_angle.shape[0]:,...]


                #    gt_angle = np.concatenate([np.zeros([326,3]),gt_angle],axis=1)

                    T_head2world = results['Head_trans_global'].clone()
                    T_head2root_pred = torch.eye(4).repeat(T_head2world.shape[0],1,1).cuda()
#                    embed()
#                    rotation_local_matrot = aa2matrot(torch.tensor(np.concatenate([np.zeros([predicted_angle.shape[0],3]),predicted_angle[...,3:66]],axis=1)).reshape(-1,3)).reshape(predicted_angle.shape[0],-1,9)
                    rotation_local_matrot = aa2matrot(torch.cat([torch.zeros([predicted_angle.shape[0],3]).cuda(),predicted_angle[...,3:66]],dim=1).reshape(-1,3)).reshape(predicted_angle.shape[0],-1,9)
                    rotation_global_matrot = local2global_pose(rotation_local_matrot, vis.bm.kintree_table[0][:22].long())
                    head2root_rotation = rotation_global_matrot[:,15,:]

                    body_pose_local_pred=vis.bm(**{'pose_body':predicted_angle[...,3:66]})
                    head2root_translation = body_pose_local_pred.Jtr[:,15,:]

#                    embed()
                    T_head2root_pred[:,:3,:3] = head2root_rotation
                    T_head2root_pred[:,:3,3] = head2root_translation


                    t_head2world = T_head2world[:,:3,3].clone()
                    T_head2world[:,:3,3] = 0

#                    embed()
                    T_root2world_pred = torch.matmul(T_head2world, torch.inverse(T_head2root_pred))

                    rotation_root2world_pred = matrot2aa(T_root2world_pred[:,:3,:3])
                    translation_root2world_pred = T_root2world_pred[:,:3,3]
#                    embed()
#                    body_pose=vis.bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas','trans', 'root_orient']})
#                    body_pose_local=vis.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':rotation_root2world_pred})
                    body_pose_local=vis.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3]})
                    position_global_full_local = body_pose_local.Jtr[:,:22,:]
                    t_head2root = position_global_full_local[:,15,:]
                    t_root2world = -t_head2root+t_head2world.cuda()
#                    embed()
                    body_pose_head_align=vis.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':rotation_root2world_pred.cuda(), 'trans': t_root2world})
                    position_global_full_pred_head_align = body_pose_head_align.Jtr[:,:22,:]
                    predicted_position_all_head_align = position_global_full_pred_head_align.cpu().numpy()

                    # 'root_orient':rotation_root2world_pred.cuda()
                    body_pose=vis.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': t_root2world})

                    save_dict['pose_body'] = predicted_angle[...,3:66]
                    save_dict['trans'] = t_root2world
                    save_dict['root_orient'] = predicted_angle[...,:3]
                    save_results.append(save_dict)


                    position_global_full_pred = body_pose.Jtr[:,:22,:]
                    predicted_position_all = position_global_full_pred.cpu().numpy()

                    body_pose_pred_pelvis_pos=vis.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': pos_pelvis_pred})
                    position_global_full_pred_pelvis_pos = body_pose_pred_pelvis_pos.Jtr[:,:22,:]


                    if index % 5 == 0:
                        try:
                            save_video_path = os.path.join(video_dir, '{:d}.avi'.format(current_step))
                            if save_animation:
                                vis.save_animation(body_pose=body_pose_pred_pelvis_pos, savepath=save_video_path, fps=60)
                        except:
                            pass
                    predicted_position_all_pelvis_pos = position_global_full_pred_pelvis_pos.detach().cpu().numpy()

                    offset_vertical = torch.nn.functional.relu(-torch.amin(position_global_full_pred_pelvis_pos[:,:,2],dim=1))
                    pos_pelvis_pred[:,2] = pos_pelvis_pred[:,2] + offset_vertical
                    body_pose_pred_pelvis_pos_post=vis.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': pos_pelvis_pred})
                    position_global_full_pred_pelvis_pos_post = body_pose_pred_pelvis_pos_post.Jtr[:,:22,:]
                    predicted_position_all_pelvis_pos_post = position_global_full_pred_pelvis_pos_post.detach().cpu().numpy()


#                    print('start post processing')

                    # -------------------------------
                    # Post processing, lowest joint of human body should not be lower than ground
                    # -------------------------------

                    offset_vertical = torch.nn.functional.relu(-torch.amin(position_global_full_pred[:,:,2],dim=1))
                    t_root2world[:,2] = t_root2world[:,2] + offset_vertical
                    body_pose=vis.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': t_root2world})                    

                    position_global_full_all_post = body_pose.Jtr[:,:22,:].cpu().numpy()


                    body_pose_pseudo=vis.bm(**{'pose_body': body_parms['pose_body'], 'trans': body_parms['trans'], 'root_orient': body_parms['root_orient']})
                    position_global_full_pred_pseudo = body_pose_pseudo.Jtr[:,:22,:]
                    offset_vertical = torch.nn.functional.relu(-torch.amin(position_global_full_pred_pseudo[:,:,2],dim=1))
                    t_root2world_psuedo = body_parms['trans']
                    t_root2world_psuedo[:,2] = t_root2world_psuedo[:,2] + offset_vertical

                    body_pose_pseudo_post=vis.bm(**{'pose_body': body_parms['pose_body'], 'trans': t_root2world_psuedo, 'root_orient': body_parms['root_orient']})
                    position_global_full_pred_pseudo_post = body_pose_pseudo_post.Jtr[:,:22,:]
                    position_global_full_all_pseudo_post = position_global_full_pred_pseudo_post.cpu().numpy()

                    gt_position_all = position_global_full_gt.cpu().numpy()

                    predicted_position_pseudo_all = position_global_full_pred_pseudo.cpu().numpy()


#                    embed()
                    predicted_angle_all = np.array(predicted_angle.cpu()).reshape(results['E'].shape[0],-1,3)                    
                    gt_angle_all = np.array(gt_angle.cpu()).reshape(results['E'].shape[0],-1,3)

#                    embed()





#                    print('start calculating error')


#                    embed()
                    rot_error_ = np.mean(np.absolute(gt_angle_all-predicted_angle_all))










                    pos_error_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all),axis=-1)))

                    gt_velocity_all = (gt_position_all[1:,...] - gt_position_all[:-1,...])*60
                    predicted_velocity_all = (predicted_position_all[1:,...] - predicted_position_all[:-1,...])*60
                    vel_error_ = np.mean(np.sqrt(np.sum(np.square(gt_velocity_all-predicted_velocity_all),axis=-1)))


                    pos_error_pelvis_pos_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all_pelvis_pos),axis=-1)))
                    pos_error_pelvis_pos_hands_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all),axis=-1))[...,[20,21]])
                    pos_error_pelvis_pos_head_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all),axis=-1))[...,[15]])

                    pos_error_pelvis_pos_post_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all_pelvis_pos_post),axis=-1)))
                    pos_error_head_align_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all_head_align),axis=-1)))
                    pos_error_post_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-position_global_full_all_post),axis=-1)))
                    pos_error_pseudo_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_pseudo_all),axis=-1)))
                    pos_error_pseudo_post_ = np.mean(np.sqrt(np.sum(np.square(gt_position_all-position_global_full_all_pseudo_post),axis=-1)))
                    '''
                    pos_error_median_ = np.median(np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all),axis=-1)),axis=-1))
                    pos_error_median_pelvis_pos_ = np.median(np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all_pelvis_pos),axis=-1)),axis=-1))
                    pos_error_median_pelvis_pos_post_ = np.median(np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all_pelvis_pos_post),axis=-1)),axis=-1))
                    pos_error_median_head_align_ = np.median(np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_all_head_align),axis=-1)),axis=-1))
                    pos_error_post_median_ = np.median(np.mean(np.sqrt(np.sum(np.square(gt_position_all-position_global_full_all_post),axis=-1)),axis=-1))
                    pos_error_pseudo_median_ = np.median(np.mean(np.sqrt(np.sum(np.square(gt_position_all-predicted_position_pseudo_all),axis=-1)),axis=-1))
                    pos_error_pseudo_post_median_ = np.median(np.mean(np.sqrt(np.sum(np.square(gt_position_all-position_global_full_all_pseudo_post),axis=-1)),axis=-1))
                    '''

                    rot_error.append(rot_error_)
                    pos_error.append(pos_error_)
                    vel_error.append(vel_error_)

                    pos_error_pelvis_pos.append(pos_error_pelvis_pos_)
                    pos_error_pelvis_pos_hands.append(pos_error_pelvis_pos_hands_)
                    pos_error_pelvis_pos_head.append(pos_error_pelvis_pos_head_)

                    pos_error_pelvis_pos_post.append(pos_error_pelvis_pos_post_)
                    pos_error_head_align.append(pos_error_head_align_)
                    pos_error_post.append(pos_error_post_)
                    pos_error_pseudo.append(pos_error_pseudo_)
                    pos_error_pseudo_post.append(pos_error_pseudo_post_)

                    '''
                    pos_error_median.append(pos_error_median_)
                    pos_error_median_pelvis_pos.append(pos_error_median_pelvis_pos_)
                    pos_error_median_pelvis_pos_post.append(pos_error_median_pelvis_pos_post_)
                    pos_error_median_head_align.append(pos_error_median_head_align_)
                    pos_error_post_median.append(pos_error_post_median_)
                    pos_error_pseudo_median.append(pos_error_pseudo_median_)
                    pos_error_pseudo_post_median.append(pos_error_pseudo_post_median_)
                    '''

                    idx += 1


#                    print('finish calculating error')


                output = open('prediction_ours_3inputs_cameraready.pkl', 'wb')
                pickle.dump(save_results, output)
                output.close()
                print('save prediction done!')

                output = open('gt_ours_3inputs_cameraready.pkl', 'wb')
                pickle.dump(save_results_gt, output)
                output.close()
                print('save gt done!')


                rot_error = sum(rot_error)/len(rot_error)
                pos_error = sum(pos_error)/len(pos_error)
                vel_error = sum(vel_error)/len(vel_error)
                pos_error_pelvis_pos = sum(pos_error_pelvis_pos)/len(pos_error_pelvis_pos)
                pos_error_pelvis_pos_hands = sum(pos_error_pelvis_pos_hands)/len(pos_error_pelvis_pos_hands)
                pos_error_pelvis_pos_head = sum(pos_error_pelvis_pos_head)/len(pos_error_pelvis_pos_head)

                pos_error_pelvis_pos_post = sum(pos_error_pelvis_pos_post)/len(pos_error_pelvis_pos_post)
                pos_error_head_align = sum(pos_error_head_align)/len(pos_error_head_align)
                pos_error_post = sum(pos_error_post)/len(pos_error_post)
                pos_error_pseudo = sum(pos_error_pseudo)/len(pos_error_pseudo)
                pos_error_pseudo_post = sum(pos_error_pseudo_post)/len(pos_error_pseudo_post)

                '''
                pos_error_median = sum(pos_error_median)/len(pos_error_median)
                pos_error_median_pelvis_pos = sum(pos_error_median_pelvis_pos)/len(pos_error_median_pelvis_pos)
                pos_error_median_pelvis_pos_post = sum(pos_error_median_pelvis_pos_post)/len(pos_error_median_pelvis_pos_post)
                pos_error_median_head_align = sum(pos_error_median_head_align)/len(pos_error_median_head_align)
                pos_error_post_median = sum(pos_error_post_median)/len(pos_error_post_median)
                pos_error_pseudo_median = sum(pos_error_pseudo_median)/len(pos_error_pseudo_median)
                pos_error_pseudo_post_median = sum(pos_error_pseudo_post_median)/len(pos_error_pseudo_post_median)
                '''
                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average rotational error : {:<.5f},\
                    Average positional error (pelvis pos) : {:<.5f},Average positional error (pelvis pos, post) : {:<.5f},\
                    Average positional error (head align) : {:<.5f},  Average positional error : {:<.5f},\
                     Average positional error (post) : {:<.5f}, Average positional error pseudo: {:<.5f}, \
                     Average positional error pseudo (post): {:<.5f}\n'.format(epoch, current_step,rot_error*57.2958, \
                        pos_error_pelvis_pos,pos_error_pelvis_pos_post,\
                         pos_error_head_align, pos_error,\
                          pos_error_post, pos_error_pseudo,\
                           pos_error_pseudo_post))

                logger.info('velocity_error: {:<.5f}'.format(vel_error))
                logger.info('hand_error: {:<.5f}'.format(pos_error_pelvis_pos_hands))
                logger.info('head_error: {:<.5f}'.format(pos_error_pelvis_pos_head))

#                logger.info('<epoch:{:3d}, iter:{:8,d}, Average rotational error : {:<.5f},Median positional error (pelvis pos) : {:<.5f},Median positional error (pelvis pos, post) : {:<.5f},Median positional error (head align) : {:<.5f},  Median positional error : {:<.5f}, Median positional error (post) : {:<.5f}, Median positional error pseudo: {:<.5f}, Median positional error pseudo (post): {:<.5f}\n'.format(epoch, current_step, rot_error, pos_error_median_pelvis_pos,pos_error_median_pelvis_pos_post, pos_error_median_head_align, pos_error_median, pos_error_post_median, pos_error_pseudo_median, pos_error_pseudo_post_median))


    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
