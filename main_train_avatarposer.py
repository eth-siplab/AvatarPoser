import os.path
import math
import argparse
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import utils_transform
import pickle
from utils import utils_visualize as vis


save_animation = False
resolution = (800,800)

def main(json_path='options/train_avatarposer.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

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

    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1
            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------
            
            model.feed_data(train_data)

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


                rot_error = []
                pos_error = []
                vel_error = []
                pos_error_hands = []

                for index, test_data in enumerate(test_loader):
                    logger.info("testing the sample {}/{}".format(index, len(test_loader)))

                    model.feed_data(test_data, test=True)

                    model.test()

                    body_parms_pred = model.current_prediction()
                    body_parms_gt = model.current_gt()
                    predicted_angle = body_parms_pred['pose_body']
                    predicted_position = body_parms_pred['position']
                    predicted_body = body_parms_pred['body']

                    gt_angle = body_parms_gt['pose_body']
                    gt_position = body_parms_gt['position']
                    gt_body = body_parms_gt['body']



                    if index in [0, 10, 20] and save_animation:
                        video_dir = os.path.join(opt['path']['images'], str(index))
                        if not os.path.exists(video_dir):
                            os.makedirs(video_dir)

                        save_video_path_gt = os.path.join(video_dir, 'gt.avi')
                        if not os.path.exists(save_video_path_gt):
                            vis.save_animation(body_pose=gt_body, savepath=save_video_path_gt, bm = model.bm, fps=60, resolution = resolution)

                        save_video_path = os.path.join(video_dir, '{:d}.avi'.format(current_step))
                        vis.save_animation(body_pose=predicted_body, savepath=save_video_path, bm = model.bm, fps=60, resolution = resolution)


                    predicted_position = predicted_position#.cpu().numpy()
                    gt_position = gt_position#.cpu().numpy()

                    predicted_angle = predicted_angle.reshape(body_parms_pred['pose_body'].shape[0],-1,3)                    
                    gt_angle = gt_angle.reshape(body_parms_gt['pose_body'].shape[0],-1,3)


                    rot_error_ = torch.mean(torch.absolute(gt_angle-predicted_angle))
                    pos_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1)))
                    pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]])

                    gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*60
                    predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*60
                    vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))

                    rot_error.append(rot_error_)
                    pos_error.append(pos_error_)
                    vel_error.append(vel_error_)

                    pos_error_hands.append(pos_error_hands_)



                rot_error = sum(rot_error)/len(rot_error)
                pos_error = sum(pos_error)/len(pos_error)
                vel_error = sum(vel_error)/len(vel_error)
                pos_error_hands = sum(pos_error_hands)/len(pos_error_hands)


                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average rotational error [degree]: {:<.5f}, Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}, Average positional error at hand [cm]: {:<.5f}\n'.format(epoch, current_step,rot_error*57.2958, pos_error*100, vel_error*100, pos_error_hands*100))


    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
