import torch
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose
import random
from utils import utils_transform

from scipy import signal

import glob
from IPython import embed
import time
import copy
import pickle


class AMASS_Dataset(Dataset):
    """Motion Capture dataset"""

    def __init__(self, opt):
#        embed()
        self.opt = opt
        self.window_size = opt['window_size']
        self.num_input = opt['num_input']

        self.batch_size = opt['dataloader_batch_size']
        dataroot = opt['dataroot']


# CMU,BioMotionLab_NTroje,MPI_HDM05
        if self.opt['phase'] == 'train':
            self.filename_list = glob.glob('data_fps60/*/train/*.pkl')
        else:
            self.filename_list = glob.glob('data_fps60/*/test/*.pkl')

            print('-------------------------------number of test data is{}'.format(len(self.filename_list)))

    def __len__(self):

        return max(len(self.filename_list), self.batch_size)


    def __getitem__(self, idx):

        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if self.opt['phase'] == 'train':
            while data['rotation_local_full_gt_list'].shape[0] <self.window_size:
                idx = random.randint(0,idx)
                filename = self.filename_list[idx]
                with open(filename, 'rb') as f:
                    data = pickle.load(f)


        rotation_local_full_gt_list = data['rotation_local_full_gt_list']

        hmd_position_global_full_gt_list = data['hmd_position_global_full_gt_list']

        body_parms_list = data['body_parms_list']

        head_global_trans_list = data['head_global_trans_list']


        if self.opt['phase'] == 'train':
#            embed()
#            start = time.perf_counter()
            frame = np.random.randint(hmd_position_global_full_gt_list.shape[0] - self.window_size + 1 - 1)
            input_hmd  = hmd_position_global_full_gt_list[frame:frame + self.window_size+1,...].reshape(self.window_size+1, -1).float()
#            output_gt = rotation_local_full_gt_list[frame + self.window_size - 1+40,...].float()
            output_gt = rotation_local_full_gt_list[frame + self.window_size - 1 : frame + self.window_size - 1 + 1,...].float()
            output_gt_last = rotation_local_full_gt_list[frame + self.window_size - 2 : frame + self.window_size - 2 + 1,...].float()

#            print('finish load')

            output_gt_velocity_matrot = torch.matmul(torch.inverse(utils_transform.sixd2matrot(output_gt_last.reshape(-1,6))),utils_transform.sixd2matrot(output_gt.reshape(-1,6)))
            output_gt_velocity = utils_transform.matrot2sixd(output_gt_velocity_matrot).reshape(-1,132)



            return {'L': input_hmd,
                    'H': output_gt,
                    'H_v':output_gt_velocity,
                    'P': 1,
                    'Head_trans_global':head_global_trans_list[frame + self.window_size - 1:frame + self.window_size - 1+1,...],
                    'pos_pelvis_gt':body_parms_list['trans'][frame + self.window_size - 1:frame + self.window_size - 1+1,...],
                    'vel_pelvis_gt':body_parms_list['trans'][frame + self.window_size - 1:frame + self.window_size - 1+1,...]-body_parms_list['trans'][frame + self.window_size - 2:frame + self.window_size - 2+1,...]
#                    'pos_pelvis_gt':body_parms_list['trans'][frame + self.window_size - 1+40,...]
                    }

        else:
#            print(len(self.inputList))
#            print(idx)
            input_hmd  = hmd_position_global_full_gt_list.reshape(hmd_position_global_full_gt_list.shape[0], -1)[1:]

            output_gt = rotation_local_full_gt_list[1:]
            output_gt_last = rotation_local_full_gt_list[:-1]

            output_gt_velocity_matrot = torch.matmul(torch.inverse(utils_transform.sixd2matrot(output_gt_last.reshape(-1,6))),utils_transform.sixd2matrot(output_gt.reshape(-1,6)))
            output_gt_velocity = utils_transform.matrot2sixd(output_gt_velocity_matrot).reshape(-1,132)



            return {'L': input_hmd.float(),
                    'H': output_gt.float(),
                    'H_v':output_gt_velocity,
                    'P': body_parms_list,
                    'Head_trans_global':head_global_trans_list[1:],
                    'pos_pelvis_gt':body_parms_list['trans'][2:],
                    'vel_pelvis_gt':body_parms_list['trans'][2:]-body_parms_list['trans'][1:-1]
                    }


    