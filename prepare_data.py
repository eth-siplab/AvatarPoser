'''
# --------------------------------------------
# data preprocessing for AMASS dataset
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (jiaxi.jiang@inf.ethz.ch)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''
import torch
import numpy as np
import os
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose
from utils import utils_transform
import time
import pickle

dataroot_amass ="amass" # root of amass dataset

for dataroot_subset in ["MPI_HDM05", "BioMotionLab_NTroje", "CMU"]:
    print(dataroot_subset)
    for phase in ["train","test"]:
        print(phase)
        savedir = os.path.join("./data_fps60", dataroot_subset, phase)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        split_file = os.path.join("./data_split", dataroot_subset, phase+"_split.txt")

        with open(split_file, 'r') as f:
            filepaths = [line.rstrip('\n') for line in f]


        rotation_local_full_gt_list = []

        hmd_position_global_full_gt_list = []

        body_parms_list = []

        head_global_trans_list = []


        support_dir = 'support_data/'
        bm_fname_male = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('male'))
        dmpl_fname_male = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('male'))

        bm_fname_female = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('female'))
        dmpl_fname_female = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('female'))

        num_betas = 16 # number of body parameters
        num_dmpls = 8 # number of DMPL parameters
        bm_male = BodyModel(bm_fname=bm_fname_male, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_male)#.to(comp_device)
        bm_female = BodyModel(bm_fname=bm_fname_female, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_female)

        idx = 0
        for filepath in filepaths:
            data = dict()
            bdata = np.load(filepath,allow_pickle=True)
            # print(list(bdata.keys())) ### check keys of body data: ['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']
            try:
                framerate = bdata["mocap_framerate"]
                print("framerate is {}".format(framerate))
            except:
                print(filepath)
                print(list(bdata.keys()))       
                continue                          # skip shape.npz
        #        pass
        #    else:
            idx+=1
            print(idx)

            if framerate == 120:
                stride = 2
            elif framerate == 60:
                stride = 1

            bdata_poses = bdata["poses"][::stride,...]
            bdata_trans = bdata["trans"][::stride,...]
            subject_gender = bdata["gender"]

            bm = bm_male# if subject_gender == 'male' else bm_female

            body_parms = {
                'root_orient': torch.Tensor(bdata_poses[:, :3]),#.to(comp_device), # controls the global root orientation
                'pose_body': torch.Tensor(bdata_poses[:, 3:66]),#.to(comp_device), # controls the body
                'trans': torch.Tensor(bdata_trans),#.to(comp_device), # controls the global body position
            }

            body_parms_list = body_parms

            body_pose_world=bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','root_orient','trans']})

        #            self.rotation_local_full_gt_list.append(body_parms['pose_body'])
        #            self.rotation_local_full_gt_list.append(torch.Tensor(bdata['poses'][:, :66]))
            output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1,3)
            output_6d = utils_transform.aa2sixd(output_aa).reshape(bdata_poses.shape[0],-1)
            rotation_local_full_gt_list = output_6d[1:]

            rotation_local_matrot = aa2matrot(torch.tensor(bdata_poses).reshape(-1,3)).reshape(bdata_poses.shape[0],-1,9)
            rotation_global_matrot = local2global_pose(rotation_local_matrot, bm.kintree_table[0].long()) # rotation of joints relative to the origin

            head_rotation_global_matrot = rotation_global_matrot[:,[15],:,:]

            rotation_global_6d = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1,3,3)).reshape(rotation_global_matrot.shape[0],-1,6)
            input_rotation_global_6d = rotation_global_6d[1:,[15,20,21],:]

            rotation_velocity_global_matrot = torch.matmul(torch.inverse(rotation_global_matrot[:-1]),rotation_global_matrot[1:])
            rotation_velocity_global_6d = utils_transform.matrot2sixd(rotation_velocity_global_matrot.reshape(-1,3,3)).reshape(rotation_velocity_global_matrot.shape[0],-1,6)
            input_rotation_velocity_global_6d = rotation_velocity_global_6d[:,[15,20,21],:]

            position_global_full_gt_world = body_pose_world.Jtr[:,:22,:] # position of joints relative to the world origin

            position_head_world = position_global_full_gt_world[:,15,:] # world position of head

            head_global_trans = torch.eye(4).repeat(position_head_world.shape[0],1,1)
            head_global_trans[:,:3,:3] = head_rotation_global_matrot.squeeze()
            head_global_trans[:,:3,3] = position_global_full_gt_world[:,15,:]

            head_global_trans_list = head_global_trans[1:]




            num_frames = position_global_full_gt_world.shape[0]-1


            hmd_position_global_full_gt_list = torch.cat([
                                                                    input_rotation_global_6d.reshape(num_frames,-1),
                                                                    input_rotation_velocity_global_6d.reshape(num_frames,-1),
                                                                    position_global_full_gt_world[1:, [15,20,21], :].reshape(num_frames,-1), 
                                                                    position_global_full_gt_world[1:, [15,20,21], :].reshape(num_frames,-1)-position_global_full_gt_world[:-1, [15,20,21], :].reshape(num_frames,-1)], dim=-1)

            data_count = len(hmd_position_global_full_gt_list)

            print(str(idx)+'/'+str(len(filepaths)))


            data['rotation_local_full_gt_list'] = rotation_local_full_gt_list

            data['hmd_position_global_full_gt_list'] = hmd_position_global_full_gt_list

            data['body_parms_list'] = body_parms_list

            data['head_global_trans_list'] = head_global_trans_list

            data['framerate'] = 60

            data['gender'] = subject_gender

            data['filepath'] = filepath


            with open(os.path.join(savedir,'{}.pkl'.format(idx)), 'wb') as f:
                pickle.dump(data, f)
