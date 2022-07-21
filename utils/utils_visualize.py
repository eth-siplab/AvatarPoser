import torch
import cv2
import os
import numpy as np
import trimesh
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose,matrot2aa

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os


from trimesh.creation import uv_sphere
import trimesh.transformations as tf
from trimesh.creation import cylinder 
import trimesh.util as util

from IPython import embed
os.environ['PYOPENGL_PLATFORM'] = 'egl'

support_dir = 'support_data/'

"""
# --------------------------------
# Import npz file from amass dataset
# --------------------------------
"""

amass_npz_fname = os.path.join('../../Datasets/amass/CMU/01/01_01_poses.npz') # the path to body data
bdata = np.load(amass_npz_fname)
subject_gender = bdata['gender']
time_length = len(bdata['trans'])
#print('time_length = {}'.format(time_length))


"""
# --------------------------------
# Set up body model
# --------------------------------
"""

bm_fname = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters
bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
faces = c2c(bm.f)




"""
# --------------------------------
# Extract useful information from bdata
# --------------------------------
"""
body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}

#print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))


"""
# --------------------------------
# Visualize avatar using body pose information and body model
# --------------------------------
"""

body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas','trans', 'root_orient']})








def axis(origin_size=0.04,
         transform=None,
         origin_color=None,
         axis_radius=None,
         axis_length=None):
    """
    Return an XYZ axis marker as a  Trimesh, which represents position
    and orientation. If you set the origin size the other parameters
    will be set relative to it.
    Parameters
    ----------
    transform : (4, 4) float
      Transformation matrix
    origin_size : float
      Radius of sphere that represents the origin
    origin_color : (3,) float or int, uint8 or float
      Color of the origin
    axis_radius : float
      Radius of cylinder that represents x, y, z axis
    axis_length: float
      Length of cylinder that represents x, y, z axis
    Returns
    -------
    marker : trimesh.Trimesh
      Mesh geometry of axis indicators
    """
    # the size of the ball representing the origin
    origin_size = float(origin_size)

    # set the transform and use origin-relative
    # sized for other parameters if not specified
    if transform is None:
        transform = np.eye(4)
    if origin_color is None:
        origin_color = [255, 255, 255, 255]
    if axis_radius is None:
        axis_radius = origin_size / 5.0
    if axis_length is None:
        axis_length = 0.4 #origin_size * 10.0 0.4

    # generate a ball for the origin
    axis_origin = uv_sphere(radius=origin_size,
                            count=[10, 10])
    axis_origin.apply_transform(transform)

    # apply color to the origin ball
    #axis_origin.visual.face_colors = origin_color

    # create the cylinder for the z-axis
    translation = tf.translation_matrix(
        [0, 0, axis_length / 2])
    z_axis = cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(translation))
    # XYZ->RGB, Z is blue
    #z_axis.visual.face_colors = [0, 0, 255]

    # create the cylinder for the y-axis
    translation = tf.translation_matrix(
        [0, 0, axis_length / 2])
    rotation = tf.rotation_matrix(np.radians(-90),
                                  [1, 0, 0])
    y_axis = cylinder(
        radius=axis_radius,
        height=axis_length*2,
        transform=transform.dot(rotation).dot(translation))
    # XYZ->RGB, Y is green
    #y_axis.visual.face_colors = [0, 255, 0]

    # create the cylinder for the x-axis
    translation = tf.translation_matrix(
        [0, 0, axis_length / 2])
    rotation = tf.rotation_matrix(np.radians(90),
                                  [0, 1, 0])
    x_axis = cylinder(
        radius=axis_radius,
        height=axis_length*10,
        transform=transform.dot(rotation).dot(translation))
    # XYZ->RGB, X is red
    #x_axis.visual.face_colors = [255, 0, 0]

    # append the sphere and three cylinders
    marker = util.concatenate([axis_origin,
                               x_axis,
                               y_axis,
                               z_axis])
    return marker




def save_animation(body_pose, savepath = '../video/project.avi', fps = 120, resolution = (800,800), visualize_axis=False, pose_body=None,root_orient=None):
    imw, imh=800, 800
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    img_array = []
    for fId in range(body_pose.v.shape[0]):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        surface_mesh = trimesh.Trimesh(vertices=[[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]], faces=[[0, 1, 2], [0, 2, 3]]) #, vertex_colors=np.tile(colors['grey'], (4, 1))
        xyz_mesh = axis()

        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        surface_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        surface_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        surface_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        if visualize_axis:

            rotation_local_matrot = aa2matrot(torch.cat([root_orient,pose_body],dim=1).reshape(-1,3)).reshape(root_orient.shape[0],-1,9)
            rotation_global_matrot = local2global_pose(rotation_local_matrot, bm.kintree_table[0][:22].long())

            head2root_rotation = rotation_global_matrot[:,15,:]
            head_position = body_pose.Jtr[:,15,:]
            Head_transform_matrix = torch.eye(4).repeat(head_position.shape[0],1,1)
            Head_transform_matrix[:,:3,:3] = head2root_rotation
            Head_transform_matrix[:,:3,3] = head_position

            Lefthand2root_rotation = rotation_global_matrot[:,20,:]
            Lefthand_posotion = body_pose.Jtr[:,20,:]
            Lefthand_transform_matrix = torch.eye(4).repeat(head_position.shape[0],1,1)
            Lefthand_transform_matrix[:,:3,:3] = Lefthand2root_rotation
            Lefthand_transform_matrix[:,:3,3] = Lefthand_posotion

            Righthand2root_rotation = rotation_global_matrot[:,21,:]
            Righthand_posotion = body_pose.Jtr[:,21,:]
            Righthand_transform_matrix = torch.eye(4).repeat(head_position.shape[0],1,1)
            Righthand_transform_matrix[:,:3,:3] = Righthand2root_rotation
            Righthand_transform_matrix[:,:3,3] = Righthand_posotion

            xyz_mesh_head = axis(transform=Head_transform_matrix[fId].numpy(),axis_length=0.2)
            xyz_mesh_left_hand = axis(transform=Lefthand_transform_matrix[fId].numpy(),axis_length=0.2)
            xyz_mesh_right_hand = axis(transform=Righthand_transform_matrix[fId].numpy(),axis_length=0.2)

            xyz_mesh_head.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
            xyz_mesh_head.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
            xyz_mesh_head.apply_transform(trimesh.transformations.scale_matrix(0.5))

            xyz_mesh_left_hand.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
            xyz_mesh_left_hand.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
            xyz_mesh_left_hand.apply_transform(trimesh.transformations.scale_matrix(0.5))

            xyz_mesh_right_hand.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
            xyz_mesh_right_hand.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
            xyz_mesh_right_hand.apply_transform(trimesh.transformations.scale_matrix(0.5))

            mv.set_static_meshes([body_mesh, surface_mesh, xyz_mesh,xyz_mesh_head,xyz_mesh_left_hand,xyz_mesh_right_hand])

        else:
            mv.set_static_meshes([body_mesh, surface_mesh, xyz_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        img_array.append(body_image)
    out = cv2.VideoWriter(savepath,cv2.VideoWriter_fourcc(*'DIVX'), fps, resolution)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def save_animation_input(Head_transform_matrix=None,
                         Lefthand_transform_matrix=None,
                         Righthand_transform_matrix=None,
                         savepath = '../video/project.avi', fps = 120, resolution = (800,800)):
    imw, imh=800, 800
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    img_array = []
    for fId in range(Head_transform_matrix.shape[0]):
        surface_mesh = trimesh.Trimesh(vertices=[[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]], faces=[[0, 1, 2], [0, 2, 3]]) #, vertex_colors=np.tile(colors['grey'], (4, 1))
        xyz_mesh = axis()
        xyz_mesh_head = axis(transform=Head_transform_matrix[fId],axis_length=0.2)
        xyz_mesh_left_hand = axis(transform=Lefthand_transform_matrix[fId],axis_length=0.2)
        xyz_mesh_right_hand = axis(transform=Righthand_transform_matrix[fId],axis_length=0.2)


        surface_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        surface_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        surface_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_head.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_head.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_head.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_left_hand.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_left_hand.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_left_hand.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_right_hand.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_right_hand.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_right_hand.apply_transform(trimesh.transformations.scale_matrix(0.5))

        mv.set_static_meshes([surface_mesh, xyz_mesh,xyz_mesh_head,xyz_mesh_left_hand,xyz_mesh_right_hand])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        img_array.append(body_image)
    out = cv2.VideoWriter(savepath,cv2.VideoWriter_fourcc(*'DIVX'), fps, resolution)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



def save_animation_input_full(Head_transform_matrix=None,
                         Lefthand_transform_matrix=None,
                         Righthand_transform_matrix=None,
                         Hips_transform_matrix=None,
                         Leftfoot_transform_matrix=None,
                         Rightfoot_transform_matrix=None,

                         savepath = '../video/project.avi', fps = 120, resolution = (800,800)):
    imw, imh=800, 800
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    img_array = []
    for fId in range(Head_transform_matrix.shape[0]):
        surface_mesh = trimesh.Trimesh(vertices=[[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]], faces=[[0, 1, 2], [0, 2, 3]]) #, vertex_colors=np.tile(colors['grey'], (4, 1))
        xyz_mesh = axis()
#        embed()
        xyz_mesh_head = axis(transform=Head_transform_matrix[fId],axis_length=0.2)
        xyz_mesh_left_hand = axis(transform=Lefthand_transform_matrix[fId],axis_length=0.2)
        xyz_mesh_right_hand = axis(transform=Righthand_transform_matrix[fId],axis_length=0.2)

        xyz_mesh_hips = axis(transform=Hips_transform_matrix[fId],axis_length=0.2)
        xyz_mesh_left_foot = axis(transform=Leftfoot_transform_matrix[fId],axis_length=0.2)
        xyz_mesh_right_foot = axis(transform=Rightfoot_transform_matrix[fId],axis_length=0.2)


        surface_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        surface_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        surface_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_head.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_head.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_head.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_left_hand.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_left_hand.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_left_hand.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_right_hand.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_right_hand.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_right_hand.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_hips.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_hips.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_hips.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_left_foot.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_left_foot.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_left_foot.apply_transform(trimesh.transformations.scale_matrix(0.5))

        xyz_mesh_right_foot.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        xyz_mesh_right_foot.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        xyz_mesh_right_foot.apply_transform(trimesh.transformations.scale_matrix(0.5))


        mv.set_static_meshes([surface_mesh, xyz_mesh,xyz_mesh_head,xyz_mesh_left_hand,xyz_mesh_right_hand,
                              xyz_mesh_hips,xyz_mesh_left_foot,xyz_mesh_right_foot])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        img_array.append(body_image)
    out = cv2.VideoWriter(savepath,cv2.VideoWriter_fourcc(*'DIVX'), fps, resolution)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

#save_animation(body_pose = body_pose_beta)