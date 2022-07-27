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
import trimesh.util as util
from psbody.mesh import Mesh

os.environ['PYOPENGL_PLATFORM'] = 'egl'

"""
# --------------------------------
# CheckerBoard, from Xianghui Xie
# --------------------------------
"""

class CheckerBoard:
    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array(white)/255.
        self.black = np.array(black)/255.
        self.verts, self.faces, self.texts = None, None, None
        self.offset = None

    def init_checker(self, offset, plane='xz', xlength=500, ylength=200, square_size=0.5):
        "generate checkerboard and prepare v, f, t"
        checker = self.gen_checker_xy(self.black, self.white, square_size, xlength, ylength)
        rot = np.eye(3)
        if plane == 'xz':
            # rotate around x-axis by 90
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplemented
        checker.v = np.matmul(checker.v, rot.T)

        # apply offsets
        checker.v += offset
        self.offset = offset

        self.verts, self.faces, self.texts = self.prep_checker_rend(checker)

    def get_rends(self):
        return self.verts, self.faces, self.texts

    def append_checker(self, checker):
        "append another checker"
        v, f, t = checker.get_rends()
        nv = self.verts.shape[1]
        self.verts = torch.cat([self.verts, v], 1)
        self.faces = torch.cat([self.faces, f+nv], 1)
        self.texts = torch.cat([self.texts, t], 1)

    @staticmethod
    def gen_checkerboard(square_size=0.5, total_size=50.0, plane='xz'):
        "plane: the checkboard is in parallal to which plane"
        checker = CheckerBoard.gen_checker_xy(square_size, total_size)
        rot = np.eye(3)
        if plane == 'xz':
            # rotate around x-axis by 90, so that the checker plane is perpendicular to y-axis
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplemented
        checker.v = np.matmul(checker.v, rot.R)
        return checker

    def prep_checker_rend(self, checker:Mesh):
        verts = torch.from_numpy(checker.v.astype(np.float32)).cuda().unsqueeze(0)
        faces = torch.from_numpy(checker.f.astype(int)).cuda().unsqueeze(0)
        nf = checker.f.shape[0]
        texts = torch.zeros(1, nf, 4, 4, 4, 3).cuda()
        for i in range(nf):
            texts[0, i, :, :, :, :] = torch.tensor(checker.fc[i], dtype=torch.float32).cuda()
        return verts, faces, texts

    @staticmethod
    def gen_checker_xy(black, white, square_size=0.5, xlength=50.0, ylength=50.0):
        """
        generate a checker board in parallel to x-y plane
        starting from (0, 0) to (xlength, ylength), in meters
        return: psbody.Mesh
        """
        xsquares = int(xlength / square_size)
        ysquares = int(ylength / square_size)
        verts, faces, texts = [], [], []
        fcount = 0
        # black = torch.tensor([0, 0, 0.], dtype=torch.float32).cuda()
        # white = torch.tensor([1., 1., 1.], dtype=torch.float32).cuda()
        # white = np.array([247, 246, 244]) / 255.
        # black = np.array([146, 163, 171]) / 255.
        for i in range(xsquares):
            for j in range(ysquares):
                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, j * square_size, 0])
                p3 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])
                p3 = np.array([i * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                if (i + j) % 2 == 0:
                    texts.append(black)
                    texts.append(black)
                else:
                    texts.append(white)
                    texts.append(white)
                    

                    
                    
        # now compose as mesh
        mesh = Mesh(v=np.array(verts), f=np.array(faces), fc=np.array(texts))
        # mesh.write_ply("/BS/xxie2020/work/hoi3d/utils/checkerboards/mychecker.ply")
        mesh.v += np.array([-5, -5, 0])
        return mesh

    @staticmethod
    def from_meshes(meshes, yaxis_up=True, xlength=50, ylength=20):
        """
        initialize checkerboard ground from meshes
        """
        vertices = [x.v for x in meshes]
        if yaxis_up:
            # take ymin
            y_off = np.min(np.concatenate(vertices, 0), 0)
        else:
            # take ymax
            y_off = np.min(np.concatenate(vertices, 0), 0)
        offset = np.array([xlength/2, y_off[1], ylength/2]) # center to origin
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength)
        return checker

    @staticmethod
    def from_verts(verts, yaxis_up=True, xlength=5, ylength=5, square_size=0.2):
        """
        verts: (1, N, 3)
        """
        if yaxis_up:
            y_off = torch.min(verts[0], 0)[0].cpu().numpy()
        else:
            y_off = torch.max(verts[0], 0)[0].cpu().numpy()
        # print(verts.shape, y_off.shape)
        offset = np.array([-xlength/2, y_off[1], -ylength/2])
        print(offset, torch.min(verts[0], 0)[0].cpu().numpy(), torch.max(verts[0], 0)[0].cpu().numpy())
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength, square_size=square_size)
        return checker


"""
# --------------------------------
# Visualize avatar using body pose information and body model
# --------------------------------
"""




def save_animation(body_pose, savepath, bm, fps = 60, resolution = (800,800)):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    for fId in range(body_pose.v.shape[0]):
        body_mesh = trimesh.Trimesh(vertices=c2c(body_pose.v[fId]), faces=faces, vertex_colors=np.tile(colors['purple'], (6890, 1)))


        generator = CheckerBoard()
        checker = generator.gen_checker_xy(generator.black, generator.white)
        checker_mesh = trimesh.Trimesh(checker.v,checker.f,process=False,face_colors=checker.fc)

        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        checker_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 10)))
        checker_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (10, 0, 0)))
        checker_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

        mv.set_static_meshes([checker_mesh,body_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)

        img_array.append(body_image)
    out = cv2.VideoWriter(savepath,cv2.VideoWriter_fourcc(*'DIVX'), fps, resolution)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

