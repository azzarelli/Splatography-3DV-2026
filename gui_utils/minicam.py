
import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch



def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, time, from_training_view=False):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.time = time

        
        loaded = False
        if from_training_view:
            if type(c2w) == type({'type':'dict'}):
                self.world_view_transform = c2w['world_view_transform']

                self.c2w = torch.linalg.inv(self.world_view_transform.cuda().transpose(0,1))
                self.projection_matrix = c2w['projection_matrix']
                self.full_proj_transform = c2w['full_proj_transform']

                loaded = True
        
        if not loaded:
            self.c2w = c2w
            w2c = np.linalg.inv(c2w)
            # rectify...
            w2c[1:3, :3] *= -1
            w2c[:3, 3] *= -1

            self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
            self.projection_matrix = (
                getProjectionMatrix(
                    znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
                )
                .transpose(0, 1)
                .cuda().float()
            )
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        self.camera_center = -torch.tensor(self.c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

 
class OrbitCamera:
    def __init__(self, W, H, r=2, fov=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        
        self.fovy = 2 * np.arctan(np.tan(fov[0]/2)* 0.25)
        self.fovy = np.deg2rad(self.fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.array([[1., 0., 0.,],
                                           [0., 0., -1.],
                                           [0., 1., 0.]]))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.side = np.array([1, 0, 0], dtype=np.float32)

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        up = self.rot.as_matrix()[:3, 1]
        rotvec_x = up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0001 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

    def get_proj_matrix(self):
        tanHalfFovY = math.tan((self.fovy  / 2))
        tanHalfFovX = math.tan((self.fovx / 2))

        top = tanHalfFovY * self.near
        bottom = -top
        right = tanHalfFovX * self.near
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * self.near / (right - left)
        P[1, 1] = 2.0 * self.near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.far / (self.far - self.near)
        P[2, 3] = -(self.far * self.near) / (self.far - self.near)
        return P

# Going from 3x3 rotation matrix to quaternion values
def R_to_q(R,eps=1e-8): # [B,3,3]
            # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
            # FIXME: this function seems a bit problematic, need to double-check
            row0,row1,row2 = R.unbind(dim=-2)
            R00,R01,R02 = row0.unbind(dim=-1)
            R10,R11,R12 = row1.unbind(dim=-1)
            R20,R21,R22 = row2.unbind(dim=-1)
            t = R[...,0,0]+R[...,1,1]+R[...,2,2]
            r = (1+t+eps).sqrt()
            qa = 0.5*r
            qb = (R21-R12).sign()*0.5*(1+R00-R11-R22+eps).sqrt()
            qc = (R02-R20).sign()*0.5*(1-R00+R11-R22+eps).sqrt()
            qd = (R10-R01).sign()*0.5*(1-R00-R11+R22+eps).sqrt()
            q = torch.stack([qa,qb,qc,qd],dim=-1)
            
            for i,qi in enumerate(q):
                if torch.isnan(qi).any():
                    K = torch.stack([torch.stack([R00-R11-R22,R10+R01,R20+R02,R12-R21],dim=-1),
                                    torch.stack([R10+R01,R11-R00-R22,R21+R12,R20-R02],dim=-1),
                                    torch.stack([R20+R02,R21+R12,R22-R00-R11,R01-R10],dim=-1),
                                    torch.stack([R12-R21,R20-R02,R01-R10,R00+R11+R22],dim=-1)],dim=-2)/3.0
                    K = K[i]
                    eigval,eigvec = torch.linalg.eigh(K)
                    V = eigvec[:,eigval.argmax()]
                    q[i] = torch.stack([V[3],V[0],V[1],V[2]])
            return q

def hash_cams(cam):
    return cam.camera_center.sum().item()

class GaussianCameraModel:

    def get_camera_stuff(self, cam_list):
        xyzs = []
        qs = []
        
        hash_camlist = []

        # For each camera in list get the camera position and rotation
        for i, cam in enumerate(cam_list):
            # Hash the camera (i.e. avoid displaying cameras with the same R and T)
            cam_hash = hash_cams(cam)
            # We do this but summing the camera center points (sure it seems silly but this is realistically fine....(I think)
            if cam_hash not in hash_camlist:
                hash_camlist.append(cam_hash)

                for xyz in self.cam_xyzs:
                    T = cam.camera_center + xyz.cpu()
                    w2c = cam.world_view_transform.transpose(0,1)
                    R = w2c[:3, :3].transpose(0,1)
                    T = torch.matmul(R, xyz.unsqueeze(-1).cpu()).squeeze(-1) + cam.camera_center

                    q = R_to_q(R)

                    xyzs.append(T.unsqueeze(0))
                    qs.append(q.unsqueeze(0))
            # else:
            #     break

        self.qs = torch.cat(qs, dim=0).cuda()
        self.xyzs = torch.cat(xyzs, dim=0).cuda()

        self.xyzs.requires_grad = True
        self.qs.requires_grad = True

    def get_cam_model(self, cam, W, H):
        w = float(W/2.)
        h = float(H/2.)
        
        x_num_pts = 11

        w2c = cam.world_view_transform.transpose(0,1)
        R = w2c[:3, :3].transpose(0,1)
        xyzs = []
        qs = []

        if w > h:
            x_size = float(4.*x_num_pts)
            y_size = float(3.*x_num_pts)
        elif w < h:
            x_size = float(3.*x_num_pts)
            y_size = float(4.*x_num_pts)
        else:
            x_size = float(3.*x_num_pts)
            y_size = float(3.*x_num_pts)
            
        # Line in +X (Right) 
        for i in range(0, x_num_pts+1):
            x = float(i-(x_num_pts)/2.)/x_size
            T = torch.tensor([x, +0.2, 0.])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))

        for i in range(0, x_num_pts+1):
            x = float(i-(x_num_pts)/2.)/x_size
            T = torch.tensor([x, -0.2, 0.])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))
        
        # Line in -Z (Back)
        for i in range(0, 5):
            T = torch.tensor([0., 0., - float(i/10.)])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))

        # Line in +Y (Down) Show direction
        T = torch.tensor([-0.1, 0., 0.])
        q = R_to_q(R)
        xyzs.append(T.unsqueeze(0))
        qs.append(q.unsqueeze(0))

        # Show -x -y for the topleft hand image
        T = torch.tensor([0., -0.1, 0.])
        q = R_to_q(R)
        xyzs.append(T.unsqueeze(0))
        qs.append(q.unsqueeze(0))
        
        
        self.cam_qs = torch.cat(qs, dim=0)
        self.cam_xyzs = torch.cat(xyzs, dim=0)

    def __init__(self, cam_list, W, H) -> None:

        max_i = 1
        if len(cam_list) > 100: max_i = 300

        try:
            cam = cam_list[0][0]
            cam_list = [cam_list[i][0] for i in range(len(cam_list)) if i % max_i == 0]
        except:
            cam = cam_list[0]
            cam_list = [cam_list[i] for i in range(len(cam_list)) if i % max_i == 0]
        self.get_cam_model(cam, W, H)

        self.get_camera_stuff(cam_list)

        cam_centers = []
        cam_hash = []

        for i, cam in enumerate(cam_list):
            T = cam.camera_center
            if str(T) not in cam_hash:
                cam_centers.append(T)
                cam_hash.append(str(T))
        tensor_stack = torch.stack(cam_centers)
        self.cam_center = torch.mean(tensor_stack, dim=0)
        
        # Blob scale NOT camera scale
        self.scale = torch.tensor([0.001, 0.001, 0.001]).cuda()
