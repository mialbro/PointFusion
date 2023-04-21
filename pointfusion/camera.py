import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# camera = Camera(camera_matrix=self.intrinsics[index], rvec=self.rvecs[index], tvec=self.tvecs[index])

class Camera:
    def __init__(self, camera_matrix=None, rotation=None, translation=None, frame_id='camera', parent_id='model'):
        self._camera_matrix = np.asarray(camera_matrix).reshape((3, 3))
        self._rotation = None
        if len(rotation) == 3:
            self._rotation = R.from_rotvec(np.asarray(rotation).reshape((3, 1)))
        elif len(rotation) == 9:
            self._rotation = R.from_matrix(np.asarray(rotation).reshape((3, 3)))
        self._translation = np.asarray(translation).reshape((3, 1))
        self._frame_id = frame_id
        self._parent_id = parent_id

    @property
    def frame_id(self):
        return self._frame_id
    
    @property
    def parent_id(self):
        return self._parent_id

    @property
    def intrinsics(self):
        return self._camera_matrix
    
    @property
    def pose(self):
        return self._extrinsics
    
    @property
    def rmat(self):
        return self._rotation.as_matrix()
    
    @property
    def rvec(self):
        return self._rotation.as_rotvec()
    
    @property
    def tvec(self):
        return self._translation
    
    @property
    def projection_matrix(self):
        return np.multiply(self.intrinsics, self.extrinsics)
    
    @property
    def fx(self):
        return self.intrinsics[0, 0]
    
    @property
    def fy(self):
        return self.intrinsics[1, 1]
    
    @property
    def cx(self):
        return self.intrinsics[0, 2]
    
    @property
    def cy(self):
        return self.intrinsics[1, 2]
    
    def project(self, points):
        return np.matmul(self.projection_matrix, points)
    
    def transform(self, points):
        return (np.matmul(self.rmat, points.T) + self.tvec).T
    
    def backProject(self, uv, z): # [n x 2], [n x 1]
        # ([n x 2] - self.cx) / self.fx
        x = (uv - self.cx) / self.fx
        y = (uv - self.cy) / self.fy
    
    def backProject(self, depth, mask): # [r x c], 
        # [r x c], [r x c] -> 
        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]), sparse=False)
        valid = (depth > 0) & (mask != False)
        z = np.where(valid, depth, np.nan)
        x = np.where(valid, (z * (u - K['cx'])) / K['fx'], 0)
        y = np.where(valid, (z * (v - K['cy'])) / K['fy'], 0)

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        points = np.dstack((x, y, z))[0]
        points = points[~np.isnan(points).any(axis=1)]

    def depth_to_cloud(self, depth):
        valid = (depth > 0)
        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]), sparse=False)
        x = ((depth * (u - self.cx)) / self.fx)[valid == True].flatten()
        y = ((depth * (v - self.cy)) / self.fy)[valid == True].flatten()
        z = depth[valid == True].flatten()
        points = np.stack((x,y,z), axis=-1)
        return points
    
    def inverse(self):
        rotation = np.transpose(self.rmat)
        translation = -np.matmul(rotation, self.tvec)

        rotation = rotation.flatten().tolist()
        translation = translation.flatten().tolist()
        return Camera(camera_matrix=self.intrinsics, rotation=rotation, translation=translation)