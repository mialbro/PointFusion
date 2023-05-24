import open3d as o3d

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

class Camera:
    def __init__(
            self,
            intrinsics=None,
            camera_matrix=np.eye(3), 
            dist_coeffs=np.zeros(5), 
            rotation=np.zeros(3), 
            translation=np.zeros(3), 
            depth_scale=1.0, 
            frame_id='camera', 
            parent_id='model'
        ):
        # Intrinsics
        if isinstance(intrinsics, rs.pyrealsense2.intrinsics):
            self._camera_matrix = np.eye(3)
            self._camera_matrix[0, 0] = intrinsics.fx
            self._camera_matrix[1, 1] = intrinsics.fy
            self._camera_matrix[0, 2] = intrinsics.ppx
            self._camera_matrix[1, 2] = intrinsics.ppy
            self._dist_coeffs = np.asarray(intrinsics.coeffs).reshape((5, 1))
        elif isinstance(camera_matrix, np.ndarray):
            self._camera_matrix = np.asarray(camera_matrix).reshape((3, 3))
        elif isinstance(camera_matrix, list):
            self._camera_matrix = np.asarray(camera_matrix).reshape((3, 3))
        self._dist_coeffs = np.asarray(dist_coeffs).reshape((5, 1))
        self._depth_scale = depth_scale

        # Extrinsics
        numel = len(rotation) if isinstance(rotation, list) else rotation.size
        if numel == 3:
            self._rotation = R.from_rotvec(np.asarray(rotation))
        elif numel == 9:
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
    def camera_matrix(self):
        return self._camera_matrix
    
    @property
    def pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.rmat
        pose[:3, 3] = self.tvec.reshape((3,))
        return pose
    
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
        tmat = self.pose[:3, :]
        return np.matmul(self.intrinsics, tmat)
    
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
        if points.ndim == 1:
            points = points.reshape((3, 1))
            points = np.vstack((points, np.ones((1, 1))))
            image_points = np.matmul(self.projection_matrix, points)
            image_points = image_points / image_points[2, :]
            return image_points[:2, :]
        elif points.ndim == 2 and points.shape[-1] == 3:
            points = np.transpose(np.hstack((points, np.ones((points.shape[0], 1)))))
            image_points = np.matmul(self.projection_matrix, points)
            image_points = image_points / image_points[-1, :]
            image_points = np.transpose(image_points)[:, :2]
            return image_points
    
    def transform(self, points):
        return (np.matmul(self.rmat, points.T) + self.tvec).T

    def back_project(self, depth, color_image=None):
        depth = depth * self._depth_scale
        valid = (depth > 0)
        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]), sparse=False)
        uv = np.stack((u.flatten(), v.flatten()), axis=-1).astype(np.float32)
        uv = np.squeeze(cv2.undistortPoints(uv, self.intrinsics, self._dist_coeffs))
        x = ((depth * (u - self.cx)) / self.fx)[valid == True].flatten()
        y = ((depth * (v - self.cy)) / self.fy)[valid == True].flatten()
        z = depth[valid == True].flatten()
        points = np.stack((x, y ,z), axis=-1)
        if color_image is not None:
            colors = color_image[valid == True].reshape(-1, 3)
        else:
            colors = np.ones((points.shape[0], 3))
        return points, colors
    
    def inverse(self):
        rotation = np.transpose(self.rmat)
        translation = -np.matmul(rotation, self.tvec)
        rotation = rotation.flatten().tolist()
        translation = translation.flatten().tolist()
        return Camera(camera_matrix=self.intrinsics, rotation=rotation, translation=translation)