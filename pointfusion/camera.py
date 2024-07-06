import cv2
import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from typing import Optional

class Camera:
    """Pinhole camera model
    Args:
        intrinsics (rs.pyrealsense2.intrinsics): RealSense camera model
        camera_matrix (np.ndarray): Camera intrinsics matrix
        dist_coeffs (npt.NDArray[np.float64]): Distortion coefficients
        rotation (npt.NDArray[np.float64]): Rotation matrix
        translation (npt.NDArray[np.float64]): Translation vector
        depth_scale (Optional[float]): Depth scale
        frame_id (Optional[str]): Current camera frame
        parent_id (Optional[str]): Camera parent frame
    """
    def __init__(
            self,
            intrinsics: Optional[rs.pyrealsense2.intrinsics] = None,
            camera_matrix: Optional[np.ndarray] = np.eye(3),
            dist_coeffs: Optional[np.ndarray] = np.zeros(5),
            rotation: Optional[np.ndarray] = np.zeros(3),
            translation: Optional[np.ndarray] = np.zeros(3),
            depth_scale: Optional[float] = 1.0,
            frame_id: Optional[str] = 'camera',
            parent_id: Optional[str] = 'model'
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
    def frame_id(self) -> str:
        """Get frame id from camera
        Returns:
            str: Frame ID
        """
        return self._frame_id

    @property
    def parent_id(self) -> str:
        """Get parent id from camera
        Returns:
            str: Parent frame ID
        """
        return self._parent_id

    @property
    def intrinsics(self) -> npt.NDArray[np.float64]:
        """Get camera matrix from camera
        Returns:
            npt.NDArray[np.float64]
        """
        return self._camera_matrix

    @property
    def camera_matrix(self) -> npt.NDArray[np.float64]:
        """Get camera matrix from camera
        Returns:
            npt.NDArray[np.float64]: Camera Matrix
        """
        return self._camera_matrix

    @property
    def pose(self) -> npt.NDArray[np.float64]:
        """Get 3D rigid transformation
        Returns:
            npt.NDArray[np.float64]: 4x4 transformation matrix
        """
        pose = np.eye(4)
        pose[:3, :3] = self.rmat
        pose[:3, 3] = self.tvec.reshape((3,))
        return pose

    @property
    def rmat(self) -> npt.NDArray[np.float64]:
        """Get rotation matrix
        Returns:
            npt.NDArray[np.float64]: 3x3 rotation matrix
        """
        return self._rotation.as_matrix()

    @property
    def rvec(self) -> npt.NDArray[np.float64]:
        """Get rotation vector
        Returns:
            npt.NDArray[np.float64]: Rotation vector
        """
        return self._rotation.as_rotvec()

    @property
    def tvec(self) -> npt.NDArray[np.float64]:
        """Get translation vector
        Returns:
            npt.NDArray[np.float64]: 3x1 translation vector
        """
        return self._translation

    @property
    def P(self) -> npt.NDArray[np.float64]:
        """Get projection matrix
        Returns:
            npt.NDArray[np.float64]: 3x4 PRojection matrix
        """
        tmat = self.pose[:3, :]
        return np.matmul(self.intrinsics, tmat)

    @property
    def fx(self) -> float:
        """Get focal length (x)
        Returns:
            float: Horizontal focal length
        """
        return self.intrinsics[0, 0]

    @property
    def fy(self) -> float:
        """Get focal length (y)
        Returns:
            float: Vertical focal length
        """
        return self.intrinsics[1, 1]

    @property
    def cx(self) -> float:
        """Get optical center (x)
        Returns:
            Horizontal optical center
        """
        return self.intrinsics[0, 2]

    @property
    def cy(self) -> float:
        """Get optical center (y)
        Returns:
            float: Vertical optical center
        """
        return self.intrinsics[1, 2]

    def transform(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform 3D points into camera frame
        Args:
            points (npt.NDArray[np.float64]): Nx3 Point Cloud
        Returns:
            npt.NDArray[np.float64]: Transformed Nx3 Point Cloud
        """
        return (np.matmul(self.rmat, points.T) + self.tvec).T

    def inverse(self) -> "Camera":
        """Return inverse of camera
        Returns:
            Camera: Inversed camera
        """
        rotation = np.transpose(self.rmat)
        translation = -np.matmul(rotation, self.tvec)
        rotation = rotation.flatten().tolist()
        translation = translation.flatten().tolist()
        return Camera(camera_matrix=self.intrinsics, rotation=rotation, translation=translation)

    def project(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project object points into Image frame
        Returns:
            npt.NDArray[np.float64]: Camera points projected  projected from 
        """
        if points.ndim == 1:
            points = points.reshape((3, 1))
            points = np.vstack((points, np.ones((1, 1))))
            image_points = np.matmul(self.P, points)
            image_points = image_points / image_points[2, :]
            return image_points[:2, :]
        elif points.ndim == 2 and points.shape[-1] == 3:
            points = np.transpose(np.hstack((points, np.ones((points.shape[0], 1)))))
            image_points = np.matmul(self.P, points)
            image_points = image_points / image_points[-1, :]
            image_points = np.transpose(image_points)[:, :2]
            return image_points

    def back_project(
            self,
            depth: np.ndarray,
            color_image: Optional[np.ndarray] = None
        ) -> npt.NDArray[np.float64]:
        """
        Back project image points using camera model and depth
        """
        depth = depth * self._depth_scale
        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]), sparse=False)
        uv = np.stack((u.flatten(), v.flatten()), axis=-1).astype(np.float32)
        uv = np.squeeze(cv2.undistortPoints(uv, self.intrinsics, self._dist_coeffs))
        x = ((depth * (u - self.cx)) / self.fx)[depth > 0].flatten()
        y = ((depth * (v - self.cy)) / self.fy)[depth > 0].flatten()
        z = depth[depth > 0].flatten()
        points = np.stack((x, y ,z), axis=-1)
        if color_image is not None:
            colors = color_image[depth > 0].reshape(-1, 3)
        else:
            colors = np.ones((points.shape[0], 3))
        return points, colors
