import open3d as o3d

import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs
from typing import Optional, Tuple

from pointfusion.camera import Camera

class D455(Camera):
    def __init__(
            self,
            width: Optional[int] = 1280,
            height: Optional[int] = 720,
            fps: Optional[int] = 30
        ) -> None:
        """
        RealSense D455 camera driver
        Args:
            width (int): Camera width
            height (int): Camera height
            fps (int): Camera frame rate
        Attributes:
            _width (int)
            _height (int)
            _fps (int)
            _pipeline (pyrealsense2.pipeline)
            _config (pyrealsense2.config)
        """
        # Configure depth and color streams
        self._pipeline = rs.pipeline()
        config = rs.config()
        if config.can_resolve(self._pipeline):
            # Get device product line for setting a supporting resolution
            self._pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
            config.resolve(self._pipeline_wrapper)
            config.enable_stream(
                rs.stream.depth,
                width,
                height,
                rs.format.z16,
                fps
            )
            config.enable_stream(
                rs.stream.color,
                width,
                height,
                rs.format.bgr8,
                fps
            )
            # Start streaming
            self._pipeline_profile = self._pipeline.start(config)
            self._align = rs.align(rs.stream.color)
        # Constructor for parent camera class
        super().__init__(intrinsics=self.depth_intrinsics, depth_scale=self.depth_scale)

    def __del__(self) -> None:
        """Stops camera model on return"""
        if hasattr(self, '_pipeline') and self._pipeline is not None:
            self._pipeline.stop()

    def __iter__(self) -> "D455":
        return self

    def __next__(
            self
        ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Returns the next images in the iteration
        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        """
        ret, frames = self._pipeline.try_wait_for_frames(100)
        while ret is False:
            ret, frames = self._pipeline.try_wait_for_frames(100)
        # Get RGB and Depth frames
        aligned_frames = self._align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # Filter depth image
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image[depth_image > 3.0 / self.depth_scale] = 0
        color_image = np.asanyarray(color_frame.get_data())
        # Backproject point cloud
        points, colors = self.back_project(depth_image, color_image[...,::-1])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
        import pdb; pdb.set_trace()
        return color_image, depth_image, point_cloud

    @property
    def depth_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """Returns Depth camera intrinsics"""
        profile = self.get_frames().get_depth_frame().get_profile().as_video_stream_profile()
        return profile.get_intrinsics()

    @property
    def color_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """Returns RGB camera intrinsics"""
        profile = self.get_frames().get_color_frame().get_profile().as_video_stream_profile()
        return profile.get_intrinsics()

    @property
    def depth_scale(self) -> float:
        """Returns depth scale"""
        return self._pipeline_profile.get_device().first_depth_sensor().get_depth_scale()

    def get_frames(self) -> np.ndarray:
        """Returns aligned frames"""
        ret, frames = self._pipeline.try_wait_for_frames(100)
        while ret is False:
            ret, frames = self._pipeline.try_wait_for_frames(100)
        aligned_frames = self._align.process(frames)
        return aligned_frames
