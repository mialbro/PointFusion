import open3d as o3d

import numpy as np
import pyrealsense2 as rs

from typing import Optional

from pointfusion.camera import Camera

class D455(Camera):
    def __init__(self, width: Optional[int] = 1280, height: Optional[int] = 720, fps: Optional[int] = 30):
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
        self._width = width
        self._height = height
        self._fps = fps
        # Configure depth and color streams
        self._pipeline = rs.pipeline()
        self._config = rs.config()

        # Get device product line for setting a supporting resolution
        self._pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        self._config.resolve(self._pipeline_wrapper)

        self._config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        self._config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

        # Start streaming
        self._pipeline_profile = self._pipeline.start(self._config)
        self._align = rs.align(rs.stream.color)

        # Constructor for parent camera class
        super().__init__(intrinsics=self.depth_intrinsics, depth_scale=self.depth_scale)

    def __del__(self):
        """
        Stops camera model on return
        """
        if hasattr(self, '_pipeline') and self._pipeline is not None:
            self._pipeline.stop()

    def __iter__(self):
        return self
    
    def __next__(self) -> tuple:
        """
        Returns synchronized RGB Image, Depth Image, Point Cloud
        """
        ret, frames = self._pipeline.try_wait_for_frames(100)

        while ret is False:
            ret, frames = self._pipeline.try_wait_for_frames(100)

        aligned_frames = self._align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image[depth_image > 3.0 / self.depth_scale] = 0
        color_image = np.asanyarray(color_frame.get_data())

        points, colors = self.back_project(depth_image, color_image[...,::-1])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

        return color_image, depth_image, point_cloud

    @property
    def depth_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """
        Returns Depth camera intrinsics
        """
        return self.get_frames().get_depth_frame().get_profile().as_video_stream_profile().get_intrinsics()
    
    @property
    def color_intrinsics(self) -> rs.pyrealsense2.intrinsics:
        """
        Returns RGB camera intrinsics
        """
        return self.self.get_frames().get_color_frame().get_profile().as_video_stream_profile().get_intrinsics()
    
    @property
    def depth_scale(self) -> float:
        """
        Returns depth scale
        """
        return self._pipeline_profile.get_device().first_depth_sensor().get_depth_scale()
    
    def get_frames(self) -> np.ndarray:
        """
        Returns aligned frames
        """
        ret, frames = self._pipeline.try_wait_for_frames(100)
        while ret is False:
            ret, frames = self._pipeline.try_wait_for_frames(100)
        aligned_frames = self._align.process(frames)
        return aligned_frames