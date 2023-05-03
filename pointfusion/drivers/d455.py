import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

import pointfusion

class D455:
    def __init__(self, width=1280, height=720, fps=30, viewer=True):
        self._viewer = viewer
        self._width = width
        self._height = height
        self._fps = fps
        # Configure depth and color streams
        self._pipeline = rs.pipeline()
        self._config = rs.config()

        # Get device product line for setting a supporting resolution
        self._pipeline_wrapper = rs.pipeline_wrapper(self._pipeline)
        self._pipeline_profile = self._config.resolve(self._pipeline_wrapper)
        self._device = self._pipeline_profile.get_device()
        self._device_product_line = str(self._device.get_info(rs.camera_info.product_line))

        self._config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        self._config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

        # Start streaming
        self._pipeline_profile = self._pipeline.start(self._config)

        align_to = rs.stream.color
        self._align = rs.align(align_to)

        self._depth_frame = None
        self._color_frame = None
        self._camera = None

        if self._viewer:
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window()

    def __del__(self):
        if hasattr(self, '_pipeline') and self._pipeline is not None:
            self._pipeline.stop()
        if hasattr(self, '_device') and self._device is not None:
            self._device.hardware_reset()
        if hasattr(self, '_vis') and self._vis is not None:
            self._vis.destroy_window()

    @property
    def depth_intrinsics(self):
        if self._depth_frame is None:
            return rs.video_stream_profile(self._pipeline_profile.get_stream(rs.stream.depth)).get_intrinsics()
        else:
            return self._depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
    
    @property
    def color_intrinsics(self):
        if self._color_frame is None:
            return rs.video_stream_profile(self._pipeline_profile.get_stream(rs.stream.color)).get_intrinsics()
        else:
            return self._color_frame.get_profile().as_video_stream_profile().get_intrinsics()
    @property
    def depth_scale(self):
        return self._pipeline_profile.get_device().first_depth_sensor().get_depth_scale()

    def run(self):
        frame_count = 0
        pcd = o3d.geometry.PointCloud()
        while True:
            ret, frames = self._pipeline.try_wait_for_frames(100)
            if ret is True:
                aligned_frames = self._align.process(frames)
                self._depth_frame = aligned_frames.get_depth_frame()
                self._color_frame = aligned_frames.get_color_frame()

                depth_image = np.asanyarray(self._depth_frame.get_data())
                depth_image[depth_image > 3.0 / self.depth_scale] = 0
                color_image = np.asanyarray(self._color_frame.get_data())[...,::-1]

                if self._camera is None:
                    self._camera = pointfusion.Camera.from_rs2(self.depth_intrinsics, self.depth_scale)

                points, colors = self._camera.depth_to_cloud(depth_image, color_image)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

                if self._viewer is True:
                    if frame_count == 0:
                        self._vis.add_geometry(pcd)
                    self._vis.update_geometry(pcd)
                    self._vis.poll_events()
                    self._vis.update_renderer()
                    frame_count += 1
                else:
                    yield depth_image, color_image

if __name__ == '__main__':
    d455 = D455()
    d455.run()