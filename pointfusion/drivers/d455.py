import pyrealsense2 as rs
import numpy as np
import cv2

import pointfusion

class D455:
    def __init__(self, width=1280, height=720, fps=30):
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

    def __del__(self):
        if hasattr(self, '_pipeline') and self._pipeline is not None:
            self._pipeline.stop()
        if hasattr(self, '_device') and self._device is not None:
            self._device.hardware_reset()

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
        while True:
            # Wait for a coherent pair of frames: depth and color
            ret, frames = self._pipeline.try_wait_for_frames(100)
            if ret is True:
                aligned_frames = self._align.process(frames)
                self._depth_frame = aligned_frames.get_depth_frame()
                self._color_frame = aligned_frames.get_color_frame()

                depth_image = np.asanyarray(self._depth_frame.get_data()).astype(np.float32)
                color_image = np.asanyarray(self._color_frame.get_data())
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
                # Show images
                cv2.imshow('RealSense', images)
                print(depth_image.shape)
                cv2.waitKey(1)

                if self._camera is None:
                    self._camera = pointfusion.Camera.from_rs2(self.depth_intrinsics, self.depth_scale)
                import pdb; pdb.set_trace()

                depth_cloud = self._camera.depth_to_cloud(depth_image)

                #yield depth_image, color_image

if __name__ == '__main__':
    d455 = D455()
    d455.run()