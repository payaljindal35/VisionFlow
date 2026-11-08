
"""
Webcam management module for VisionFlow.
Supports camera discovery, switching, resolution changes and mirroring.
"""

import cv2
from visionflow.config import DEFAULT_CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    """Manages webcam initialization, capture and configuration."""

    def __init__(
        self,
        camera_index=DEFAULT_CAMERA_INDEX,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.mirror = False

    @staticmethod
    def discover_cameras(max_index=5):
        """Return indices of cameras that can be opened."""
        available = []

        for index in range(max_index):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)

            if cap.isOpened():
                available.append(index)

            cap.release()

        return available

    def start(self):
        """Open the selected camera."""
        self.release()

        self.cap = cv2.VideoCapture(
            self.camera_index, cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.release()
            return False

        self.set_resolution(self.width, self.height)
        return True

    def switch_camera(self, camera_index):
        """Switch to another camera, restoring the previous one on failure."""
        previous_index = self.camera_index

        self.release()
        self.camera_index = camera_index

        if self.start():
            return True

        self.camera_index = previous_index
        self.start()
        return False

    def set_resolution(self, width, height):
        """Request a new camera resolution."""
        self.width = width
        self.height = height

        if self.is_opened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def toggle_mirror(self):
        """Toggle horizontal mirroring of captured frames."""
        self.mirror = not self.mirror
        return self.mirror

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        """Capture a frame and apply mirroring if enabled."""
        if not self.is_opened():
            return None

        success, frame = self.cap.read()

        if not success:
            return None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return frame

    def release(self):
        """Release camera hardware."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
