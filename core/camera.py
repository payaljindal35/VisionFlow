"""
Webcam management module for VisionFlow.
Handles camera opening, frame capture, and clean resource release.
"""

import cv2
from visionflow.config import DEFAULT_CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    """
    Wrapper around OpenCV VideoCapture for webcam management.
    Provides clean start, read, and release interfaces.
    """

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def start(self) -> bool:
        """
        Initializes and opens the webcam capture device.
        Tries DirectShow (CAP_DSHOW) on Windows for faster startup,
        with a fallback to default backend.
        
        Returns:
            bool: True if camera opened successfully, False otherwise.
        """
        # Attempt opening with DirectShow backend for faster startup on Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        # Fallback to default backend if DirectShow did not open
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            return False

        # Set frame dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def is_opened(self) -> bool:
        """Checks if the webcam is currently open and accessible."""
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        """
        Captures a single frame from the webcam.
        
        Returns:
            numpy.ndarray: The captured BGR image frame, or None if failed.
        """
        if not self.is_opened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        """Releases the camera hardware device cleanly."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
