"""
Hand Tracking module for VisionFlow (Milestone 5).
Uses Google MediaPipe HandLandmarker to track 21 hand joints in real time,
identifying key interaction landmarks (index fingertip, thumb tip, wrist).
"""

import os
import urllib.request
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from visionflow.config import (
    HAND_MAX_NUM_HANDS,
    HAND_MODEL_NAME,
    HAND_MODEL_URL,
    HAND_LANDMARK_COLOR,
    HAND_CONNECTION_COLOR,
    CURSOR_COLOR,
    CURSOR_RADIUS,
)

# Standard 21 MediaPipe hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
]


class HandTracker:
    """
    Detects hands and extracts 21 skeletal landmarks using MediaPipe.
    Provides pixel coordinates for interaction fingertips.
    """

    def __init__(self, model_path: str = None, max_num_hands: int = HAND_MAX_NUM_HANDS):
        """
        Initializes the MediaPipe HandLandmarker.

        Args:
            model_path (str, optional): Path to hand_landmarker.task model.
            max_num_hands (int): Maximum number of hands to detect.
        """
        self.model_path = model_path if model_path is not None else HAND_MODEL_NAME
        self.max_num_hands = max_num_hands

        # Ensure model weights file exists locally, download if missing
        if not os.path.exists(self.model_path):
            print(f"[INFO] HandLandmarker model not found. Downloading to '{self.model_path}'...")
            urllib.request.urlretrieve(HAND_MODEL_URL, self.model_path)
            print("[INFO] HandLandmarker model downloaded successfully.")

        print(f"[INFO] Initializing MediaPipe HandLandmarker (model='{self.model_path}')...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.max_num_hands,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        print("[INFO] MediaPipe HandLandmarker initialized successfully.")

    def detect_hands(self, frame) -> List[Dict[str, Any]]:
        """
        Detects hands in an OpenCV BGR frame.

        Args:
            frame: OpenCV BGR image frame (numpy.ndarray).

        Returns:
            list of dict: Hand detection data for each detected hand:
                {
                    "pixel_landmarks": [(x, y), ... 21 points],
                    "wrist": (x, y),
                    "thumb_tip": (x, y),
                    "index_tip": (x, y),
                    "middle_tip": (x, y),
                    "ring_tip": (x, y),
                    "pinky_tip": (x, y),
                }
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        # MediaPipe expects RGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            results = self.detector.detect(mp_image)
        except Exception as e:
            print(f"[WARNING] HandLandmarker inference error: {e}")
            return []

        hands_data = []
        if not results or not results.hand_landmarks:
            return hands_data

        for hand_lms in results.hand_landmarks:
            if len(hand_lms) < 21:
                continue

            # Convert normalized coordinates [0.0, 1.0] to pixel coordinates [0, width/height]
            pixel_landmarks = []
            for lm in hand_lms:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pixel_landmarks.append((px, py))

            hand_info = {
                "pixel_landmarks": pixel_landmarks,
                "wrist": pixel_landmarks[0],
                "thumb_tip": pixel_landmarks[4],
                "index_tip": pixel_landmarks[8],
                "middle_tip": pixel_landmarks[12],
                "ring_tip": pixel_landmarks[16],
                "pinky_tip": pixel_landmarks[20],
            }
            hands_data.append(hand_info)

        return hands_data

    def draw_landmarks(self, frame, hands_data: List[Dict[str, Any]]) -> None:
        """
        Draws hand skeleton connections, landmark points, and index fingertip cursor.

        Args:
            frame (numpy.ndarray): Frame to draw onto in-place.
            hands_data (list): List of detected hand dictionaries.
        """
        if frame is None or not hands_data:
            return

        for hand in hands_data:
            pts = hand.get("pixel_landmarks", [])
            if len(pts) < 21:
                continue

            # Draw connection lines
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = pts[start_idx]
                p2 = pts[end_idx]
                cv2.line(frame, p1, p2, HAND_CONNECTION_COLOR, 2, cv2.LINE_AA)

            # Draw small circles on all landmarks
            for pt in pts:
                cv2.circle(frame, pt, 3, HAND_LANDMARK_COLOR, -1, cv2.LINE_AA)

            # Highlight index fingertip as the primary interaction cursor
            index_tip = hand.get("index_tip")
            if index_tip:
                cv2.circle(frame, index_tip, CURSOR_RADIUS, CURSOR_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, index_tip, CURSOR_RADIUS + 3, (0, 0, 0), 1, cv2.LINE_AA)
