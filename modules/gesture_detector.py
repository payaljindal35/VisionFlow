"""
Gesture Detection module for VisionFlow (Milestone 5).
Classifies basic hand gestures (PINCH, POINT, OPEN_PALM, NONE)
using distance and finger-extension geometric rules.
"""

import math
from typing import Dict, Any, Tuple
from visionflow.config import (
    GESTURE_PINCH,
    GESTURE_POINT,
    GESTURE_OPEN_PALM,
    GESTURE_NONE,
    PINCH_DISTANCE_THRESHOLD,
)


class GestureDetector:
    """
    Evaluates 21 hand landmarks to identify core interaction gestures:
    - PINCH: Thumb tip and index tip in close proximity (used for click/select).
    - POINT: Index finger extended while middle, ring, pinky are curled.
    - OPEN_PALM: All four primary fingers extended.
    - NONE: Unclassified or relaxed pose.
    """

    def __init__(self, pinch_threshold: float = PINCH_DISTANCE_THRESHOLD):
        """
        Args:
            pinch_threshold (float): Maximum pixel distance between thumb and index tip for PINCH.
        """
        self.pinch_threshold = pinch_threshold

    @staticmethod
    def _euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculates 2D Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _is_finger_extended(
        self, pts: list, tip_idx: int, pip_idx: int, wrist_idx: int = 0
    ) -> bool:
        """
        A finger is considered extended if its fingertip is further from the wrist
        than its middle joint (PIP).
        """
        dist_tip_wrist = self._euclidean_dist(pts[tip_idx], pts[wrist_idx])
        dist_pip_wrist = self._euclidean_dist(pts[pip_idx], pts[wrist_idx])
        return dist_tip_wrist > dist_pip_wrist

    def classify_gesture(self, hand_data: Dict[str, Any]) -> str:
        """
        Determines the active gesture for a single hand.

        Args:
            hand_data (dict): Hand information containing 'pixel_landmarks', 'thumb_tip', 'index_tip'.

        Returns:
            str: 'PINCH', 'POINT', 'OPEN_PALM', or 'NONE'
        """
        if not hand_data:
            return GESTURE_NONE

        pts = hand_data.get("pixel_landmarks", [])
        if len(pts) < 21:
            return GESTURE_NONE

        thumb_tip = hand_data.get("thumb_tip", pts[4])
        index_tip = hand_data.get("index_tip", pts[8])

        # Rule 1: PINCH (Thumb tip and Index fingertip touch or are close)
        pinch_dist = self._euclidean_dist(thumb_tip, index_tip)
        if pinch_dist < self.pinch_threshold:
            return GESTURE_PINCH

        # Rule 2: Evaluate extension state of the 4 fingers
        index_extended = self._is_finger_extended(pts, tip_idx=8, pip_idx=6)
        middle_extended = self._is_finger_extended(pts, tip_idx=12, pip_idx=10)
        ring_extended = self._is_finger_extended(pts, tip_idx=16, pip_idx=14)
        pinky_extended = self._is_finger_extended(pts, tip_idx=20, pip_idx=18)

        # Rule 3: POINT (Index extended, others curled)
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return GESTURE_POINT

        # Rule 4: OPEN_PALM (Index, Middle, Ring, and Pinky all extended)
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return GESTURE_OPEN_PALM

        return GESTURE_NONE
