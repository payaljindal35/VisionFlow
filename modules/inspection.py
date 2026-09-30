"""
Inspection Mode and Color Identification module for VisionFlow (Milestone 8).
Enables fingertip-based object targeting and dominant color extraction
using index fingertip coordinates from MediaPipe and YOLOv8n bounding boxes.
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np

from visionflow.config import (
    FINGERTIP_SAMPLE_RADIUS,
    GESTURE_POINT,
)


class InspectionProcessor:
    """
    Coordinates fingertip targeting and real-time color extraction.
    When the user points with the index finger at a detected object:
    - Identifies the targeted YOLO object.
    - Samples a small neighborhood around the fingertip.
    - Classifies the dominant basic color (RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, BLACK, WHITE, GRAY).
    """

    def __init__(
        self,
        sample_radius: int = FINGERTIP_SAMPLE_RADIUS,
        tolerance: int = 15,
    ):
        """
        Args:
            sample_radius (int): Radius around the fingertip pixel for color sampling.
            tolerance (int): Pixel margin around bounding boxes to accommodate border pointing.
        """
        self.sample_radius = sample_radius
        self.tolerance = tolerance

    def classify_color(self, patch: np.ndarray) -> str:
        """
        Determines the dominant basic color of an image patch using HSV thresholding.

        Args:
            patch (numpy.ndarray): Small OpenCV BGR image patch.

        Returns:
            str: One of 'RED', 'GREEN', 'BLUE', 'YELLOW', 'ORANGE', 'PURPLE',
                 'BLACK', 'WHITE', 'GRAY', or 'UNKNOWN'.
        """
        if patch is None or not isinstance(patch, np.ndarray) or patch.size == 0:
            return "UNKNOWN"

        # Compute average BGR to suppress noise and sensor grain
        mean_bgr = np.mean(patch, axis=(0, 1))
        patch_pixel = np.uint8([[mean_bgr]])
        hsv = cv2.cvtColor(patch_pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        # Step 1: Achromatic colors based on Value and Saturation
        if v < 50:
            return "BLACK"
        if s < 40:
            return "WHITE" if v > 200 else "GRAY"

        # Step 2: Chromatic colors based on Hue (0 to 179 in OpenCV)
        if h < 10 or h >= 170:
            return "RED"
        elif 10 <= h < 25:
            return "ORANGE"
        elif 25 <= h < 35:
            return "YELLOW"
        elif 35 <= h < 85:
            return "GREEN"
        elif 85 <= h < 130:
            return "BLUE"
        elif 130 <= h < 170:
            return "PURPLE"

        return "UNKNOWN"

    def sample_color_at_point(
        self, frame: np.ndarray, point: Optional[Tuple[int, int]]
    ) -> str:
        """
        Extracts a localized patch around the given point and classifies its color.

        Args:
            frame (numpy.ndarray): Raw or processed OpenCV BGR image.
            point (tuple): (x, y) pixel coordinates.

        Returns:
            str: Identified color string or 'UNKNOWN'.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or point is None:
            return "UNKNOWN"

        h, w = frame.shape[:2]
        ix, iy = int(point[0]), int(point[1])

        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return "UNKNOWN"

        r = self.sample_radius
        y1 = max(0, iy - r)
        y2 = min(h, iy + r + 1)
        x1 = max(0, ix - r)
        x2 = min(w, ix + r + 1)

        patch = frame[y1:y2, x1:x2]
        return self.classify_color(patch)

    def find_target_object(
        self, detections: List[Dict[str, Any]], fingertip: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Finds the detected object whose bounding box encloses or is closest to the fingertip.

        Args:
            detections (list): Output list from ObjectDetector.
            fingertip (tuple): (x, y) index fingertip pixel coordinates.

        Returns:
            dict or None: Targeted detection object if located, else None.
        """
        if not detections or fingertip is None:
            return None

        ix, iy = fingertip
        tol = self.tolerance
        candidates = []

        for det in detections:
            bbox = det.get("bbox") or det.get("box")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            if (x1 - tol <= ix <= x2 + tol) and (y1 - tol <= iy <= y2 + tol):
                # Calculate bounding box area
                area = max(0, x2 - x1) * max(0, y2 - y1)
                candidates.append((area, det))

        if not candidates:
            return None

        # Sort by area ascending so user selects the most localized/specific object
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def inspect(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fingertip: Optional[Tuple[int, int]],
        gesture: str,
    ) -> Dict[str, Any]:
        """
        Executes full inspection workflow:
        Checks for POINT gesture, resolves targeted object, and extracts dominant color.

        Args:
            frame: OpenCV BGR image frame.
            detections: List of detected objects.
            fingertip: (x, y) coordinates of index fingertip.
            gesture: Active hand gesture (e.g. 'POINT', 'PINCH', 'NONE').

        Returns:
            dict:
                {
                    "target_found": bool,
                    "object_name": Optional[str],
                    "color": str,
                    "confidence": Optional[float],
                    "fingertip": Optional[Tuple[int, int]],
                    "bbox": Optional[Tuple[int, int, int, int]],
                }
        """
        empty_result = {
            "target_found": False,
            "object_name": None,
            "color": "UNKNOWN",
            "confidence": None,
            "fingertip": fingertip,
            "bbox": None,
        }

        if frame is None or fingertip is None or gesture != GESTURE_POINT:
            return empty_result

        target = self.find_target_object(detections, fingertip)
        color = self.sample_color_at_point(frame, fingertip)

        if target is None:
            return {
                "target_found": False,
                "object_name": None,
                "color": color,
                "confidence": None,
                "fingertip": fingertip,
                "bbox": None,
            }

        return {
            "target_found": True,
            "object_name": target.get("class_name", "object"),
            "color": color,
            "confidence": target.get("confidence", 0.0),
            "fingertip": fingertip,
            "bbox": target.get("bbox"),
        }
