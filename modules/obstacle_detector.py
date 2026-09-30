"""
Obstacle Detection module for VisionFlow (Milestone 3).
Analyzes YOLOv8 detections to identify potential physical obstacles,
computes their spatial zone (LEFT, CENTER, RIGHT), estimates proximity
(FAR, MEDIUM, NEAR) based on bounding-box area ratio, and produces navigation warnings.
"""

from typing import List, Dict, Any, Tuple
from visionflow.config import (
    OBSTACLE_CLASSES,
    ZONE_LEFT,
    ZONE_CENTER,
    ZONE_RIGHT,
    PROXIMITY_FAR,
    PROXIMITY_MEDIUM,
    PROXIMITY_NEAR,
    PROXIMITY_NEAR_THRESHOLD,
    PROXIMITY_MEDIUM_THRESHOLD,
    WARNING_OBSTACLE_AHEAD,
    WARNING_OBSTACLE_LEFT,
    WARNING_OBSTACLE_RIGHT,
    WARNING_PATH_CLEAR,
)


class ObstacleDetector:
    """
    Evaluates detected objects for obstacle navigation.
    Filters candidate physical objects, assigns horizontal zone,
    calculates proximity based on relative bounding-box area, and generates
    real-time path warnings.
    """

    def __init__(
        self,
        obstacle_classes=None,
        near_threshold: float = PROXIMITY_NEAR_THRESHOLD,
        medium_threshold: float = PROXIMITY_MEDIUM_THRESHOLD,
    ):
        """
        Initializes the obstacle detector.

        Args:
            obstacle_classes (iterable, optional): Set of class names to treat as obstacles.
            near_threshold (float): Bounding-box to frame area ratio for NEAR proximity.
            medium_threshold (float): Bounding-box to frame area ratio for MEDIUM proximity.
        """
        if obstacle_classes is not None:
            self.obstacle_classes = {cls.lower().strip() for cls in obstacle_classes}
        else:
            self.obstacle_classes = {cls.lower().strip() for cls in OBSTACLE_CLASSES}

        self.near_threshold = near_threshold
        self.medium_threshold = medium_threshold

    def is_obstacle(self, class_name: str) -> bool:
        """
        Checks if a detected object class is classified as a potential obstacle.

        Args:
            class_name (str): The label/class from object detection.

        Returns:
            bool: True if class is in the obstacle list, False otherwise.
        """
        if not class_name:
            return False
        return class_name.lower().strip() in self.obstacle_classes

    def calculate_zone(self, bbox: Tuple[int, int, int, int], frame_width: int) -> str:
        """
        Divides the horizontal frame into 3 equal zones:
        LEFT:   0 to (frame_width / 3)
        CENTER: (frame_width / 3) to (2 * frame_width / 3)
        RIGHT:  (2 * frame_width / 3) to frame_width

        Args:
            bbox: (x1, y1, x2, y2) bounding box.
            frame_width: Total width of the video frame.

        Returns:
            str: 'LEFT', 'CENTER', or 'RIGHT'
        """
        if frame_width <= 0:
            return ZONE_CENTER

        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0

        one_third = frame_width / 3.0
        two_thirds = 2.0 * frame_width / 3.0

        if center_x < one_third:
            return ZONE_LEFT
        elif center_x > two_thirds:
            return ZONE_RIGHT
        else:
            return ZONE_CENTER

    def calculate_proximity(
        self, bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int
    ) -> str:
        """
        Estimates obstacle proximity based on bounding-box area relative to frame area.

        Formula: area_ratio = (bbox_width * bbox_height) / (frame_width * frame_height)
        - area_ratio >= near_threshold   => 'NEAR'
        - area_ratio >= medium_threshold => 'MEDIUM'
        - otherwise                      => 'FAR'

        Args:
            bbox: (x1, y1, x2, y2) bounding box.
            frame_width: Total width of the video frame.
            frame_height: Total height of the video frame.

        Returns:
            str: 'FAR', 'MEDIUM', or 'NEAR'
        """
        frame_area = frame_width * frame_height
        if frame_area <= 0:
            return PROXIMITY_FAR

        x1, y1, x2, y2 = bbox
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        bbox_area = bbox_w * bbox_h

        area_ratio = bbox_area / float(frame_area)

        if area_ratio >= self.near_threshold:
            return PROXIMITY_NEAR
        elif area_ratio >= self.medium_threshold:
            return PROXIMITY_MEDIUM
        else:
            return PROXIMITY_FAR

    def generate_warning(self, obstacles: List[Dict[str, Any]]) -> str:
        """
        Determines the appropriate warning string based on proximity and zone of obstacles.

        Warning Rules:
        1. If any obstacle is NEAR in CENTER -> "WARNING: OBSTACLE AHEAD" (Highest priority)
        2. Else if any obstacle is NEAR on LEFT -> "WARNING: OBSTACLE LEFT"
        3. Else if any obstacle is NEAR on RIGHT -> "WARNING: OBSTACLE RIGHT"
        4. If no NEAR obstacles -> "PATH CLEAR"

        Args:
            obstacles: List of analyzed obstacle dictionaries.

        Returns:
            str: Warning message.
        """
        near_obstacles = [
            obs for obs in obstacles if obs.get("proximity") == PROXIMITY_NEAR
        ]

        if not near_obstacles:
            return WARNING_PATH_CLEAR

        # Priority 1: Center obstacle ahead
        for obs in near_obstacles:
            if obs.get("zone") == ZONE_CENTER:
                return WARNING_OBSTACLE_AHEAD

        # Priority 2: Left obstacle
        for obs in near_obstacles:
            if obs.get("zone") == ZONE_LEFT:
                return WARNING_OBSTACLE_LEFT

        # Priority 3: Right obstacle
        for obs in near_obstacles:
            if obs.get("zone") == ZONE_RIGHT:
                return WARNING_OBSTACLE_RIGHT

        return WARNING_PATH_CLEAR

    def process_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        """
        Filters detected objects for physical obstacles, calculates their zone and
        proximity, creates formatted obstacle labels, and determines the overall path warning.

        Args:
            detections: List of detection dicts from ObjectDetector.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            dict containing:
                - 'obstacles': List of processed obstacle dicts with zone, proximity, label.
                - 'warning': Navigation warning text ('PATH CLEAR' or 'WARNING: ...').
                - 'is_warning': Boolean flag indicating if an active hazard warning is present.
        """
        obstacles = []

        for det in detections:
            class_name = det.get("class_name", "")
            if not self.is_obstacle(class_name):
                continue

            bbox = det.get("bbox", (0, 0, 0, 0))
            zone = self.calculate_zone(bbox, frame_width)
            proximity = self.calculate_proximity(bbox, frame_width, frame_height)

            # Formatted label: e.g. "person | CENTER | NEAR"
            obstacle_label = f"{class_name} | {zone} | {proximity}"

            obstacles.append({
                "class_name": class_name,
                "confidence": det.get("confidence", 0.0),
                "bbox": bbox,
                "zone": zone,
                "proximity": proximity,
                "label": obstacle_label,
            })

        warning = self.generate_warning(obstacles)
        is_warning = warning != WARNING_PATH_CLEAR

        return {
            "obstacles": obstacles,
            "warning": warning,
            "is_warning": is_warning,
        }
