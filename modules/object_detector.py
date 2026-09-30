"""
Object Detection module for VisionFlow using YOLOv8n.
Provides real-time detection of common everyday objects and people.
"""

from typing import List, Dict, Any
from ultralytics import YOLO
from visionflow.config import YOLO_MODEL_NAME, CONFIDENCE_THRESHOLD


class ObjectDetector:
    """
    Object detector powered by YOLOv8n (nano).
    Accepts raw OpenCV BGR frames, runs inference, and returns detected objects
    with their class name, confidence score, and bounding box coordinates.
    """

    def __init__(self, model_name: str = YOLO_MODEL_NAME, conf_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Initializes the YOLO object detector.
        
        Args:
            model_name (str): Model name or path (defaults to 'yolov8n.pt').
            conf_threshold (float): Minimum confidence threshold for filtering detections.
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        print(f"[INFO] Loading YOLO model '{self.model_name}'...")
        # Load YOLOv8n (nano model, ~6MB weights)
        self.model = YOLO(self.model_name)
        print("[INFO] YOLO model loaded successfully.")

    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Runs object detection on an OpenCV frame.

        Args:
            frame (numpy.ndarray): Input frame in BGR format.

        Returns:
            list of dict: Detected objects, where each dict has:
                - 'class_name' (str): Name of detected class (e.g. 'person', 'chair', 'cell phone')
                - 'confidence' (float): Detection confidence score (between 0.0 and 1.0)
                - 'bbox' (tuple): Coordinates (x1, y1, x2, y2) as integers
        """
        if frame is None:
            return []

        # Run inference. verbose=False prevents console spam on every frame
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        if not results:
            return detections

        # Extract predictions from the first result (single frame)
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        names = result.names  # Mapping from class ID to label string

        for box in boxes:
            # Extract bounding box coordinates [x1, y1, x2, y2]
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            # Extract confidence and class ID
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = names.get(cls_id, f"class_{cls_id}")

            detections.append({
                "class_name": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        return detections
