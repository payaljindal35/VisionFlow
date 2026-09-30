"""
Optical Character Recognition (OCR) module for VisionFlow (Milestone 4).
Uses EasyOCR to recognize printed or environmental text from camera frames.
"""

from typing import List, Dict, Any
import numpy as np
import easyocr
from visionflow.config import OCR_LANGUAGES, OCR_CONFIDENCE_THRESHOLD


class OCRReader:
    """
    OCR Reader powered by EasyOCR.
    Processes OpenCV BGR frames and extracts detected text strings,
    confidence probabilities, and bounding box coordinates.
    """

    def __init__(
        self,
        languages: list = None,
        gpu: bool = False,
        conf_threshold: float = OCR_CONFIDENCE_THRESHOLD,
    ):
        """
        Initializes EasyOCR reader.

        Args:
            languages (list, optional): List of language codes (defaults to ['en']).
            gpu (bool): Whether to use GPU acceleration (defaults to False for CPU).
            conf_threshold (float): Minimum confidence to keep a detection.
        """
        self.languages = languages if languages is not None else list(OCR_LANGUAGES)
        self.gpu = gpu
        self.conf_threshold = conf_threshold

        print(f"[INFO] Initializing EasyOCR Reader (languages={self.languages}, gpu={self.gpu})...")
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
        print("[INFO] EasyOCR Reader initialized successfully.")

    def read_text(self, frame) -> List[Dict[str, Any]]:
        """
        Extracts text from a given frame.

        Args:
            frame: OpenCV image frame (numpy.ndarray) in BGR format.

        Returns:
            list of dict: List of recognized text items:
                [
                    {
                        "text": str,
                        "confidence": float,
                        "bbox": (x1, y1, x2, y2)
                    },
                    ...
                ]
        """
        # Safely handle None or invalid frames
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        try:
            # EasyOCR accepts numpy array images directly
            raw_results = self.reader.readtext(frame)
        except Exception as e:
            print(f"[WARNING] EasyOCR inference error: {e}")
            return []

        parsed_results = []
        for item in raw_results:
            # raw_results format: (bbox_corners, text, confidence)
            # bbox_corners: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            if len(item) < 3:
                continue

            bbox_pts, text, prob = item[0], item[1], item[2]
            confidence = float(prob)

            # Filter by confidence threshold
            if confidence < self.conf_threshold:
                continue

            text_str = str(text).strip()
            if not text_str:
                continue

            # Convert 4 corner points to standard bounding box (x1, y1, x2, y2)
            try:
                xs = [int(pt[0]) for pt in bbox_pts]
                ys = [int(pt[1]) for pt in bbox_pts]
                x1, y1 = max(0, min(xs)), max(0, min(ys))
                x2, y2 = max(xs), max(ys)
            except Exception:
                continue

            parsed_results.append({
                "text": text_str,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            })

        return parsed_results
