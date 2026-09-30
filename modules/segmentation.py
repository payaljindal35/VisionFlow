"""
Visual Segmentation and Spotlight Module for VisionFlow (Milestone 7).
Uses Google MediaPipe ImageSegmenter (selfie_segmenter.tflite) to segment the user
from the background and apply a focused spotlight visual effect.
"""

import os
import urllib.request
from typing import Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from visionflow.config import (
    SEGMENTATION_ENABLED,
    SEGMENTATION_MODEL_NAME,
    SEGMENTATION_MODEL_URL,
    SPOTLIGHT_ALPHA,
    SPOTLIGHT_BLUR,
)


class SegmentationProcessor:
    """
    Extracts person segmentation masks and renders spotlight visualization effects.
    Keeps the foreground person crisp while dimming and softly blurring the background.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        enabled: bool = SEGMENTATION_ENABLED,
        alpha: float = SPOTLIGHT_ALPHA,
        blur: bool = SPOTLIGHT_BLUR,
    ):
        """
        Initializes the segmentation processor with MediaPipe ImageSegmenter.

        Args:
            model_path (str, optional): Path to the selfie_segmenter.tflite model file.
            enabled (bool): Whether segmentation processing is active.
            alpha (float): Brightness multiplier for the background in spotlight mode.
            blur (bool): Whether to apply subtle Gaussian blur to the background.
        """
        self.model_path = model_path if model_path is not None else SEGMENTATION_MODEL_NAME
        self.enabled = enabled
        self.alpha = alpha
        self.blur = blur
        self.segmenter = None

        if not self.enabled:
            return

        # Ensure model weights file exists locally
        if not os.path.exists(self.model_path):
            print(f"[INFO] Segmentation model not found. Downloading to '{self.model_path}'...")
            urllib.request.urlretrieve(SEGMENTATION_MODEL_URL, self.model_path)
            print("[INFO] Segmentation model downloaded successfully.")

        try:
            print(f"[INFO] Initializing MediaPipe ImageSegmenter (model='{self.model_path}')...")
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=True,
                output_confidence_masks=True,
            )
            self.segmenter = vision.ImageSegmenter.create_from_options(options)
            print("[INFO] MediaPipe ImageSegmenter initialized successfully.")
        except Exception as e:
            print(f"[WARNING] ImageSegmenter initialization failed: {e}")
            self.segmenter = None

    def generate_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Generates a 2D binary segmentation mask where 255 represents foreground person
        and 0 represents background.

        Args:
            frame (numpy.ndarray): OpenCV BGR input frame.

        Returns:
            numpy.ndarray or None: 2D uint8 binary mask of shape (H, W).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        if self.segmenter is None or not self.enabled:
            return None

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            results = self.segmenter.segment(mp_image)
        except Exception as e:
            print(f"[WARNING] Segmentation inference error: {e}")
            return None

        if not results:
            return None

        # Priority 1: Use confidence mask if available (prob > 0.5)
        if results.confidence_masks and len(results.confidence_masks) > 0:
            conf = results.confidence_masks[0].numpy_view()
            if len(conf.shape) == 3:
                conf = conf[:, :, 0]
            mask = (conf > 0.5).astype(np.uint8) * 255
            return mask

        # Priority 2: Use category mask (category 0 is foreground person in selfie segmenter)
        if results.category_mask is not None:
            cat = results.category_mask.numpy_view()
            if len(cat.shape) == 3:
                cat = cat[:, :, 0]
            mask = (cat == 0).astype(np.uint8) * 255
            return mask

        return None

    def apply_spotlight(
        self, frame: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Applies spotlight visual effect: foreground remains bright, background is dimmed and blurred.

        Args:
            frame (numpy.ndarray): OpenCV BGR input frame.
            mask (numpy.ndarray, optional): Precomputed binary segmentation mask.

        Returns:
            numpy.ndarray: Processed frame with spotlight effect.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        if not self.enabled:
            return frame.copy()

        # Generate mask if not provided
        if mask is None:
            mask = self.generate_mask(frame)

        if mask is None:
            return frame.copy()

        # Ensure mask is 2D and matches frame dimensions
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create dimmed background
        dark_bg = (frame.astype(np.float32) * self.alpha).astype(np.uint8)

        # Optional soft Gaussian blur on background for depth-of-field effect
        if self.blur:
            dark_bg = cv2.GaussianBlur(dark_bg, (15, 15), 0)

        # Blend: Keep foreground original, replace background with dark_bg
        spotlight_frame = np.where(mask[:, :, None] > 0, frame, dark_bg)
        return spotlight_frame
