"""
Configuration settings for VisionFlow Assistive Real-Time Computer Vision System.
Contains display parameters, camera configuration, model paths, and UI overlay styles.
"""

import cv2

# Window settings
WINDOW_NAME = "VisionFlow - Assistive Vision System"

# Camera settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Status overlay styling
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)        # Bright Green (BGR)
STATUS_BG_COLOR = (25, 25, 25)  # Dark Gray background for high readability
STATUS_PADDING = 8              # Padding around status box in pixels

# YOLO Model and detection settings (Milestone 2)
YOLO_MODEL_NAME = "yolov8n.pt"  # Nano model: fast, lightweight (~6 MB)
CONFIDENCE_THRESHOLD = 0.45      # Minimum detection confidence
BBOX_COLOR = (0, 255, 0)         # Green bounding box (BGR)
BBOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 1
LABEL_BG_COLOR = (0, 200, 0)     # Label background
LABEL_TEXT_COLOR = (0, 0, 0)     # Black text for high readability

# Navigation & Obstacle Detection settings (Milestone 3)
ZONE_LEFT = "LEFT"
ZONE_CENTER = "CENTER"
ZONE_RIGHT = "RIGHT"

PROXIMITY_FAR = "FAR"
PROXIMITY_MEDIUM = "MEDIUM"
PROXIMITY_NEAR = "NEAR"

# Area ratio thresholds (bounding box area / frame area)
PROXIMITY_NEAR_THRESHOLD = 0.15    # >= 15% frame area indicates close proximity
PROXIMITY_MEDIUM_THRESHOLD = 0.05  # >= 5% frame area indicates medium proximity

# Configurable list of physical obstacle classes (COCO classes)
OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "boat",
    "chair",
    "couch",
    "bed",
    "dining table",
    "backpack",
    "suitcase",
}

# Warning messages
WARNING_OBSTACLE_AHEAD = "WARNING: OBSTACLE AHEAD"
WARNING_OBSTACLE_LEFT = "WARNING: OBSTACLE LEFT"
WARNING_OBSTACLE_RIGHT = "WARNING: OBSTACLE RIGHT"
WARNING_PATH_CLEAR = "PATH CLEAR"

# Warning overlay colors
COLOR_SAFE = (0, 200, 0)          # Green
COLOR_WARNING = (0, 0, 255)       # Red
COLOR_OBSTACLE_TAG = (0, 140, 255) # Orange for obstacle specific tag

# OCR (Text Recognition) settings (Milestone 4)
OCR_LANGUAGES = ["en"]
OCR_CONFIDENCE_THRESHOLD = 0.35     # Minimum confidence to accept recognized text
OCR_INTERVAL_FRAMES = 15            # Process OCR once every 15 frames (~0.5s at 30 FPS)
OCR_BBOX_COLOR = (255, 180, 0)      # Sky blue / Cyan (BGR)
OCR_BBOX_THICKNESS = 2
OCR_LABEL_FONT_SCALE = 0.45
OCR_LABEL_FONT_THICKNESS = 1
OCR_LABEL_BG_COLOR = (180, 100, 0)  # Darker cyan/blue background for text tag
OCR_LABEL_TEXT_COLOR = (255, 255, 255) # White text
OCR_SUMMARY_BG_COLOR = (45, 30, 20) # Subtle dark slate background for OCR summary badge
OCR_SUMMARY_TEXT_COLOR = (0, 230, 255) # Yellow-gold text for high visibility

# Hand Tracking & Gesture Interaction settings (Milestone 5)
HAND_MAX_NUM_HANDS = 2
HAND_MODEL_NAME = "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Gestures
GESTURE_PINCH = "PINCH"
GESTURE_POINT = "POINT"
GESTURE_OPEN_PALM = "OPEN_PALM"
GESTURE_NONE = "NONE"

PINCH_DISTANCE_THRESHOLD = 45       # Max pixel distance between thumb and index tip for PINCH

# Visual Hand styling
HAND_LANDMARK_COLOR = (0, 255, 255)   # Yellow points for hand landmarks
HAND_CONNECTION_COLOR = (180, 180, 0) # Cyan-ish connection lines
CURSOR_COLOR = (0, 255, 255)          # Bright Yellow circle on index fingertip
CURSOR_RADIUS = 8

# Virtual HUD Buttons: (x, y, width, height)
HUD_BUTTONS = [
    {"name": "NAVIGATION", "mode": "navigation", "rect": (460, 30, 160, 42)},
    {"name": "OCR", "mode": "ocr", "rect": (460, 80, 160, 42)},
    {"name": "OBJECTS", "mode": "objects", "rect": (460, 130, 160, 42)},
    {"name": "SPOTLIGHT", "mode": "spotlight", "rect": (460, 180, 160, 42)},
    {"name": "INSPECT", "mode": "inspect", "rect": (460, 230, 160, 42)},
]

HUD_BTN_BG_COLOR = (40, 40, 40)          # Normal dark button background
HUD_BTN_BORDER_COLOR = (100, 100, 100)   # Normal border color
HUD_BTN_HOVER_COLOR = (0, 220, 255)      # Yellow/Gold highlight on hover
HUD_BTN_ACTIVE_COLOR = (0, 180, 0)       # Green highlight when mode is selected
HUD_BTN_TEXT_COLOR = (255, 255, 255)     # White text
HUD_STATUS_BG_COLOR = (30, 20, 50)       # Dark purple/navy for gesture status banner
HUD_STATUS_TEXT_COLOR = (255, 120, 255)  # Magenta text for gesture display

# Audio Feedback & Text-to-Speech settings (Milestone 6)
AUDIO_ENABLED = True
SPEECH_RATE = 160               # Words per minute (moderate, easily understandable)
SPEECH_VOLUME = 1.0             # Volume level (0.0 to 1.0)
SPEECH_COOLDOWN = 3.0           # Seconds before repeating identical obstacle warning
OCR_SPEECH_COOLDOWN = 6.0       # Seconds before repeating identical OCR text
PRIORITY_OBSTACLE = 1           # Highest priority (hazard warning)
PRIORITY_MODE = 2               # Medium priority (HUD interaction)
PRIORITY_OCR = 3                # Low priority (background text reading)

# Visual Segmentation & Spotlight Mode settings (Milestone 7)
SEGMENTATION_ENABLED = True
SEGMENTATION_MODEL_NAME = "selfie_segmenter.tflite"
SEGMENTATION_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
SPOTLIGHT_ALPHA = 0.3           # Background brightness multiplier (30% brightness)
SPOTLIGHT_BLUR = True           # Apply subtle Gaussian blur to background

# Inspection Mode & Color Identification settings (Milestone 8)
INSPECTION_ENABLED = True
INSPECTION_COOLDOWN = 3.0       # Seconds before repeating identical inspection announcement
FINGERTIP_SAMPLE_RADIUS = 8     # Pixel radius around fingertip for color sampling
TARGET_HIGHLIGHT_COLOR = (255, 0, 255) # Bright Magenta highlight for targeted object

# Statistics Overlay settings
STATS_ENABLED_BY_DEFAULT = False
STATS_BG_COLOR = (25, 25, 25)
STATS_TEXT_COLOR = (255, 255, 255)
STATS_TITLE_COLOR = (0, 220, 255)
STATS_FONT_SCALE = 0.45
STATS_FONT_THICKNESS = 1
STATS_PADDING = 8

