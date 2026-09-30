# VisionFlow — Assistive Real-Time Computer Vision System

VisionFlow is a real-time assistive computer-vision application developed in Python. It processes a live webcam stream to provide visual heads-up display (HUD) elements and offline spoken audio feedback, helping users understand their immediate environment, navigate around obstacles, read visible text, and interact touchlessly using hand gestures.

---

## Features

- **Webcam Capture**: Live video stream capture using OpenCV with clean hardware initialization and release.
- **YOLOv8n Real-Time Object Detection**: Detects common everyday objects and people with confidence scores and bounding boxes.
- **Navigation & Obstacle Detection**: Evaluates detected objects in the user's field of view to warn of potential navigation hazards.
- **LEFT / CENTER / RIGHT Spatial Zones**: Divides the camera view into three horizontal corridors to report where obstacles are located.
- **FAR / MEDIUM / NEAR Proximity Estimation**: Estimates obstacle distance based on the ratio of bounding-box area to frame area.
- **OCR Text Detection & Reading**: Offline text extraction powered by EasyOCR, executed periodically to maintain steady video rendering.
- **MediaPipe Hand Tracking**: Real-time 21-joint skeletal hand landmark tracking.
- **OPEN_PALM / PINCH / POINT / NONE Gestures**: Rule-based geometric gesture classifier for touchless interaction.
- **Virtual HUD Interaction**: On-screen buttons that highlight on fingertip hover and trigger on pinch gestures.
- **Offline Audio Feedback**: Non-blocking speech synthesis using Windows SAPI5 through `pyttsx3`, with priority queuing and cooldown debouncing.
- **Spotlight / Segmentation Mode**: Uses portrait segmentation to dim and blur the background by 70%, keeping the foreground person sharp and reducing visual clutter.
- **Object Inspection**: Point at any detected object with an index finger to identify the object.
- **Basic Color Identification**: Samples a pixel patch around the fingertip in HSV color space to classify dominant colors (Red, Green, Blue, Yellow, Orange, Purple, Black, White, Gray).
- **Mode Switching Through Virtual HUD**: Switch between system modes (Navigation, OCR, Objects, Spotlight, Inspect) directly via hand gestures.

---

## How It Works

VisionFlow follows a modular pipeline architecture that coordinates input capture, vision inference, user interface rendering, and speech feedback without blocking the main display loop:

```
[Webcam Capture (Camera)]
           │ (Raw BGR Frame)
           ▼
[Pipeline Manager (Pipeline)]
    ├── Background Segmentation (SegmentationProcessor)
    ├── Object Detection (ObjectDetector / YOLOv8n)
    ├── Obstacle & Corridor Analysis (ObstacleDetector)
    ├── Text Recognition (OCRReader / EasyOCR - Cached)
    ├── Skeletal Hand Tracking (HandTracker / MediaPipe)
    ├── Gesture Classification (GestureDetector)
    ├── Fingertip Inspection & Color Classification (InspectionProcessor)
    └── Virtual HUD & Mode Management (AppState)
           │
           ├──► [Visual Renderer] ──► OpenCV GUI Window
           └──► [Priority Speech Queue] ──► Offline TTS Worker Thread (SpeechManager)
```

### Core Components and Modules

- **`visionflow.core.camera.Camera`**: Manages the OpenCV `VideoCapture` device. Attempts fast DirectShow (`CAP_DSHOW`) startup on Windows with fallback to the default backend, and handles clean device release.
- **`visionflow.core.state.AppState`**: Thread-safe state container tracking the active operating mode (`navigation`, `ocr`, `objects`, `spotlight`, `inspect`) and lifecycle flags.
- **`visionflow.core.pipeline.Pipeline`**: The central processing pipeline that receives frames from the camera, invokes active perception modules, renders visual HUD overlays, and dispatches audio events.
- **`visionflow.modules.object_detector.ObjectDetector`**: Wraps Ultralytics YOLOv8n to perform real-time inference on CPU and extract bounding boxes, labels, and confidence scores.
- **`visionflow.modules.obstacle_detector.ObstacleDetector`**: Filters detected objects against a configured list of physical obstacles, maps their center coordinates to horizontal zones (`LEFT`, `CENTER`, `RIGHT`), and calculates area ratios for proximity (`FAR`, `MEDIUM`, `NEAR`).
- **`visionflow.modules.ocr_reader.OCRReader`**: Wraps EasyOCR to extract printed and environmental text. Runs on a periodic frame interval (every 15 frames) to avoid display lag.
- **`visionflow.modules.hand_tracker.HandTracker`**: Uses MediaPipe Tasks `HandLandmarker` to locate 21 skeletal hand landmarks in pixel space.
- **`visionflow.modules.gesture_detector.GestureDetector`**: Calculates Euclidean distances and joint extension angles to classify gestures into `PINCH`, `POINT`, `OPEN_PALM`, or `NONE`.
- **`visionflow.modules.segmentation.SegmentationProcessor`**: Uses MediaPipe Tasks `ImageSegmenter` to generate a binary foreground mask of the user, dimming the background to 30% brightness with a soft Gaussian blur.
- **`visionflow.modules.inspection.InspectionProcessor`**: Maps the index fingertip position under the `POINT` gesture to candidate YOLO bounding boxes (prioritizing smaller localized objects on overlap) and classifies the dominant median HSV color.
- **`visionflow.audio.speech.SpeechManager`**: Dispatches spoken audio through Windows SAPI5 (`pyttsx3`) using a dedicated background worker thread and a priority queue. Handles cooldown debouncing to suppress repetitive alerts.
- **`visionflow.main`**: The main application entry point orchestrating camera polling, pipeline processing, GUI window display, and keyboard exit handling.

---

## Project Structure

```
visionflow/
├── __init__.py              # Package initialization and module exports
├── config.py                # Central configuration constants, thresholds, and HUD layout
├── main.py                  # Main application loop and display window
├── audio/
│   ├── __init__.py          # Audio subpackage initialization
│   └── speech.py            # SpeechManager: Offline TTS with priority and debouncing
├── core/
│   ├── __init__.py          # Core subpackage initialization
│   ├── camera.py            # Camera capture wrapper with clean hardware release
│   ├── pipeline.py          # Unified frame processing pipeline and HUD renderer
│   └── state.py             # AppState managing system modes and lifecycle
└── modules/
    ├── __init__.py          # Module exports
    ├── gesture_detector.py  # Rule-based gesture classifier (PINCH, POINT, OPEN_PALM)
    ├── hand_tracker.py      # MediaPipe 21-joint skeletal hand tracking
    ├── inspection.py        # Point-to-inspect object targeting and HSV color classifier
    ├── object_detector.py   # YOLOv8n object detection wrapper
    ├── obstacle_detector.py # Spatial corridor and proximity obstacle analyzer
    ├── ocr_reader.py        # EasyOCR text recognition wrapper with periodic caching
    └── segmentation.py      # MediaPipe ImageSegmenter and Spotlight visual effect

tests/
├── __init__.py              # Test suite initialization
├── test_camera.py           # Camera interface unit tests
├── test_gesture_detector.py # Gesture classification unit tests
├── test_hand_tracker.py     # Hand tracking landmark unit tests
├── test_inspection.py       # Inspection and color identification unit tests
├── test_object_detector.py  # YOLOv8n detection unit tests
├── test_obstacle_detector.py# Obstacle corridor and proximity unit tests
├── test_ocr_reader.py       # OCR extraction and handling unit tests
├── test_pipeline.py         # Pipeline processing and mode transition tests
├── test_segmentation.py     # Person segmentation and spotlight unit tests
├── test_speech.py           # SpeechManager priority and debouncing tests
└── test_state.py            # AppState lifecycle unit tests

hand_landmarker.task         # MediaPipe HandLandmarker task model
selfie_segmenter.tflite      # MediaPipe ImageSegmenter model
yolov8n.pt                   # YOLOv8 nano model weights
final_verification.jpg       # Sample verification output image
requirements.txt             # Python dependencies
run.py                       # Application launcher script
verify_final.py              # Automated system verification script
README.md                    # Project documentation
```

---

## Requirements

- **Operating System**: Windows (verified on Windows 11)
- **Python Version**: Python 3.13.7 (verified environment; Python 3.8+ supported)
- **Hardware**: Standard integrated or USB webcam; audio output (speakers or headphones)
- **Execution Target**: CPU operation is fully supported and configured by default
- **Python Packages** (from `requirements.txt`):
  - `opencv-python>=4.8.0`
  - `numpy>=1.24.0`
  - `ultralytics>=8.0.0`
  - `easyocr>=1.7.0`
  - `mediapipe>=0.10.0`
  - `pyttsx3>=2.90`

---

## Installation

Follow these steps to set up and run VisionFlow:

### 1. Clone the Repository
```bash
git clone https://github.com/payaljindal35/crash-injury-risk-predictor.git
cd crash-injury-risk-predictor
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model Files
VisionFlow relies on three pre-trained model files located in the root project directory:
- **`yolov8n.pt`** (~6 MB): YOLOv8 nano model for object detection. Ultralytics automatically downloads this file on first run if it is not found locally.
- **`hand_landmarker.task`** (~7.8 MB): Google MediaPipe HandLandmarker model. Downloaded automatically by `HandTracker` on first run if missing.
- **`selfie_segmenter.tflite`** (~250 KB): Google MediaPipe SelfieSegmenter model. Downloaded automatically by `SegmentationProcessor` on first run if missing.

All three model files are already included locally in the repository root for immediate offline use.

---

## Running the Application

To start the live VisionFlow system:

```bash
python run.py
```

### What to Expect
1. A console banner confirms component initialization (AppState, Camera, YOLOv8n, EasyOCR, MediaPipe, SpeechManager).
2. The webcam feed opens in an OpenCV window titled `VisionFlow - Assistive Vision System`.
3. The visual HUD overlays appear on the feed:
   - Top status banner showing detected object count, current mode, and exit instructions.
   - Hazard warning banner (`PATH CLEAR` or directional warning).
   - OCR text summary badge.
   - Gesture interaction status badge.
   - Virtual interactive mode buttons on the right side of the screen.
4. Spoken audio confirmations play through your default Windows audio output device.
5. To exit, focus the video window and press the **`Q`** key. The webcam hardware, display window, and audio worker thread will shut down cleanly.

---

## Verification and Tests

### Automated System Verification
To verify camera capture, model loading, mode transitions, and clean shutdown without a long interactive webcam session, run:

```bash
python verify_final.py
```

This script verifies all 12 system capabilities sequentially and writes a sample output image to `final_verification.jpg`.

### Running the Full Test Suite
To execute all project unit tests:

```bash
python -m unittest discover -s tests -v
```

**Test Results**:
- Total Tests: **71**
- Passed: **71**
- Failed: **0**
- Skipped: **0**

---

## Controls & Gestures

VisionFlow tracks hand landmarks in real time and classifies geometric poses:

- **`POINT` Gesture**: Extend the index finger while keeping other fingers curled.
  - Acts as a virtual pointer. In `INSPECT` mode, point at any detected object to target it and hear its name and dominant color.
- **`PINCH` Gesture**: Bring the thumb tip and index fingertip close together (distance < 45 pixels).
  - Acts as a click action. Hover over any virtual HUD button and pinch to select that mode.
- **`OPEN_PALM` Gesture**: Extend all four primary fingers outward.
  - Recognized as an open hand state.
- **`NONE`**: Default relaxed or unclassified hand pose.
- **Keyboard `Q`**: Press `Q` while the display window is active to safely terminate the program.

---

## HUD Modes

The virtual HUD offers 5 operating modes selectable via touchless gestures:

1. **`NAVIGATION`**:
   - Focuses on spatial corridor awareness and obstacle avoidance.
   - Monitors `LEFT`, `CENTER`, and `RIGHT` zones.
   - Triggers red visual alerts and spoken warnings for near obstacles (`WARNING: OBSTACLE AHEAD`, etc.).
2. **`OCR`**:
   - Focuses on reading environmental text, signs, and labels.
   - Outlines text blocks in cyan and speaks recognized text aloud.
3. **`OBJECTS`**:
   - Displays bounding boxes, class labels, and confidence percentages for all recognized YOLO items.
4. **`SPOTLIGHT`**:
   - Activates portrait segmentation, dimming the background by 70% with soft blur while keeping the user clear.
5. **`INSPECT`**:
   - Enables point-to-inspect interaction. Point at an object to highlight it with a magenta box, identify its class, and determine its dominant color.

---

## Audio Feedback

- **Engine**: Completely offline Windows SAPI5 speech synthesis via `pyttsx3`. No internet connection or cloud service is required.
- **Asynchronous Execution**: Audio runs in a separate background daemon thread using a `PriorityQueue`, ensuring the video stream and computer vision pipeline never freeze during speech.
- **Event Priorities**:
  - **Priority 1 (Urgent)**: Near obstacle warnings (e.g. `"Obstacle ahead"`). Preempts lower-priority speech.
  - **Priority 2 (Medium)**: HUD mode change announcements and inspection results.
  - **Priority 3 (Low)**: Periodic background OCR text readings.
- **Debouncing**: Cooldown timers prevent repetitive chatter:
  - 3.0-second cooldown for identical obstacle alerts.
  - 6.0-second cooldown for identical OCR text.
  - 3.0-second cooldown for identical inspection announcements.
  - State-transition detection speaks `"Path clear"` once when moving away from a hazard.

---

## Performance & Limitations

- **CPU Framerate Under Combined Load**: Running YOLOv8n, MediaPipe HandLandmarker, and ImageSegmenter simultaneously on CPU achieves approximately 4–6 FPS. OCR is scheduled periodically (every 15 frames) to avoid dropping the display framerate.
- **Lighting Sensitivity**: HSV color classification relies on ambient lighting. Low-light environments, intense reflections, or harsh shadows may cause chromatic colors to be classified as Gray or Black.
- **Fingertip Occlusion**: When pointing at very small items, the user's hand may partially occlude the item, temporarily reducing YOLO detection confidence.
- **Proximity Approximation**: Proximity estimation is computed using the ratio of bounding-box area to frame area (>= 15% Near, >= 5% Medium, < 5% Far). It serves as a spatial heuristic rather than a physical distance measurement in millimeters.

---

## Future Improvements

The following items are potential areas for future enhancement:

- **Multi-threaded Inference Pipeline**: Decouple YOLO detection and segmentation into dedicated background worker processes to increase camera display framerate on CPU.
- **Stereo / Depth Camera Integration**: Incorporate hardware depth sensors (such as Intel RealSense) for metric distance measurements.
- **Multi-language OCR**: Extend EasyOCR language models beyond English for multilingual signage reading.
- **Adaptive Color Normalization**: Implement illumination-invariant color normalization to improve classification accuracy across changing ambient lighting conditions.
