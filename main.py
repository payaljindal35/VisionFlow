"""
Main application loop for VisionFlow.
Orchestrates webcam capture, frame processing, display window, and exit handling.
"""

import sys
import cv2
from visionflow.config import WINDOW_NAME, DEFAULT_CAMERA_INDEX
from visionflow.core.camera import Camera
from visionflow.core.state import AppState
from visionflow.core.pipeline import Pipeline


def main():
    """Starts the VisionFlow real-time system."""
    print("=" * 70)
    print("  VisionFlow — Assistive Real-Time Computer Vision System")
    print("  Active: Objects | Obstacles | OCR | Gestures & HUD | Audio | Spotlight | Inspect")
    print("=" * 70)

    # 1. Initialize State
    print("[INFO] Initializing application state...")
    state = AppState()

    # 2. Connect to Camera
    print(f"[INFO] Connecting to webcam (device index {DEFAULT_CAMERA_INDEX})...")
    camera = Camera(camera_index=DEFAULT_CAMERA_INDEX)

    if not camera.start():
        print(f"[ERROR] Could not access webcam at index {DEFAULT_CAMERA_INDEX}.")
        print("[HINT] Please ensure your camera is connected and not being used by another application.")
        return 1

    print("[INFO] Webcam connected successfully.")

    # 3. Initialize Pipeline
    print("[INFO] Setting up frame processing pipeline...")
    pipeline = Pipeline(state=state)

    print("[INFO] Starting live video display.")
    print("[INFO] Focus the video window and press 'Q' to exit cleanly.\n")

    try:
        while state.is_running:
            # Step A: Capture frame from camera
            frame = camera.read_frame()
            if frame is None:
                print("[WARNING] Could not read frame from webcam. Ending stream.")
                break

            # Step B: Process frame through pipeline
            processed_frame = pipeline.process_frame(frame)

            # Step C: Display live feed in OpenCV window
            cv2.imshow(WINDOW_NAME, processed_frame)

            # Step D: Check for 'Q' key to exit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("[INFO] 'Q' pressed by user. Exiting VisionFlow...")
                state.stop()
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C). Exiting...")
    finally:
        # Step E: Release camera hardware and close GUI window cleanly
        print("[INFO] Releasing webcam hardware...")
        camera.release()
        print("[INFO] Closing OpenCV display windows...")
        cv2.destroyAllWindows()
        if 'pipeline' in locals() and hasattr(pipeline, "speech_manager") and pipeline.speech_manager:
            print("[INFO] Stopping speech manager thread...")
            pipeline.speech_manager.stop()
        print("[INFO] VisionFlow shut down cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
