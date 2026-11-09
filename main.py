
"""
Main application loop for VisionFlow.
Supports camera controls and real-time computer vision processing.
"""

import sys
import cv2

from visionflow.config import WINDOW_NAME, DEFAULT_CAMERA_INDEX
from visionflow.core.camera import Camera
from visionflow.core.state import AppState
from visionflow.core.pipeline import Pipeline


RESOLUTIONS = {
    ord("1"): (640, 480),
    ord("2"): (1280, 720),
    ord("3"): (1920, 1080),
}


def main():
    """Starts the VisionFlow real-time system."""

    print("=" * 65)
    print("VISIONFLOW - ASSISTIVE COMPUTER VISION SYSTEM")
    print("=" * 65)

    state = AppState()

    # Discover available cameras
    print("[INFO] Discovering available cameras...")
    available_cameras = Camera.discover_cameras()

    if not available_cameras:
        print("[ERROR] No cameras detected.")
        return 1

    print(f"[INFO] Available cameras: {available_cameras}")

    initial_camera = (
        DEFAULT_CAMERA_INDEX
        if DEFAULT_CAMERA_INDEX in available_cameras
        else available_cameras[0]
    )

    camera = Camera(camera_index=initial_camera)

    if not camera.start():
        print("[ERROR] Could not initialize the webcam.")
        return 1

    print(f"[INFO] Camera {initial_camera} connected.")

    pipeline = None

    try:
        print("[INFO] Initializing processing pipeline...")
        pipeline = Pipeline(state=state)

        print("\n[INFO] Camera controls:")
        print(" C - Switch camera")
        print(" M - Toggle mirror")
        print(" 1 - 640x480")
        print(" 2 - 1280x720")
        print(" 3 - 1920x1080")

        print("\n[INFO] OCR controls:")
        print(" O - Single OCR scan")
        print(" T - Toggle continuous OCR")
        print(" K - Copy recognized text")
        print(" S - Save recognized text")
        print(" V - Read recognized text aloud")

        print("\n Q - Quit\n")

        camera_position = available_cameras.index(initial_camera)

        while state.is_running:

            frame = camera.read_frame()

            if frame is None:
                print("[WARNING] Unable to read camera frame.")
                break

            processed_frame = pipeline.process_frame(frame)

            cv2.imshow(WINDOW_NAME, processed_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print("[INFO] Exiting VisionFlow...")
                state.stop()
                break

            elif key in (ord("o"), ord("O")):
                pipeline.request_ocr_scan()
                print("[INFO] Single OCR scan requested.")

            elif key in (ord("t"), ord("T")):
                continuous = pipeline.toggle_ocr_continuous()

                print(
                    f"[INFO] Continuous OCR: "
                    f"{'ON' if continuous else 'OFF'}"
                )

            elif key in (ord("k"), ord("K")):
                if pipeline.copy_ocr_text():
                    print("[INFO] OCR text copied to clipboard.")
                else:
                    print("[WARNING] No OCR text available or clipboard failed.")

            elif key in (ord("s"), ord("S")):
                saved_path = pipeline.save_ocr_text()

                if saved_path:
                    print(f"[INFO] OCR text saved to: {saved_path}")
                else:
                    print("[WARNING] No OCR text available or saving failed.")

            elif key in (ord("v"), ord("V")):
                if pipeline.speak_ocr_text():
                    print("[INFO] Reading OCR text aloud.")
                else:
                    print("[WARNING] No OCR text available or speech unavailable.")
                    
            elif key in (ord("m"), ord("M")):
                mirror_state = camera.toggle_mirror()
                print(
                    f"[INFO] Mirror mode: "
                    f"{'ON' if mirror_state else 'OFF'}"
                )

            elif key in (ord("c"), ord("C")):

                if len(available_cameras) <= 1:
                    print("[INFO] No additional cameras available.")
                    continue

                next_position = (
                    camera_position + 1
                ) % len(available_cameras)

                next_camera = available_cameras[next_position]

                if camera.switch_camera(next_camera):
                    camera_position = next_position
                    print(
                        f"[INFO] Switched to camera {next_camera}"
                    )
                else:
                    print(
                        f"[ERROR] Could not switch to camera "
                        f"{next_camera}. Previous camera restored."
                    )

            elif key in RESOLUTIONS:

                width, height = RESOLUTIONS[key]
                camera.set_resolution(width, height)

                print(
                    f"[INFO] Requested resolution: "
                    f"{width}x{height}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        print("[INFO] Releasing camera...")
        camera.release()

        cv2.destroyAllWindows()

        if (
            pipeline is not None
            and hasattr(pipeline, "speech_manager")
            and pipeline.speech_manager
        ):
            pipeline.speech_manager.stop()

        print("[INFO] VisionFlow shut down cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
