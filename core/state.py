"""
State management for VisionFlow.
Tracks runtime status, active mode, and lifecycle flags.
"""


class AppState:
    """
    Represents the real-time application state.
    Allows different components of VisionFlow to read and update system status.
    """

    def __init__(self):
        self.is_running = True
        self.status_message = "VisionFlow | Press Q to Exit"
        self.current_mode = "LIVE"

    def set_status(self, message: str) -> None:
        """Update the status message shown on the video feed."""
        self.status_message = message

    def set_mode(self, mode: str) -> None:
        """Change system operating mode (e.g. LIVE, NAVIGATION, OCR)."""
        self.current_mode = mode

    def stop(self) -> None:
        """Signal the system to stop running and exit the main loop."""
        self.is_running = False
