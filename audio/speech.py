"""
Speech and Audio Feedback module for VisionFlow (Milestone 6).
Provides asynchronous, non-blocking offline text-to-speech using pyttsx3 (Windows SAPI5)
with debouncing, priority queuing, and category-specific cooldowns.
"""

import time
import queue
import threading
from typing import List, Dict, Any, Optional

from visionflow.config import (
    AUDIO_ENABLED,
    SPEECH_RATE,
    SPEECH_VOLUME,
    SPEECH_COOLDOWN,
    OCR_SPEECH_COOLDOWN,
    PRIORITY_OBSTACLE,
    PRIORITY_MODE,
    PRIORITY_OCR,
    INSPECTION_COOLDOWN,
    WARNING_OBSTACLE_AHEAD,
    WARNING_OBSTACLE_LEFT,
    WARNING_OBSTACLE_RIGHT,
    WARNING_PATH_CLEAR,
)


class SpeechManager:
    """
    Manages non-blocking speech output for VisionFlow.
    Features:
    - Runs speech synthesis in a background daemon thread so video frames never freeze.
    - Suppresses repeated identical warnings (cooldown debouncing).
    - Prioritizes urgent obstacle warnings over background OCR speech.
    - Speaks mode transitions only on actual selection.
    - Gracefully handles missing audio drivers or disabled state without crashing.
    """

    def __init__(
        self,
        enabled: bool = AUDIO_ENABLED,
        rate: int = SPEECH_RATE,
        volume: float = SPEECH_VOLUME,
        cooldown: float = SPEECH_COOLDOWN,
        ocr_cooldown: float = OCR_SPEECH_COOLDOWN,
        inspection_cooldown: float = INSPECTION_COOLDOWN,
        engine=None,
    ):
        """
        Initializes the SpeechManager.

        Args:
            enabled (bool): Whether audio speech is enabled.
            rate (int): Words per minute speech rate.
            volume (float): Speech volume between 0.0 and 1.0.
            cooldown (float): Seconds before repeating identical obstacle warnings.
            ocr_cooldown (float): Seconds before repeating identical OCR text.
            inspection_cooldown (float): Seconds before repeating identical inspection target text.
            engine: Optional mock or pre-initialized pyttsx3 engine (used for unit tests).
        """
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.cooldown = cooldown
        self.ocr_cooldown = ocr_cooldown
        self.inspection_cooldown = inspection_cooldown
        self.mock_engine = engine

        self.queue: queue.Queue = queue.Queue()
        self.is_running = True

        # State tracking for debouncing
        self.last_obstacle_warning: Optional[str] = None
        self.last_obstacle_time: float = 0.0
        self.had_obstacle_warning: bool = False

        self.last_ocr_text: Optional[str] = None
        self.last_ocr_time: float = 0.0

        self.last_inspection_text: Optional[str] = None
        self.last_inspection_time: float = 0.0

        self.last_mode: Optional[str] = None
        self.last_spoken_text: Optional[str] = None

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _clear_pending_queue(self) -> None:
        """Clears any pending low-priority messages so urgent alerts are spoken first."""
        with self.queue.mutex:
            self.queue.queue.clear()

    def speak(self, text: str, priority: int = PRIORITY_OCR) -> bool:
        """
        Enqueues text for asynchronous spoken output.

        Args:
            text (str): Message to speak.
            priority (int): 1=Obstacle (urgent), 2=Mode (medium), 3=OCR (low).

        Returns:
            bool: True if queued, False if discarded/disabled.
        """
        if not self.enabled:
            return False

        if text is None or not isinstance(text, str):
            return False

        clean_text = text.strip()
        if not clean_text:
            return False

        # Higher priority events preempt pending lower-priority messages
        if priority <= PRIORITY_MODE:
            self._clear_pending_queue()

        self.last_spoken_text = clean_text
        self.queue.put({"text": clean_text, "priority": priority})
        return True

    def speak_obstacle(self, warning_text: str) -> bool:
        """
        Speaks navigation warnings with debouncing and state-transition handling.
        - CENTER: 'Obstacle ahead'
        - LEFT: 'Obstacle left'
        - RIGHT: 'Obstacle right'
        - PATH CLEAR: Spoken once only when transitioning from an active hazard to clear.

        Args:
            warning_text (str): Warning string from ObstacleDetector.

        Returns:
            bool: True if speech was triggered, False if suppressed or clear.
        """
        if not self.enabled or not warning_text:
            return False

        # Map to concise, natural spoken phrases
        if warning_text == WARNING_OBSTACLE_AHEAD:
            phrase = "Obstacle ahead"
        elif warning_text == WARNING_OBSTACLE_LEFT:
            phrase = "Obstacle left"
        elif warning_text == WARNING_OBSTACLE_RIGHT:
            phrase = "Obstacle right"
        elif warning_text == WARNING_PATH_CLEAR:
            if self.had_obstacle_warning:
                # Transitioned from hazard to clear: speak once
                self.had_obstacle_warning = False
                self.last_obstacle_warning = None
                return self.speak("Path clear", priority=PRIORITY_OBSTACLE)
            return False
        else:
            return False

        self.had_obstacle_warning = True
        now = time.time()

        # Suppress identical warning if still within cooldown period
        if phrase == self.last_obstacle_warning and (now - self.last_obstacle_time < self.cooldown):
            return False

        self.last_obstacle_warning = phrase
        self.last_obstacle_time = now
        return self.speak(phrase, priority=PRIORITY_OBSTACLE)

    def speak_ocr(self, ocr_results: List[Dict[str, Any]]) -> bool:
        """
        Speaks recognized OCR text once, suppressing unchanged or repeating text.

        Args:
            ocr_results (list): Parsed OCR items containing 'text'.

        Returns:
            bool: True if speech was queued, False if empty/suppressed.
        """
        if not self.enabled or not ocr_results:
            return False

        texts = [
            item.get("text", "").strip()
            for item in ocr_results
            if isinstance(item, dict) and item.get("text") and item.get("text", "").strip()
        ]
        if not texts:
            return False

        # Combine top 3 text items and keep reasonably short
        combined = " ".join(texts[:3])
        if len(combined) > 80:
            combined = combined[:77] + "..."

        now = time.time()

        # Suppress if text is identical to last spoken OCR text and within cooldown
        if combined == self.last_ocr_text and (now - self.last_ocr_time < self.ocr_cooldown):
            return False

        self.last_ocr_text = combined
        self.last_ocr_time = now
        return self.speak(combined, priority=PRIORITY_OCR)

    def speak_mode(self, mode_name: str) -> bool:
        """
        Speaks confirmation when a HUD mode is selected/changed.
        Example: 'Navigation mode selected', 'OCR mode selected', 'Inspection mode selected'.

        Args:
            mode_name (str): The newly activated mode name.

        Returns:
            bool: True if mode change speech was queued, False if unchanged.
        """
        if not self.enabled or not mode_name or not isinstance(mode_name, str):
            return False

        clean_mode = mode_name.strip().lower()
        if clean_mode == self.last_mode:
            # Mode has not changed; do not repeat
            return False

        self.last_mode = clean_mode
        if clean_mode in ("inspect", "inspection"):
            phrase = "Inspection mode selected"
        else:
            phrase = f"{clean_mode.capitalize()} mode selected"
        return self.speak(phrase, priority=PRIORITY_MODE)

    def speak_inspection(self, object_name: str, color: Optional[str] = None) -> bool:
        """
        Speaks inspection target result with debouncing.
        E.g. 'Blue chair' or 'Chair'.

        Args:
            object_name (str): YOLO detected class name.
            color (str, optional): Identified dominant color or None.

        Returns:
            bool: True if speech was queued, False if suppressed or disabled.
        """
        if not self.enabled or not object_name:
            return False

        clean_obj = object_name.strip()
        clean_col = color.strip().lower() if color and color.strip().upper() != "UNKNOWN" else None

        if clean_col:
            phrase = f"{clean_col.capitalize()} {clean_obj}"
        else:
            phrase = clean_obj.capitalize()

        now = time.time()
        # Suppress identical announcement if within inspection cooldown
        if phrase == self.last_inspection_text and (now - self.last_inspection_time < self.inspection_cooldown):
            return False

        self.last_inspection_text = phrase
        self.last_inspection_time = now
        return self.speak(phrase, priority=PRIORITY_MODE)

    def _worker(self) -> None:
        """Background worker thread that processes speech requests via pyttsx3."""
        engine = None
        if self.mock_engine is not None:
            engine = self.mock_engine
        elif self.enabled:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                if self.rate:
                    engine.setProperty("rate", self.rate)
                if self.volume:
                    engine.setProperty("volume", self.volume)
            except Exception as e:
                print(f"[WARNING] Speech engine initialization failed: {e}")
                engine = None

        while self.is_running:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                # Poison pill received: shutdown thread
                break

            text = item.get("text", "")
            if not text or not self.enabled:
                self.queue.task_done()
                continue

            try:
                if engine is not None:
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                print(f"[WARNING] Speech playback error: {e}")
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        """Stops the speech manager and terminates the background worker thread."""
        self.is_running = False
        try:
            self.queue.put(None)
            if self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1.0)
        except Exception:
            pass
