"""
Computer vision processing pipeline for VisionFlow.
Coordinates frame capture, object detection (YOLOv8n), obstacle analysis,
periodic text recognition (EasyOCR), MediaPipe hand tracking, virtual HUD interaction,
and asynchronous offline audio feedback (pyttsx3).
"""

import cv2
from visionflow.config import (
    FONT_FACE,
    FONT_SCALE,
    FONT_THICKNESS,
    TEXT_COLOR,
    STATUS_BG_COLOR,
    STATUS_PADDING,
    BBOX_COLOR,
    BBOX_THICKNESS,
    LABEL_FONT_SCALE,
    LABEL_FONT_THICKNESS,
    LABEL_BG_COLOR,
    LABEL_TEXT_COLOR,
    COLOR_SAFE,
    COLOR_WARNING,
    COLOR_OBSTACLE_TAG,
    PROXIMITY_NEAR,
    WARNING_PATH_CLEAR,
    OCR_INTERVAL_FRAMES,
    OCR_BBOX_COLOR,
    OCR_BBOX_THICKNESS,
    OCR_LABEL_FONT_SCALE,
    OCR_LABEL_FONT_THICKNESS,
    OCR_LABEL_BG_COLOR,
    OCR_LABEL_TEXT_COLOR,
    OCR_SUMMARY_BG_COLOR,
    OCR_SUMMARY_TEXT_COLOR,
    GESTURE_PINCH,
    GESTURE_POINT,
    GESTURE_OPEN_PALM,
    GESTURE_NONE,
    HUD_BUTTONS,
    HUD_BTN_BG_COLOR,
    HUD_BTN_BORDER_COLOR,
    HUD_BTN_HOVER_COLOR,
    HUD_BTN_ACTIVE_COLOR,
    HUD_BTN_TEXT_COLOR,
    HUD_STATUS_BG_COLOR,
    HUD_STATUS_TEXT_COLOR,
    CURSOR_COLOR,
    CURSOR_RADIUS,
    TARGET_HIGHLIGHT_COLOR,
)
from visionflow.core.state import AppState
from visionflow.modules.object_detector import ObjectDetector
from visionflow.modules.obstacle_detector import ObstacleDetector
from visionflow.modules.ocr_reader import OCRReader
from visionflow.modules.hand_tracker import HandTracker
from visionflow.modules.gesture_detector import GestureDetector
from visionflow.modules.segmentation import SegmentationProcessor
from visionflow.modules.inspection import InspectionProcessor
from visionflow.audio.speech import SpeechManager


class Pipeline:
    """
    Main frame processing pipeline.
    Milestone 1: Status overlay.
    Milestone 2: YOLOv8n object detection and bounding boxes.
    Milestone 3: Spatial obstacle zone detection, proximity estimation, and navigation warnings.
    Milestone 4: Periodic OCR text recognition and on-screen display.
    Milestone 5: Hand gesture tracking (MediaPipe) and Virtual HUD fingertip interaction.
    Milestone 6: Asynchronous offline speech feedback for navigation, OCR, and mode events.
    Milestone 7: Person segmentation and Spotlight visual effect mode.
    Milestone 8: Fingertip-based object inspection and color identification.
    """

    def __init__(
        self,
        state: AppState,
        detector: ObjectDetector = None,
        obstacle_detector: ObstacleDetector = None,
        ocr_reader: OCRReader = None,
        ocr_interval: int = OCR_INTERVAL_FRAMES,
        hand_tracker: HandTracker = None,
        gesture_detector: GestureDetector = None,
        speech_manager: SpeechManager = None,
        segmenter: SegmentationProcessor = None,
        inspection_processor: InspectionProcessor = None,
    ):
        self.state = state

        # Inject or instantiate ObjectDetector
        if detector is not None:
            self.detector = detector
        else:
            self.detector = ObjectDetector()

        # Inject or instantiate ObstacleDetector
        if obstacle_detector is not None:
            self.obstacle_detector = obstacle_detector
        else:
            self.obstacle_detector = ObstacleDetector()

        # Inject or instantiate OCRReader
        if ocr_reader is not None:
            self.ocr_reader = ocr_reader
        else:
            self.ocr_reader = OCRReader()

        # Inject or instantiate HandTracker
        if hand_tracker is not None:
            self.hand_tracker = hand_tracker
        else:
            self.hand_tracker = HandTracker()

        # Inject or instantiate GestureDetector
        if gesture_detector is not None:
            self.gesture_detector = gesture_detector
        else:
            self.gesture_detector = GestureDetector()

        # Inject or instantiate SpeechManager (Milestone 6)
        if speech_manager is not None:
            self.speech_manager = speech_manager
        else:
            self.speech_manager = SpeechManager()

        # Inject or instantiate SegmentationProcessor (Milestone 7)
        if segmenter is not None:
            self.segmenter = segmenter
        else:
            self.segmenter = SegmentationProcessor()

        # Inject or instantiate InspectionProcessor (Milestone 8)
        if inspection_processor is not None:
            self.inspection_processor = inspection_processor
        else:
            self.inspection_processor = InspectionProcessor()

        # OCR periodic processing settings
        self.ocr_interval = ocr_interval
        self.frame_counter = 0
        self.cached_ocr_results = []

        # Hand & HUD interaction state
        self.last_gesture = GESTURE_NONE
        self.hovered_button = None

    def get_hud_buttons(self, frame_width: int):
        """Returns button definitions adapted to current frame width."""
        buttons = []
        for btn in HUD_BUTTONS:
            bx, by, bw, bh = btn["rect"]
            effective_x = frame_width - bw - 20 if frame_width != 640 else bx
            buttons.append({
                "name": btn["name"],
                "mode": btn["mode"],
                "rect": (effective_x, by, bw, bh),
            })
        return buttons

    def process_frame(self, frame):
        """
        Processes an input frame:
        1. Runs YOLOv8n object detection.
        2. Evaluates detected objects for obstacle zone (LEFT/CENTER/RIGHT) and proximity (FAR/MEDIUM/NEAR).
        3. Delivers spoken audio warning for near obstacles (with debouncing).
        4. Runs OCR periodically (once every N frames) and speaks recognized text.
        5. Detects hand landmarks and classifies hand gestures (MediaPipe).
        6. Performs virtual HUD interaction (index fingertip cursor hover and pinch-click mode select with audio).
        7. Applies Spotlight visual effect if SPOTLIGHT mode is active.
        8. Draws YOLO bounding boxes and obstacle tags.
        9. Draws OCR bounding boxes and text labels.
        10. Draws Hand skeleton, landmarks, and cursor.
        11. Renders Virtual HUD interactive buttons.
        12. Renders Top Status Banner, Navigation Warning Banner, OCR Summary, and Gesture HUD Banner.

        Args:
            frame (numpy.ndarray): Raw input frame from camera.

        Returns:
            numpy.ndarray: Processed frame with visual overlays.
        """
        if frame is None:
            return None

        # Work on a copy of the frame to keep original capture intact
        processed = frame.copy()
        frame_height, frame_width = processed.shape[:2]

        # Milestone 7: Apply Spotlight Effect when SPOTLIGHT mode is active
        if self.state.current_mode.lower() == "spotlight" and self.segmenter is not None:
            processed = self.segmenter.apply_spotlight(processed)

        # Step 1: Run object detection
        detections = []
        if self.detector is not None:
            detections = self.detector.detect(processed)

        # Step 2: Run obstacle and proximity analysis
        obstacle_info = {"obstacles": [], "warning": WARNING_PATH_CLEAR, "is_warning": False}
        if self.obstacle_detector is not None:
            obstacle_info = self.obstacle_detector.process_detections(
                detections, frame_width, frame_height
            )

        obstacles = obstacle_info["obstacles"]
        warning_text = obstacle_info["warning"]
        is_hazard = obstacle_info["is_warning"]

        # Step 2b: Audio Feedback for Navigation / Obstacles (Milestone 6)
        if self.speech_manager is not None:
            self.speech_manager.speak_obstacle(warning_text)

        # Step 3: Run OCR periodically (every N frames) to avoid dropping camera FPS
        self.frame_counter += 1
        if self.ocr_reader is not None:
            if self.frame_counter == 1 or (self.frame_counter % self.ocr_interval == 0):
                self.cached_ocr_results = self.ocr_reader.read_text(processed)
                # Audio Feedback for OCR Text Recognition (Milestone 6)
                if self.speech_manager is not None:
                    self.speech_manager.speak_ocr(self.cached_ocr_results)

        # Step 4: Run Hand Tracking & Gesture Recognition (Milestone 5)
        hands = []
        if self.hand_tracker is not None:
            hands = self.hand_tracker.detect_hands(processed)

        current_gesture = GESTURE_NONE
        index_tip = None
        if hands and self.gesture_detector is not None:
            current_gesture = self.gesture_detector.classify_gesture(hands[0])
            index_tip = hands[0].get("index_tip")
        elif hands:
            index_tip = hands[0].get("index_tip")

        self.last_gesture = current_gesture

        # Step 5: Virtual HUD Interaction (Hover & Pinch Selection)
        hud_buttons = self.get_hud_buttons(frame_width)
        hovered_btn = None

        if index_tip is not None:
            ix, iy = index_tip
            for btn in hud_buttons:
                bx, by, bw, bh = btn["rect"]
                if bx <= ix <= bx + bw and by <= iy <= by + bh:
                    hovered_btn = btn
                    # Click trigger: Pinching while hovering button
                    if current_gesture == GESTURE_PINCH:
                        if self.state.current_mode.lower() != btn["mode"].lower():
                            self.state.set_mode(btn["mode"])
                            # Audio Feedback: Mode Selection confirmation (Milestone 6)
                            if self.speech_manager is not None:
                                self.speech_manager.speak_mode(btn["mode"])
                        else:
                            self.state.set_mode(btn["mode"])
                    break

        self.hovered_button = hovered_btn

        # Step 5b: Inspection Targeting & Color Extraction (Milestone 8)
        inspection_result = {"target_found": False, "object_name": None, "color": "UNKNOWN", "bbox": None}
        if self.inspection_processor is not None and index_tip is not None and current_gesture == GESTURE_POINT:
            inspection_result = self.inspection_processor.inspect(
                frame=processed,
                detections=detections,
                fingertip=index_tip,
                gesture=current_gesture,
            )
            if inspection_result["target_found"]:
                obj_name = inspection_result["object_name"]
                color_name = inspection_result["color"]
                if self.speech_manager is not None:
                    self.speech_manager.speak_inspection(obj_name, color_name)

        # Build a mapping from bbox coordinates to obstacle data for fast lookup
        obstacle_map = {obs["bbox"]: obs for obs in obstacles}

        # Step 6: Draw YOLO bounding boxes and obstacle tags
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            class_name = det["class_name"]
            confidence = det["confidence"]
            yolo_label = f"{class_name} {confidence:.2f}"

            # Check if this object is a near obstacle
            obs_data = obstacle_map.get(bbox)
            is_near_obstacle = obs_data is not None and obs_data["proximity"] == PROXIMITY_NEAR

            # Highlight NEAR obstacles with red border, others with green
            box_color = COLOR_WARNING if is_near_obstacle else BBOX_COLOR
            cv2.rectangle(processed, (x1, y1), (x2, y2), box_color, BBOX_THICKNESS)

            # 6a. Draw YOLO class and confidence label above the bounding box
            (yolo_w, yolo_h), _ = cv2.getTextSize(
                yolo_label, FONT_FACE, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS
            )
            yolo_y1 = max(y1 - yolo_h - 6, 0)
            yolo_y2 = y1
            yolo_x2 = min(x1 + yolo_w + 6, frame_width)

            cv2.rectangle(
                processed,
                (x1, yolo_y1),
                (yolo_x2, yolo_y2),
                LABEL_BG_COLOR,
                -1,
            )
            cv2.putText(
                processed,
                yolo_label,
                (x1 + 3, yolo_y2 - 4),
                FONT_FACE,
                LABEL_FONT_SCALE,
                LABEL_TEXT_COLOR,
                LABEL_FONT_THICKNESS,
                cv2.LINE_AA,
            )

            # 6b. If an obstacle, also draw obstacle tag (name | zone | proximity) below bbox
            if obs_data is not None:
                obs_label = obs_data["label"]
                (obs_w, obs_h), _ = cv2.getTextSize(
                    obs_label, FONT_FACE, 0.45, 1
                )

                if y2 + obs_h + 8 < frame_height:
                    obs_box_y1 = y2
                    obs_box_y2 = y2 + obs_h + 8
                    text_y = y2 + obs_h + 3
                else:
                    obs_box_y1 = max(0, y2 - obs_h - 8)
                    obs_box_y2 = y2
                    text_y = y2 - 4

                obs_box_x2 = min(x1 + obs_w + 8, frame_width)
                tag_bg = COLOR_WARNING if is_near_obstacle else COLOR_OBSTACLE_TAG
                cv2.rectangle(
                    processed,
                    (x1, obs_box_y1),
                    (obs_box_x2, obs_box_y2),
                    tag_bg,
                    -1,
                )
                cv2.putText(
                    processed,
                    obs_label,
                    (x1 + 4, text_y),
                    FONT_FACE,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Step 6c: Highlight targeted object if inspection target found (Milestone 8)
        if inspection_result["target_found"] and inspection_result["bbox"] is not None:
            tx1, ty1, tx2, ty2 = inspection_result["bbox"]
            cv2.rectangle(processed, (tx1, ty1), (tx2, ty2), TARGET_HIGHLIGHT_COLOR, 3)

            target_tag = f"TARGET: {inspection_result['object_name'].upper()} | {inspection_result['color'].upper()}"
            (tt_w, tt_h), _ = cv2.getTextSize(target_tag, FONT_FACE, 0.5, 1)
            tt_y1 = max(0, ty1 - tt_h - 10)
            tt_y2 = ty1
            cv2.rectangle(processed, (tx1, tt_y1), (tx1 + tt_w + 8, tt_y2), TARGET_HIGHLIGHT_COLOR, -1)
            cv2.putText(
                processed,
                target_tag,
                (tx1 + 4, tt_y2 - 3),
                FONT_FACE,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Step 7: Draw OCR Bounding Boxes and Labels on Detected Text Regions
        for item in self.cached_ocr_results:
            tx1, ty1, tx2, ty2 = item["bbox"]
            text_str = item["text"]
            t_conf = item["confidence"]
            text_label = f"{text_str} ({t_conf:.2f})"

            # Draw text bounding box
            cv2.rectangle(processed, (tx1, ty1), (tx2, ty2), OCR_BBOX_COLOR, OCR_BBOX_THICKNESS)

            # Draw small text tag above or inside text box
            (tw, th), _ = cv2.getTextSize(
                text_label, FONT_FACE, OCR_LABEL_FONT_SCALE, OCR_LABEL_FONT_THICKNESS
            )
            tag_y1 = max(0, ty1 - th - 6)
            tag_y2 = ty1 if ty1 - th - 6 >= 0 else ty1 + th + 6
            tag_x2 = min(tx1 + tw + 6, frame_width)

            cv2.rectangle(processed, (tx1, tag_y1), (tag_x2, tag_y2), OCR_LABEL_BG_COLOR, -1)
            cv2.putText(
                processed,
                text_label,
                (tx1 + 3, tag_y2 - 3),
                FONT_FACE,
                OCR_LABEL_FONT_SCALE,
                OCR_LABEL_TEXT_COLOR,
                OCR_LABEL_FONT_THICKNESS,
                cv2.LINE_AA,
            )

        # Step 8: Draw Hand Skeleton, Landmarks & Virtual Cursor (Milestone 5)
        if self.hand_tracker is not None and hands:
            self.hand_tracker.draw_landmarks(processed, hands)

        # Draw interactive cursor enhancement on index fingertip
        if index_tip is not None:
            if current_gesture == GESTURE_PINCH:
                # Active pinch click feedback (pulsing green indicator)
                cv2.circle(processed, index_tip, CURSOR_RADIUS + 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(processed, index_tip, CURSOR_RADIUS + 7, (255, 255, 255), 2, cv2.LINE_AA)
            elif hovered_btn is not None:
                # Hover feedback (bright gold indicator)
                cv2.circle(processed, index_tip, CURSOR_RADIUS + 3, HUD_BTN_HOVER_COLOR, -1, cv2.LINE_AA)
                cv2.circle(processed, index_tip, CURSOR_RADIUS + 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Step 9: Draw Virtual HUD Buttons (Milestone 5)
        for btn in hud_buttons:
            bx, by, bw, bh = btn["rect"]
            is_active = (self.state.current_mode.lower() == btn["mode"].lower())
            is_hovered = (hovered_btn is not None and hovered_btn["mode"] == btn["mode"])

            if is_active:
                btn_border = HUD_BTN_ACTIVE_COLOR
                btn_border_thick = 3
                btn_bg = (20, 60, 20)      # Subtle green background for selected mode
                label_prefix = "> "
                label_suffix = " <"
            elif is_hovered:
                btn_border = HUD_BTN_HOVER_COLOR
                btn_border_thick = 2
                btn_bg = (60, 60, 20)      # Subtle gold highlight on hover
                label_prefix = ""
                label_suffix = ""
            else:
                btn_border = HUD_BTN_BORDER_COLOR
                btn_border_thick = 1
                btn_bg = HUD_BTN_BG_COLOR
                label_prefix = ""
                label_suffix = ""

            # Draw button background and border
            cv2.rectangle(processed, (bx, by), (bx + bw, by + bh), btn_bg, -1)
            cv2.rectangle(processed, (bx, by), (bx + bw, by + bh), btn_border, btn_border_thick)

            # Draw centered button label
            btn_label = f"{label_prefix}{btn['name']}{label_suffix}"
            (tw, th), _ = cv2.getTextSize(btn_label, FONT_FACE, 0.45, 1)
            tx = bx + (bw - tw) // 2
            ty = by + (bh + th) // 2
            cv2.putText(
                processed,
                btn_label,
                (tx, ty),
                FONT_FACE,
                0.45,
                HUD_BTN_TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        # Step 10: Render Top Status Banner
        num_objects = len(detections)
        status_text = f"VisionFlow | Objects: {num_objects} | Mode: {self.state.current_mode.upper()} | Q to Exit"
        (text_width, text_height), baseline = cv2.getTextSize(
            status_text, FONT_FACE, FONT_SCALE, FONT_THICKNESS
        )
        x, y = 20, 35
        cv2.rectangle(
            processed,
            (x - STATUS_PADDING, y - text_height - STATUS_PADDING),
            (x + text_width + STATUS_PADDING, y + baseline + STATUS_PADDING),
            STATUS_BG_COLOR,
            -1,
        )
        cv2.putText(
            processed,
            status_text,
            (x, y),
            FONT_FACE,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

        # Step 11: Render Navigation & Obstacle Warning Banner
        warn_scale = 0.65
        warn_thick = 2
        (warn_w, warn_h), warn_base = cv2.getTextSize(
            warning_text, FONT_FACE, warn_scale, warn_thick
        )
        wx, wy = 20, 75
        warn_bg = COLOR_WARNING if is_hazard else COLOR_SAFE
        cv2.rectangle(
            processed,
            (wx - STATUS_PADDING, wy - warn_h - STATUS_PADDING),
            (wx + warn_w + STATUS_PADDING, wy + warn_base + STATUS_PADDING),
            warn_bg,
            -1,
        )
        cv2.putText(
            processed,
            warning_text,
            (wx, wy),
            FONT_FACE,
            warn_scale,
            (255, 255, 255),
            warn_thick,
            cv2.LINE_AA,
        )

        # Step 12: Render Compact OCR Summary Banner
        if self.cached_ocr_results:
            combined_text = " ".join(item["text"] for item in self.cached_ocr_results)
            if len(combined_text) > 42:
                combined_text = combined_text[:39] + "..."
            ocr_summary_text = f"OCR: {combined_text}"
        else:
            ocr_summary_text = "OCR: [No text detected]"

        ocr_scale = 0.55
        ocr_thick = 1
        (ow, oh), obase = cv2.getTextSize(
            ocr_summary_text, FONT_FACE, ocr_scale, ocr_thick
        )
        ox, oy = 20, 115
        cv2.rectangle(
            processed,
            (ox - STATUS_PADDING, oy - oh - STATUS_PADDING),
            (ox + ow + STATUS_PADDING, oy + obase + STATUS_PADDING),
            OCR_SUMMARY_BG_COLOR,
            -1,
        )
        cv2.putText(
            processed,
            ocr_summary_text,
            (ox, oy),
            FONT_FACE,
            ocr_scale,
            OCR_SUMMARY_TEXT_COLOR,
            ocr_thick,
            cv2.LINE_AA,
        )

        # Step 13: Render Gesture & Virtual HUD Status Banner (Milestone 5)
        if hovered_btn is not None and current_gesture == GESTURE_PINCH:
            gesture_text = f"Gesture: PINCH -> Selected [{hovered_btn['name']}]"
            gesture_bg = (20, 80, 20)
            gesture_color = (0, 255, 0)
        elif hovered_btn is not None:
            gesture_text = f"Gesture: {current_gesture} | Hover: [{hovered_btn['name']}]"
            gesture_bg = (50, 40, 20)
            gesture_color = (0, 220, 255)
        elif current_gesture != GESTURE_NONE:
            gesture_text = f"Gesture: {current_gesture} | Mode: {self.state.current_mode.upper()}"
            gesture_bg = HUD_STATUS_BG_COLOR
            gesture_color = HUD_STATUS_TEXT_COLOR
        else:
            gesture_text = f"Gesture: NONE | Mode: {self.state.current_mode.upper()}"
            gesture_bg = HUD_STATUS_BG_COLOR
            gesture_color = (180, 180, 180)

        gx, gy = 20, 155
        g_scale = 0.55
        g_thick = 1
        (gw, gh), gbase = cv2.getTextSize(gesture_text, FONT_FACE, g_scale, g_thick)
        cv2.rectangle(
            processed,
            (gx - STATUS_PADDING, gy - gh - STATUS_PADDING),
            (gx + gw + STATUS_PADDING, gy + gbase + STATUS_PADDING),
            gesture_bg,
            -1,
        )
        cv2.putText(
            processed,
            gesture_text,
            (gx, gy),
            FONT_FACE,
            g_scale,
            gesture_color,
            g_thick,
            cv2.LINE_AA,
        )

        # Step 14: Render Inspection Status Banner (Milestone 8)
        if self.state.current_mode.lower() == "inspect" or inspection_result["target_found"]:
            if inspection_result["target_found"]:
                insp_text = f"TARGET: {inspection_result['object_name'].upper()} | COLOR: {inspection_result['color'].upper()}"
                insp_bg = (50, 20, 50)
                insp_color = (0, 255, 255)
            elif current_gesture == GESTURE_POINT:
                insp_text = "INSPECT: Point at an object"
                insp_bg = (40, 20, 40)
                insp_color = (200, 200, 200)
            else:
                insp_text = "INSPECT: Use POINT gesture to target"
                insp_bg = (40, 20, 40)
                insp_color = (180, 180, 180)

            ix_pos, iy_pos = 20, 195
            i_scale = 0.55
            i_thick = 1
            (iw, ih), ibase = cv2.getTextSize(insp_text, FONT_FACE, i_scale, i_thick)
            cv2.rectangle(
                processed,
                (ix_pos - STATUS_PADDING, iy_pos - ih - STATUS_PADDING),
                (ix_pos + iw + STATUS_PADDING, iy_pos + ibase + STATUS_PADDING),
                insp_bg,
                -1,
            )
            cv2.putText(
                processed,
                insp_text,
                (ix_pos, iy_pos),
                FONT_FACE,
                i_scale,
                insp_color,
                i_thick,
                cv2.LINE_AA,
            )

        return processed
