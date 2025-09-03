# tracking.py
# Human-following with YOLO person detection and Yahboom car control
# - English-only comments for Raspberry Pi locale compatibility

import time
import cv2
import numpy as np
from ultralytics import YOLO
import RPi.GPIO as GPIO
import yb_car  # Yahboom car library
from typing import Optional

class HumanFollower:
    """
    YOLO-based human follower for Yahboom robot car.
    - Detect 'person' bounding boxes
    - Prefer the largest box (closest person)
    - Use image center offset to steer (P-control)
    - Use bbox height to estimate distance and set speed
    - Limit detection ROI to lower half of the frame (legs-focused)
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",     # person class available in generic YOLO
        camera_index: int = 10,
        show_window: bool = False,          # set True for debug visualization
        roi_ratio: float = 0.5,             # use lower 50% as ROI
        lost_timeout_s: float = 1.0,        # stop if person lost for this duration
        base_speed: int = 65,               # nominal forward speed (0~100)
        min_speed: int = 45,                # lower bound speed
        turn_gain: float = 0.35,            # steering gain for center offset
        size_near: float = 0.55,            # bbox height ratio threshold considered 'too close'
        size_far: float = 0.25,             # bbox height ratio threshold considered 'too far'
        backoff_speed: int = 55,            # reverse or brake pulse speed
        backoff_time_s: float = 0.25,       # reverse/back-off pulse duration
        min_conf: float = 0.35,             # min confidence for person detection
        smooth_alpha: float = 0.25          # EMA smoothing for error
    ):
        # Video
        self.cap = cv2.VideoCapture(camera_index)
        # Model (person detection)
        self.model = YOLO(model_path)

        # Car
        self.car = yb_car.YB_Pcb_Car()

        # Params
        self.show_window = show_window
        self.roi_ratio = np.clip(roi_ratio, 0.1, 1.0)
        self.lost_timeout_s = lost_timeout_s
        self.base_speed = int(np.clip(base_speed, 0, 100))
        self.min_speed = int(np.clip(min_speed, 0, 100))
        self.turn_gain = turn_gain
        self.size_near = np.clip(size_near, 0.05, 0.95)
        self.size_far = np.clip(size_far, 0.02, 0.90)
        self.backoff_speed = int(np.clip(backoff_speed, 0, 100))
        self.backoff_time_s = backoff_time_s
        self.min_conf = np.clip(min_conf, 0.05, 0.95)
        self.smooth_alpha = np.clip(smooth_alpha, 0.0, 1.0)

        # State
        self.last_seen_ts = 0.0
        self.err_ema = 0.0  # smoothed center offset
        self.running = False

        # GPIO safety
        GPIO.setwarnings(False)

    def _select_person(self, results, H: int, W: int):
        """
        Select the largest-valid 'person' box inside ROI (lower H*roi_ratio..H).
        Returns (cx_norm, box_h_ratio, (x1,y1,x2,y2)) or None if not found.
        """
        best = None
        best_area = 0.0
        roi_y0 = int(H * (1.0 - self.roi_ratio))

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for b in r.boxes:
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                # 'person' class is 0 in COCO
                if cls_id != 0 or conf < self.min_conf:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                # Check intersection with lower ROI
                if y2 < roi_y0:
                    continue

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                area = w * h
                if area > best_area:
                    cx = (x1 + x2) / 2
                    cx_norm = (cx - W / 2) / (W / 2)  # [-1, 1]
                    box_h_ratio = h / H
                    best = (cx_norm, box_h_ratio, (x1, y1, x2, y2), conf)
                    best_area = area

        return best  # or None

    def _drive_none(self):
        """Stop the car if nothing detected and lost for timeout."""
        now = time.monotonic()
        if now - self.last_seen_ts > self.lost_timeout_s:
            self.car.Car_Stop()

    def _drive_follow(self, cx_norm: float, h_ratio: float):
        """
        Follow control using center offset and bbox height ratio.
        - cx_norm: horizontal offset [-1..1], left negative, right positive
        - h_ratio: bbox height / frame height, as distance proxy
        """
        # Smooth heading error (EMA)
        self.err_ema = (1.0 - self.smooth_alpha) * self.err_ema + self.smooth_alpha * cx_norm

        # Compute turn differential from error
        turn = self.err_ema * self.turn_gain  # proportional term
        turn = float(np.clip(turn, -1.0, 1.0))

        # Distance logic from box height
        #   if too far -> move forward
        #   if too close -> short back-off pulse (or stop)
        if h_ratio < self.size_far:
            # Far: go forward with base speed, add small steering differential
            left = self.base_speed * (1.0 - 0.8 * max(0.0, turn))
            right = self.base_speed * (1.0 + 0.8 * max(0.0, -turn))
            left = int(max(self.min_speed, np.clip(left, 0, 100)))
            right = int(max(self.min_speed, np.clip(right, 0, 100)))
            self.car.Car_Run(left, right)
        elif h_ratio > self.size_near:
            # Near: brief back-off or stop to keep safe gap
            self.car.Car_Back(self.backoff_speed, self.backoff_speed)
            time.sleep(self.backoff_time_s)
            self.car.Car_Stop()
        else:
            # In-range: mostly steering-in-place if needed
            if abs(turn) > 0.08:
                if turn > 0:
                    # Need to rotate right
                    self.car.Car_Spin_Right(55, 55)
                else:
                    self.car.Car_Spin_Left(55, 55)
            else:
                # Aligned: small forward nudge to keep lock
                self.car.Car_Run(self.min_speed, self.min_speed)

    def run(self, stop_event: Optional["threading.Event"] = None):
        """
        Main loop: read camera, detect person, and drive.
        Use an external threading.Event to stop gracefully.
        """
        self.running = True
        try:
            while self.running and (stop_event is None or not stop_event.is_set()):
                ok, frame = self.cap.read()
                if not ok:
                    self.car.Car_Stop()
                    time.sleep(0.02)
                    continue

                H, W = frame.shape[:2]
                # Optional: draw ROI line
                roi_y0 = int(H * (1.0 - self.roi_ratio))
                # YOLO inference (BGR frame is fine)
                results = self.model(frame, verbose=False)

                sel = self._select_person(results, H, W)

                if sel is None:
                    self._drive_none()
                    if self.show_window:
                        cv2.line(frame, (0, roi_y0), (W, roi_y0), (255, 255, 0), 2)
                        cv2.putText(frame, "Searching...", (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cx_norm, h_ratio, (x1, y1, x2, y2), conf = sel
                    self.last_seen_ts = time.monotonic()
                    # Drive toward the target
                    self._drive_follow(cx_norm, h_ratio)

                    if self.show_window:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        info = f"cx={cx_norm:+.2f} h={h_ratio:.2f} conf={conf:.2f}"
                        cv2.putText(frame, info, (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                        cv2.line(frame, (0, roi_y0), (W, roi_y0), (255, 255, 0), 2)
                        cv2.circle(frame, (W // 2, H // 2), 4, (0, 0, 255), -1)

                if self.show_window:
                    cv2.imshow("Human Follow", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Small sleep to ease CPU
                time.sleep(0.01)

        finally:
            self.car.Car_Stop()
            self.cap.release()
            if self.show_window:
                cv2.destroyAllWindows()

    def stop(self):
        """Request to stop the loop."""
        self.running = False


def run_tracking_thread(
    stop_event: Optional["threading.Event"] = None,
    camera_index: int = 0,
    show_window: bool = False
):
    """
    Convenience function to be used as a thread target in main.py
    """
    follower = HumanFollower(
        model_path="yolov8n-pose.pt",
        camera_index=camera_index,
        show_window=show_window
    )
    follower.run(stop_event=stop_event)
