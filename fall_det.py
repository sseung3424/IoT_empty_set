# fall_det.py
import RPi.GPIO as GPIO
from ultralytics import YOLO
import time
import numpy as np

# ------------------------------
# Setup buzzer
# ------------------------------
BUZZER_PIN = 32
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
buzzer = GPIO.PWM(BUZZER_PIN, 440)
buzzer.start(0)  # start with 0 duty cycle (silent)

# ------------------------------
# Load YOLO pose model
# ------------------------------
model = YOLO("yolov8n-pose.pt")

# ------------------------------
# FallDetector class
# ------------------------------
class FallDetector:
    def __init__(self, lie_frame_threshold=2):
        self.prev_state = "standing"
        self.fall_detected = False
        self.lie_counter = 0
        self.lie_frame_threshold = lie_frame_threshold  # # of consecutive frames to confirm fall

    def alert_buzzer(self, duration=0.5):
        """Trigger buzzer for alert"""
        buzzer.ChangeDutyCycle(50)  # turn on
        time.sleep(duration)
        buzzer.ChangeDutyCycle(0)   # turn off

    def process_frame(self, frame):
        """
        Process a single frame and detect fall
        Returns:
            - label: "standing" or "lying"
            - fall_detected: True if a fall is detected in this frame
        """
        results = model(frame)
        labels = []

        # Process each detected person
        for kpt in results[0].keypoints.xy:
            # Ensure keypoints are sufficient
            if len(kpt) < 13:
                continue

            try:
                left_shoulder = kpt[5]
                right_shoulder = kpt[6]
                left_hip = kpt[11]
                right_hip = kpt[12]

                center_top = (np.array(left_shoulder) + np.array(right_shoulder)) / 2
                center_bottom = (np.array(left_hip) + np.array(right_hip)) / 2

                width = abs(left_shoulder[0] - right_shoulder[0])
                height = abs(center_top[1] - center_bottom[1])

                # Determine posture
                if width > 0 and (height / width) < 0.5:
                    current_label = "lying"
                else:
                    current_label = "standing"

                labels.append(current_label)
            except Exception:
                continue  # skip if any keypoint is missing

        # Decide overall label for the frame
        label = labels[0] if labels else self.prev_state  # fallback to previous state

        # --------------------------
        # Fall detection with debounce
        # --------------------------
        if label == "lying":
            self.lie_counter += 1
        else:
            self.lie_counter = 0

        self.fall_detected = False
        if self.prev_state == "standing" and self.lie_counter >= self.lie_frame_threshold:
            self.fall_detected = True
            self.alert_buzzer()  # trigger buzzer

        self.prev_state = label
        return label, self.fall_detected

# ------------------------------
# Cleanup GPIO when module is unloaded
# ------------------------------
def cleanup():
    buzzer.stop()
    GPIO.cleanup()
