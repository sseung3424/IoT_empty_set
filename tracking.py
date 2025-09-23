# follow_person_cpu.py
# -*- coding: utf-8 -*-
"""
Raspberry Pi 4B + Picamera2 + TFLite (CPU) + YB_Pcb_Car
Human-following with wheel drive + pan/tilt servos (NO Coral).

- Camera: Picamera2 (640x480, RGB888)
- Detector: MobileNet SSD v2 COCO (quant, postprocess) - CPU only
- Drive: YB_Pcb_Car (I2C motor driver)
- Servos: Ctrl_Servo(0, pan), Ctrl_Servo(1, tilt)

Keys:
  ESC : quit
"""

import os
import time
import numpy as np
import cv2

from picamera2 import Picamera2
from YB_Pcb_Car import YB_Pcb_Car   # your provided class

# ---------------------- Config ----------------------
MODEL_DIR   = "./models"
MODEL_CPU   = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
LABELS_TXT  = "coco_labels.txt"

CONF_THRESH = 0.30
TARGET_LABEL = "person"

# Wheel drive tuning (Yahboom typical 0~255)
BASE_SPEED = 60
MAX_SPEED  = 100
MIN_SPEED  = 40

# Turning sensitivity: differential = int(Kx * x_dev)
Kx = 280.0
CENTER_DEADZONE = 0.06   # no turning if |x_dev| < deadzone
STOP_NEAR_Y     = 0.88   # stop if bbox bottom close to frame bottom

# Servo tuning
PAN_ID = 0
TILT_ID = 1
PAN_CENTER  = 90    # 0~180
TILT_CENTER = 90
PAN_GAIN  = 95.0    # degrees per 1.0 x_dev
TILT_GAIN = 95.0    # degrees per 1.0 y_dev
PAN_SMOOTH  = 0.35  # EMA smoothing (0=no smooth, 0.3~0.5 good)
TILT_SMOOTH = 0.35
PAN_MIN, PAN_MAX   = 0, 180
TILT_MIN, TILT_MAX = 45, 90
# ----------------------------------------------------


# ---------------------- TFLite helpers (CPU) ----------------------
def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pair = line.strip().split(maxsplit=1)
            if len(pair) == 2:
                idx, name = pair
                labels[int(idx)] = name
    return labels


def make_interpreter_cpu(model_path):
    # Requires: pip install tflite-runtime  (on Raspberry Pi)
    from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path=model_path)


def set_input_tensor(interpreter, image_rgb):
    """Resize to model input size and copy to input tensor (quantized ok)."""
    input_details = interpreter.get_input_details()[0]
    h, w = input_details["shape"][1], input_details["shape"][2]
    resized = cv2.resize(image_rgb, (w, h))
    if input_details["dtype"] == np.float32:
        resized = resized.astype(np.float32) / 255.0
    else:
        resized = resized.astype(np.uint8)
    interpreter.set_tensor(input_details["index"], np.expand_dims(resized, axis=0))


def get_output(interpreter, conf_thresh=0.3):
    """
    Parse SSD postprocess outputs (ymin,xmin,ymax,xmax) normalized.
    Returns: list of dict {bbox:(xmin,ymin,xmax,ymax), score, class_id}
    """
    output_details = interpreter.get_output_details()
    boxes   = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0].astype(np.int32)
    scores  = interpreter.get_tensor(output_details[2]["index"])[0]
    count   = int(interpreter.get_tensor(output_details[3]["index"])[0])

    results = []
    for i in range(count):
        if scores[i] < conf_thresh:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        results.append({
            "bbox": (float(xmin), float(ymin), float(xmax), float(ymax)),
            "score": float(scores[i]),
            "class_id": int(classes[i]),
        })
    return results
# ------------------------------------------------------------------


# ---------------------- Drive/Servo control ----------------------
class Follower:
    def __init__(self, car: YB_Pcb_Car):
        self.car = car
        self.pan = PAN_CENTER
        self.tilt = TILT_CENTER
        # Initialize servos to center (ignore errors if not connected)
        try:
            self.car.Ctrl_Servo(PAN_ID, self.pan)
            self.car.Ctrl_Servo(TILT_ID, self.tilt)
        except Exception as e:
            print("Servo init failed:", e)

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def stop(self):
        self.car.Car_Stop()

    def drive_wheels(self, x_dev, y_max):
        """
        x_dev: (0.5 - x_center), + => target is left
        y_max: bbox bottom normalized
        Policy:
          - If near (y_max>STOP_NEAR_Y): stop
          - Else forward with differential based on x_dev (P control)
        """
        if y_max > STOP_NEAR_Y:
            self.stop()
            return "Stop"

        if abs(x_dev) < CENTER_DEADZONE:
            delta = 0
        else:
            delta = -int(Kx * x_dev)

        l = self._clamp(BASE_SPEED + delta, MIN_SPEED, MAX_SPEED)
        r = self._clamp(BASE_SPEED - delta, MIN_SPEED, MAX_SPEED)
        self.car.Car_Run(l, r)

        if delta > 10:
            return f"Forward-L({l})"
        elif delta < -10:
            return f"Forward-R({r})"
        else:
            return "Forward"

    def aim_servos(self, x_center, y_center):
        """
        Map bbox center to servo angles with smoothing.
        x_center,y_center in [0,1]
        pan  : 90 + PAN_GAIN*(0.5 - x_center)
        tilt : 90 - TILT_GAIN*(y_center - 0.5)
        """
        pan_target  = PAN_CENTER + (0.5 - x_center) * PAN_GAIN
        tilt_target = TILT_CENTER - (y_center - 0.5) * TILT_GAIN

        # EMA smoothing
        self.pan  = (1 - PAN_SMOOTH)  * pan_target  + PAN_SMOOTH  * self.pan
        self.tilt = (1 - TILT_SMOOTH) * tilt_target + TILT_SMOOTH * self.tilt

        pan_cmd  = int(self._clamp(round(self.pan),  PAN_MIN,  PAN_MAX))
        tilt_cmd = int(self._clamp(round(self.tilt), TILT_MIN, TILT_MAX))

        try:
            self.car.Ctrl_Servo(PAN_ID, pan_cmd)
            self.car.Ctrl_Servo(TILT_ID, tilt_cmd)
        except Exception as e:
            print("Servo write failed:", e)

        return pan_cmd, tilt_cmd
# ----------------------------------------------------------------


def main():
    # Load labels & model (CPU)
    labels = load_labels(os.path.join(MODEL_DIR, LABELS_TXT))
    model_path = os.path.join(MODEL_DIR, MODEL_CPU)
    interpreter = make_interpreter_cpu(model_path)
    interpreter.allocate_tensors()

    # Init car + controller
    car = YB_Pcb_Car()
    follow = Follower(car)

    # Init camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)

    fps_t0 = time.time()
    fps_cnt = 0
    WINDOW = "follow"

    try:
        while True:
            # Capture RGB frame (np array, RGB888)
            rgb = picam2.capture_array()
            h, w, _ = rgb.shape

            # Inference
            set_input_tensor(interpreter, rgb)
            interpreter.invoke()
            detections = get_output(interpreter, CONF_THRESH)

            # Pick best-scoring 'person'
            best = None
            best_score = 0.0
            for det in detections:
                name = labels.get(det["class_id"], str(det["class_id"]))
                if name != TARGET_LABEL:
                    continue
                if det["score"] > best_score:
                    best = det
                    best_score = det["score"]

            status = "Idle"
            x_dev = 0.0
            y_max_draw = 0.0
            pan_cmd = follow.pan
            tilt_cmd = follow.tilt

            # Draw all detections (grey), highlight tracked (green)
            for det in detections:
                (xmin, ymin, xmax, ymax) = det["bbox"]
                x0, y0 = int(xmin * w), int(ymin * h)
                x1, y1 = int(xmax * w), int(ymax * h)
                name = labels.get(det["class_id"], str(det["class_id"]))
                score = det["score"]
                color = (0, 255, 0) if det is best else (120, 120, 120)
                cv2.rectangle(rgb, (x0, y0), (x1, y1), color, 2)
                cv2.putText(rgb, f"{name} {int(score*100)}%", (x0, max(14, y0-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if best is not None:
                xmin, ymin, xmax, ymax = best["bbox"]
                x_center = (xmin + xmax) * 0.5
                y_center = (ymin + ymax) * 0.5

                # Drive wheels
                x_dev = 0.5 - x_center
                y_max_draw = ymax
                status = follow.drive_wheels(x_dev, y_max=ymax)

                # Aim servos toward target
                pan_cmd, tilt_cmd = follow.aim_servos(x_center, y_center)

                # Visual cues
                cv2.circle(rgb, (int(x_center * w), int(y_center * h)), 6, (0, 0, 255), -1)
                # Center guide
                cv2.line(rgb, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
                # Deadzone box
                dx = int(CENTER_DEADZONE * w)
                cv2.rectangle(rgb, (w // 2 - dx, 0), (w // 2 + dx, h), (0, 200, 0), 1)
            else:
                follow.stop()
                status = "Idle"

            # FPS
            fps_cnt += 1
            if time.time() - fps_t0 >= 1.0:
                fps = fps_cnt / (time.time() - fps_t0)
                fps_t0 = time.time()
                fps_cnt = 0
            else:
                fps = 0.0

            # HUD
            cv2.rectangle(rgb, (0, 0), (w, 26), (0, 0, 0), -1)
            cv2.putText(rgb, f"STATUS: {status}", (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(rgb, f"x_dev={x_dev:+.3f}  y_max={y_max_draw:.3f}", (240, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 2)
            if fps > 0:
                cv2.putText(rgb, f"FPS={fps:.1f}", (w - 110, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 2)
            cv2.putText(rgb, f"PAN={int(pan_cmd)} TILT={int(tilt_cmd)}", (w - 240, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Show
            cv2.imshow(WINDOW, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            follow.stop()
        except:
            pass
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    for t in range(3, 0, -1):
        print(f"Starting in {t}...")
        time.sleep(1)
    main()