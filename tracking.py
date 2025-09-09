# tracking.py
import os, time, numpy as np, cv2
from picamera2 import Picamera2
from YB_Pcb_Car import YB_Pcb_Car
from tflite_runtime.interpreter import Interpreter
from frame_bus import BUS  # <-- NEW

# (CPU 공존성 향상: 수치 라이브러리 스레드 1로)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try: cv2.setNumThreads(1)
except Exception: pass

# ---------------------- TFLite helpers (네 코드 그대로) ----------------------
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
    return Interpreter(model_path=model_path)

def set_input_tensor(interpreter, image_rgb):
    input_details = interpreter.get_input_details()[0]
    h, w = input_details["shape"][1], input_details["shape"][2]
    resized = cv2.resize(image_rgb, (w, h))
    if input_details["dtype"] == np.float32:
        resized = resized.astype(np.float32) / 255.0
    else:
        resized = resized.astype(np.uint8)
    interpreter.set_tensor(input_details["index"], np.expand_dims(resized, axis=0))

def get_output(interpreter, conf_thresh=0.3):
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
# ---------------------------------------------------------------------------

# ---------------------- Config (네 값 그대로) ----------------------
MODEL_DIR   = "./models"
MODEL_CPU   = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
LABELS_TXT  = "coco_labels.txt"
CONF_THRESH = 0.40
TARGET_LABEL = "person"

BASE_SPEED = 100
MAX_SPEED  = 120
MIN_SPEED  = 60
Kx = 200.0
CENTER_DEADZONE = 0.10
STOP_NEAR_Y     = 0.90

PAN_ID = 0
TILT_ID = 1
PAN_CENTER  = 90
TILT_CENTER = 90
PAN_GAIN    = 150.0
TILT_GAIN   = 120.0
PAN_SMOOTH  = 0.20
TILT_SMOOTH = 0.30
PAN_MIN, PAN_MAX   = 0, 180
TILT_MIN, TILT_MAX = 45, 90

LOST_TIMEOUT = 3.0
SEARCH_SPEED = 50
# ----------------------------------------------------

class Follower:
    def __init__(self, car: YB_Pcb_Car):
        self.car = car
        self.pan = PAN_CENTER
        self.tilt = TILT_CENTER
        self.state = "SEARCHING"
        self.last_seen_ts = 0.0
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
        if y_max > STOP_NEAR_Y:
            self.stop()
            return "Stop"
        if abs(x_dev) < CENTER_DEADZONE:
            delta = 0
        else:
            delta = int(Kx * x_dev)
        l = self._clamp(BASE_SPEED - delta, MIN_SPEED, MAX_SPEED)
        r = self._clamp(BASE_SPEED + delta, MIN_SPEED, MAX_SPEED)
        self.car.Car_Run(l, r)
        if delta > 10: return f"Forward-R({r})"
        elif delta < -10: return f"Forward-L({l})"
        else: return "Forward"

    def aim_servos(self, x_center, y_center):
        pan_target  = PAN_CENTER + (0.5 - x_center) * PAN_GAIN
        tilt_target = TILT_CENTER - (y_center - 0.5) * TILT_GAIN
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

def run_tracking(stop_event):
    """Run your tracking loop + publish frames to BUS; stop when stop_event is set."""
    labels = load_labels(os.path.join(MODEL_DIR, LABELS_TXT))
    interpreter = make_interpreter_cpu(os.path.join(MODEL_DIR, MODEL_CPU))
    interpreter.allocate_tensors()
    car = YB_Pcb_Car()
    follow = Follower(car)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)
    WINDOW = "Follower"

    def _waitkey_or_stop():
        if stop_event.is_set():
            return 27
        ret = cv2.waitKey(1) & 0xFF
        time.sleep(0.001)  # 작은 양보
        return ret

    try:
        while not stop_event.is_set():
            rgb = picam2.capture_array()
            h, w, _ = rgb.shape

            set_input_tensor(interpreter, rgb)
            interpreter.invoke()
            detections = get_output(interpreter, CONF_THRESH)
            best = max([d for d in detections if labels.get(d["class_id"]) == TARGET_LABEL],
                       key=lambda x: x["score"], default=None)

            status = "Idle"
            pan_cmd, tilt_cmd = follow.pan, follow.tilt

            if best is not None:
                follow.state = "TRACKING"
                follow.last_seen_ts = time.monotonic()
                xmin, ymin, xmax, ymax = best["bbox"]
                x_center = (xmin + xmax) * 0.5
                y_center = (ymin + ymax) * 0.5
                x_dev = 0.5 - x_center
                status = follow.drive_wheels(x_dev, y_max=ymax)
                pan_cmd, tilt_cmd = follow.aim_servos(x_center, y_center)
            else:
                if follow.state == "TRACKING" and time.monotonic() - follow.last_seen_ts > LOST_TIMEOUT:
                    print("Target lost for too long. Entering SEARCH mode.")
                    follow.state = "SEARCHING"
                    follow.stop()
                if follow.state == "SEARCHING":
                    status = "Searching..."
                    pan_cmd, tilt_cmd = follow.aim_servos(0.5, 0.5)
                    car.Car_Spin_Left(SEARCH_SPEED, SEARCH_SPEED)

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # >>> 최신 프레임 공유 (낙상 워커가 사용)
            BUS.publish(bgr)

            if best:
                (xmin, ymin, xmax, ymax) = best["bbox"]
                x0, y0, x1, y1 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(bgr, f"STATUS: {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(bgr, f"PAN={int(pan_cmd)} TILT={int(tilt_cmd)}", (w - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow(WINDOW, bgr)

            if _waitkey_or_stop() == 27:
                break

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        try: follow.stop()
        except Exception: pass
        try: picam2.stop()
        except Exception: pass
        cv2.destroyAllWindows()
