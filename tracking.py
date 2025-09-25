# follow_person_final.py
import os
import time
import numpy as np
import cv2
from picamera2 import Picamera2
from YB_Pcb_Car import YB_Pcb_Car
from tflite_runtime.interpreter import Interpreter  # tflite import

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
        ymin, xmin, ymax, xmax = boxes[i]  # NOTE: 모델은 (ymin, xmin, ymax, xmax)
        results.append({
            "bbox": (float(xmin), float(ymin), float(xmax), float(ymax)),  # (xmin, ymin, xmax, ymax)
            "score": float(scores[i]),
            "class_id": int(classes[i]),
        })
    return results
# ------------------------------------------------------------------

# ---------------------- Config ----------------------
MODEL_DIR   = "./models"
MODEL_CPU   = "mobilenet_ssd_v2_coco_quant_postprocess.tflite"
LABELS_TXT  = "coco_labels.txt"
CONF_THRESH = 0.40
TARGET_LABEL = "person"

# 차동 구동
BASE_SPEED = 100
MAX_SPEED  = 120
MIN_SPEED  = 60
Kx = 200.0
CENTER_DEADZONE = 0.10       # x_dev 데드존(조향 0)
STOP_NEAR_Y     = 0.90       # 근접 정지 기준(ymax)

# 서보
PAN_ID = 0
TILT_ID = 1
PAN_CENTER  = 90
TILT_CENTER = 90
PAN_GAIN    = 105.0          # 흔들림 완화 위해 하향
TILT_GAIN   = 84.0
PAN_SMOOTH  = 0.20           # 서보 명령 EMA(이전값 가중치)
TILT_SMOOTH = 0.30
PAN_MIN, PAN_MAX   = 0, 180
TILT_MIN, TILT_MAX = 45, 120 # 90°(센터) 포함되도록 확장

# 상태 전이/검색
LOST_TIMEOUT = 3.0
SEARCH_SPEED = 50  # (좌스핀 제거했지만 남겨둠)

# --- 서보 흔들림 완화 ---
CENTER_EMA = 0.70            # 검출 중심 좌표 EMA(이전값 비중)
SERVO_UPDATE_HZ = 12         # 서보 갱신 빈도(Hz)
MAX_STEP_DEG = 2             # 프레임당 서보 각도 변화 상한
CENTER_DEADZONE_SERVO = 0.04 # 서보용 중심 데드존(정규화)

# --- 차동 조향 직진화 ---
CRUISE_DEADZONE = 0.18       # 이 안에선 완전 직진(좌우 동일 속도)
DELTA_EMA       = 0.60       # Δ(조향) 저역통과 EMA(이전값 비중)
MAX_DELTA_STEP  = 10.0       # 프레임당 Δ 변화 상한
MISS_ALLOW      = 2          # 연속 미검출 허용 프레임 수(히스테리시스)
# ----------------------------------------------------

class Follower:
    def __init__(self, car: YB_Pcb_Car):
        self.car = car
        self.pan = PAN_CENTER
        self.tilt = TILT_CENTER
        self.state = "SEARCHING"
        self.last_seen_ts = 0.0

        # 서보 좌표 스무딩
        self.xc_smooth = 0.5
        self.yc_smooth = 0.5
        self._servo_next_ts = 0.0

        # 조향 Δ 스무딩/리미트
        self.delta_smooth = 0.0
        self._last_delta  = 0.0

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

    # -------- 바퀴 제어(직진 중심) --------
    def drive_wheels(self, x_dev, y_max):
        # 근접 정지
        if y_max > STOP_NEAR_Y:
            self.stop()
            return "Stop"

        # ① 크루즈 창: 중앙 근처는 완전 직진
        if abs(x_dev) <= CRUISE_DEADZONE:
            self.delta_smooth = 0.0
            l = r = self._clamp(BASE_SPEED, MIN_SPEED, MAX_SPEED)
            self.car.Car_Run(l, r)
            return "Cruise(Ω)"

        # ② 원시 Δ
        raw_delta = float(Kx * x_dev)

        # ③ Δ 저역통과(EMA)
        delta_lp = DELTA_EMA * self.delta_smooth + (1.0 - DELTA_EMA) * raw_delta

        # ④ Δ 레이트 리미트
        step = delta_lp - self._last_delta
        if step >  MAX_DELTA_STEP:
            delta_lp = self._last_delta + MAX_DELTA_STEP
        elif step < -MAX_DELTA_STEP:
            delta_lp = self._last_delta - MAX_DELTA_STEP

        self._last_delta  = delta_lp
        self.delta_smooth = delta_lp

        # ⑤ 좌/우 속도 적용
        l = self._clamp(int(BASE_SPEED - delta_lp), MIN_SPEED, MAX_SPEED)
        r = self._clamp(int(BASE_SPEED + delta_lp), MIN_SPEED, MAX_SPEED)
        self.car.Car_Run(l, r)

        if delta_lp > 10:  return f"Fwd-R({r})"
        if delta_lp < -10: return f"Fwd-L({l})"
        return "Forward"

    # -------- 서보 제어(안정화 포함, 단일 정의) --------
    def aim_servos(self, x_center, y_center):
        """
        1) 검출 중심 좌표 EMA 스무딩
        2) 서보 중심 데드존
        3) 업데이트 주기 제한(SERVO_UPDATE_HZ)
        4) 목표각 계산 + 기존 EMA 완충
        5) 프레임당 각도 변화 레이트 리미트
        6) 각도 클램프 후 보드에 쓰기
        """
        # 1) 좌표 스무딩 (EMA)
        self.xc_smooth = CENTER_EMA * self.xc_smooth + (1.0 - CENTER_EMA) * x_center
        self.yc_smooth = CENTER_EMA * self.yc_smooth + (1.0 - CENTER_EMA) * y_center

        # 2) 데드존
        if abs(self.xc_smooth - 0.5) < CENTER_DEADZONE_SERVO and \
           abs(self.yc_smooth - 0.5) < CENTER_DEADZONE_SERVO:
            return int(self.pan), int(self.tilt)

        # 3) 업데이트 주기 제한
        now = time.monotonic()
        if now < self._servo_next_ts:
            return int(self.pan), int(self.tilt)
        self._servo_next_ts = now + (1.0 / SERVO_UPDATE_HZ)

        # 4) 목표각 계산(스무딩된 중심 사용) + 기존 EMA 완충
        pan_target  = PAN_CENTER + (0.5 - self.xc_smooth) * PAN_GAIN
        tilt_target = TILT_CENTER - (self.yc_smooth - 0.5) * TILT_GAIN

        new_pan  = (1 - PAN_SMOOTH)  * pan_target  + PAN_SMOOTH  * self.pan
        new_tilt = (1 - TILT_SMOOTH) * tilt_target + TILT_SMOOTH * self.tilt

        # 5) 레이트 리미트(프레임당 변화량 제한)
        def limit_step(curr, prev):
            delta = curr - prev
            if   delta >  MAX_STEP_DEG: curr = prev + MAX_STEP_DEG
            elif delta < -MAX_STEP_DEG: curr = prev - MAX_STEP_DEG
            return curr

        self.pan  = limit_step(new_pan,  self.pan)
        self.tilt = limit_step(new_tilt, self.tilt)

        # 6) 각도 클램프 + 명령
        pan_cmd  = int(self._clamp(round(self.pan),  PAN_MIN,  PAN_MAX))
        tilt_cmd = int(self._clamp(round(self.tilt), TILT_MIN, TILT_MAX))
        try:
            self.car.Ctrl_Servo(PAN_ID, pan_cmd)
            self.car.Ctrl_Servo(TILT_ID, tilt_cmd)
        except Exception as e:
            print("Servo write failed:", e)
        return pan_cmd, tilt_cmd

def main():
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

    miss_count = 0  # 연속 미검출 카운터(히스테리시스)

    try:
        while True:
            rgb = picam2.capture_array()
            h, w, _ = rgb.shape

            set_input_tensor(interpreter, rgb)
            interpreter.invoke()
            detections = get_output(interpreter, CONF_THRESH)

            best = max(
                [d for d in detections if labels.get(d["class_id"]) == TARGET_LABEL],
                key=lambda x: x["score"], default=None
            )

            status = "Idle"
            pan_cmd, tilt_cmd = follow.pan, follow.tilt

            if best is not None:
                miss_count = 0
                follow.state = "TRACKING"
                follow.last_seen_ts = time.monotonic()

                xmin, ymin, xmax, ymax = best["bbox"]
                x_center = (xmin + xmax) * 0.5
                y_center = (ymin + ymax) * 0.5
                x_dev = 0.5 - x_center

                status = follow.drive_wheels(x_dev, y_max=ymax)
                pan_cmd, tilt_cmd = follow.aim_servos(x_center, y_center)

            else:
                miss_count += 1

                # TRACKING 중이었고, 충분히 오래/여러 프레임 놓쳤을 때만 SEARCHING 전환
                if (follow.state == "TRACKING" and
                    miss_count > MISS_ALLOW and
                    (time.monotonic() - follow.last_seen_ts) > LOST_TIMEOUT):
                    print("Target lost (hysteresis). Entering SEARCH mode.")
                    follow.state = "SEARCHING"
                    follow.stop()

                if follow.state == "SEARCHING":
                    status = "Searching straight..."
                    # 시선은 중앙, 차체는 저속 직진 유지
                    pan_cmd, tilt_cmd = follow.aim_servos(0.5, 0.5)
                    car.Car_Run(MIN_SPEED, MIN_SPEED)

            # --- Rendering / HUD ---
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            label_text = "NO PERSON"
            if best:
                (xmin, ymin, xmax, ymax) = best["bbox"]
                x0, y0, x1, y1 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                label_text = f"LBL=person score={best['score']:.2f} ymax={ymax:.2f}"

            cv2.putText(bgr, f"STATUS: {status}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(bgr, label_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(bgr, f"PAN={int(pan_cmd)} TILT={int(tilt_cmd)}", (w - 230, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow(WINDOW, bgr)

            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

            # 선택: CPU 양보
            # time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        print("Cleaning up...")
        follow.stop()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
