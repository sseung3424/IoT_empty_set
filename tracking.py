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
        ymin, xmin, ymax, xmax = boxes[i]  # (ymin, xmin, ymax, xmax)
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
CENTER_DEADZONE = 0.10
STOP_NEAR_Y     = 0.90

# 서보
PAN_ID = 0
TILT_ID = 1
PAN_CENTER  = 90
TILT_CENTER = 90
PAN_GAIN    = 105.0
TILT_GAIN   = 84.0
PAN_SMOOTH  = 0.20
TILT_SMOOTH = 0.30
PAN_MIN, PAN_MAX   = 0, 180
TILT_MIN, TILT_MAX = 45, 120

# 상태 전이/검색
LOST_TIMEOUT = 0.6
SEARCH_ENTER_TIMEOUT = 2.0

# --- 서보 흔들림 완화 ---
CENTER_EMA = 0.70
SERVO_UPDATE_HZ = 12
MAX_STEP_DEG = 2
CENTER_DEADZONE_SERVO = 0.04

# --- 차동 조향 직진화 ---
CRUISE_DEADZONE = 0.18
DELTA_EMA       = 0.60
MAX_DELTA_STEP  = 10.0

# --- 검색 패턴(Searching) : 수정 포인트 ---
SEARCH_SPIN_SPEED   = 45     # 회전 속도(좌/우 같은 값)
SPIN_RATE_DEG_PER_SEC_AT_100 = 6.0   # 경험치: 속도 100일 때 각속도(°/s)
SEARCH_FULL_TURN_MARGIN = 1.10       # 360°의 10% 여유(정확히 한 바퀴 보장)

# Step-and-Stare (모션블러 저감)
SEARCH_BURST_MS = 180   # 회전하는 구간(ms)
SEARCH_HOLD_MS  = 140   # 정지하여 탐지하는 구간(ms)

# 팬 스윕(옵션: 시선 훑기)
SEARCH_SWEEP_HZ   = 0.4
SEARCH_SWEEP_AMPL = 0.30
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

        # --- Searching 상태 변수들 (새로 추가) ---
        self.search_start_ts = 0.0
        self.search_dir = +1           # +1: 좌회전(L), -1: 우회전(R)
        self.spin_accum_deg = 0.0      # 누적 회전각(°)
        self._search_phase = "BURST"   # "BURST" or "HOLD"
        self._phase_ts = 0.0           # 현재 phase 시작 시각

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
        if y_max > STOP_NEAR_Y:
            self.stop()
            return "Stop"

        if abs(x_dev) <= CRUISE_DEADZONE:
            self.delta_smooth = 0.0
            l = r = self._clamp(BASE_SPEED, MIN_SPEED, MAX_SPEED)
            self.car.Car_Run(l, r)
            return "Cruise(Ω)"

        raw_delta = float(Kx * x_dev)
        delta_lp = DELTA_EMA * self.delta_smooth + (1.0 - DELTA_EMA) * raw_delta

        step = delta_lp - self._last_delta
        if step >  MAX_DELTA_STEP: delta_lp = self._last_delta + MAX_DELTA_STEP
        elif step < -MAX_DELTA_STEP: delta_lp = self._last_delta - MAX_DELTA_STEP

        self._last_delta  = delta_lp
        self.delta_smooth = delta_lp

        l = self._clamp(int(BASE_SPEED - delta_lp), MIN_SPEED, MAX_SPEED)
        r = self._clamp(int(BASE_SPEED + delta_lp), MIN_SPEED, MAX_SPEED)
        self.car.Car_Run(l, r)

        if delta_lp > 10:  return f"Fwd-R({r})"
        if delta_lp < -10: return f"Fwd-L({l})"
        return "Forward"

    # -------- 서보 제어(안정화 포함) --------
    def aim_servos(self, x_center, y_center):
        self.xc_smooth = CENTER_EMA * self.xc_smooth + (1.0 - CENTER_EMA) * x_center
        self.yc_smooth = CENTER_EMA * self.yc_smooth + (1.0 - CENTER_EMA) * y_center

        if abs(self.xc_smooth - 0.5) < CENTER_DEADZONE_SERVO and \
           abs(self.yc_smooth - 0.5) < CENTER_DEADZONE_SERVO:
            return int(self.pan), int(self.tilt)

        now = time.monotonic()
        if now < self._servo_next_ts:
            return int(self.pan), int(self.tilt)
        self._servo_next_ts = now + (1.0 / SERVO_UPDATE_HZ)

        pan_target  = PAN_CENTER + (0.5 - self.xc_smooth) * PAN_GAIN
        tilt_target = TILT_CENTER - (self.yc_smooth - 0.5) * TILT_GAIN

        new_pan  = (1 - PAN_SMOOTH)  * pan_target  + PAN_SMOOTH  * self.pan
        new_tilt = (1 - TILT_SMOOTH) * tilt_target + TILT_SMOOTH * self.tilt

        def limit_step(curr, prev):
            delta = curr - prev
            if   delta >  MAX_STEP_DEG: curr = prev + MAX_STEP_DEG
            elif delta < -MAX_STEP_DEG: curr = prev - MAX_STEP_DEG
            return curr

        self.pan  = limit_step(new_pan,  self.pan)
        self.tilt = limit_step(new_tilt, self.tilt)

        pan_cmd  = int(self._clamp(round(self.pan),  PAN_MIN,  PAN_MAX))
        tilt_cmd = int(self._clamp(round(self.tilt), TILT_MIN, TILT_MAX))
        try:
            self.car.Ctrl_Servo(PAN_ID, pan_cmd)
            self.car.Ctrl_Servo(TILT_ID, tilt_cmd)
        except Exception as e:
            print("Servo write failed:", e)
        return pan_cmd, tilt_cmd

    # -------- 검색 패턴(한 바퀴 보장 + step-and-stare) --------
    def searching_step(self):
        """
        - 한 사이클에 '한 바퀴(360° × margin)'를 꼭 돌며,
        - BURST(짧게 회전) ↔ HOLD(멈추고 탐지) 를 반복해 모션블러를 줄인다.
        - 팬은 완만히 좌↔우 스윕(옵션)
        """
        now = time.monotonic()
        # 팬 스윕(시선 훑기)
        t = now - self.search_start_ts
        x_center = 0.5 + SEARCH_SWEEP_AMPL * np.sin(2 * np.pi * SEARCH_SWEEP_HZ * t)
        self.aim_servos(x_center, 0.5)

        # 각속도(°/s) 추정
        rate_100 = SPIN_RATE_DEG_PER_SEC_AT_100  # speed=100 기준
        rate = rate_100 * (SEARCH_SPIN_SPEED / 100.0)

        # phase 전환 로직
        if self._phase_ts == 0.0:
            self._phase_ts = now  # 초기화

        elapsed_ms = (now - self._phase_ts) * 1000.0

        if self._search_phase == "BURST":
            # 회전 수행
            if self.search_dir > 0:
                self.car.Car_Spin_Left(SEARCH_SPIN_SPEED, SEARCH_SPIN_SPEED)
            else:
                self.car.Car_Spin_Right(SEARCH_SPIN_SPEED, SEARCH_SPIN_SPEED)

            # 누적 각도 업데이트
            dt = (elapsed_ms / 1000.0)
            self.spin_accum_deg += rate * dt
            # BURST 구간 종료 판단
            if elapsed_ms >= SEARCH_BURST_MS:
                self._search_phase = "HOLD"
                self._phase_ts = now
                self.car.Car_Stop()

        else:  # HOLD
            self.car.Car_Stop()
            if elapsed_ms >= SEARCH_HOLD_MS:
                self._search_phase = "BURST"
                self._phase_ts = now

        # 한 바퀴+마진을 돌았으면 방향 전환
        full_turn_deg = 360.0 * SEARCH_FULL_TURN_MARGIN
        if self.spin_accum_deg >= full_turn_deg:
            self.spin_accum_deg = 0.0
            self.search_dir *= -1  # 방향 전환
            return f"Searching: turn swap ({'L' if self.search_dir>0 else 'R'})"

        return f"Searching: {self._search_phase} ({'L' if self.search_dir>0 else 'R'})"

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
            now = time.monotonic()

            if best is not None:
                # ---- TRACKING ----
                follow.state = "TRACKING"
                follow.last_seen_ts = now

                xmin, ymin, xmax, ymax = best["bbox"]
                x_center = (xmin + xmax) * 0.5
                y_center = (ymin + ymax) * 0.5
                x_dev = 0.5 - x_center

                status = follow.drive_wheels(x_dev, y_max=ymax)
                pan_cmd, tilt_cmd = follow.aim_servos(x_center, y_center)

                # 검색 상태 변수 리셋(다음에 SEARCHING 들어갈 때 새로 시작)
                follow.spin_accum_deg = 0.0
                follow._search_phase = "BURST"
                follow._phase_ts = 0.0

            else:
                # ---- NO DETECTION ----
                dt_since_seen = now - follow.last_seen_ts

                if dt_since_seen <= LOST_TIMEOUT:
                    follow.state = "LOST"
                    follow.stop()
                    pan_cmd, tilt_cmd = follow.aim_servos(0.5, 0.5)
                    status = "Lost: hold & center"
                elif dt_since_seen <= SEARCH_ENTER_TIMEOUT:
                    follow.state = "LOST"
                    follow.stop()
                    pan_cmd, tilt_cmd = follow.aim_servos(0.5, 0.5)
                    status = "Lost: waiting search"
                else:
                    if follow.state != "SEARCHING":
                        follow.state = "SEARCHING"
                        follow.search_start_ts = now
                        follow.spin_accum_deg = 0.0
                        follow.search_dir = +1
                        follow._search_phase = "BURST"
                        follow._phase_ts = 0.0
                        follow.stop()
                    status = follow.searching_step()
                    pan_cmd, tilt_cmd = follow.pan, follow.tilt  # searching_step 내부에서 aim_servos 호출

            # --- Rendering / HUD ---
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            label_text = "NO PERSON"
            if best:
                (xmin, ymin, xmax, ymax) = best["bbox"]
                x0, y0, x1, y1 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                label_text = f"LBL=person score={best['score']:.2f} ymax={ymax:.2f}"

            cv2.putText(bgr, f"STATE: {follow.state} | STATUS: {status}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(bgr, label_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(bgr, f"PAN={int(pan_cmd)} TILT={int(tilt_cmd)}", (w - 230, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow(WINDOW, bgr)

            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        print("Cleaning up...")
        follow.stop()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
