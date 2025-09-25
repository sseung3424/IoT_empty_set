# follow_person_final.py
import os
import time
import numpy as np
import cv2
from collections import deque
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

# --- 근접 정지(박스 높이 기반 + 히스테리시스) ---
# 초기엔 "2m쯤 거리"를 가정한 박스 높이(h_norm) 목표치로 시작(현장서 자동보정됨)
INIT_TARGET_FOLLOW_H = 0.38      # 2m 근처에서 사람 박스 높이(경험값, 자동보정보완)
NEAR_ENGAGE_FACTOR   = 1.18      # 멈춤 문턱 = target_h * 1.18
NEAR_RELEASE_FACTOR  = 0.95      # 해제 문턱 = target_h * 0.95

# 자동 보정(주행 중 측정) 파라미터
AUTO_TUNE_ENABLED  = True
TUNE_WARMUP_S      = 1.0         # Tracking 진입 후 워밍업 시간
TUNE_WINDOW_S      = 2.5         # 샘플 수집 시간
TUNE_MIN_SAMPLES   = 20
TUNE_MIN_H, TUNE_MAX_H = 0.08, 0.85  # 이상치 필터

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

# --- 검색 패턴(Searching) ---
SEARCH_SPIN_SPEED   = 45     # 회전 속도(좌/우 같은 값)
SPIN_RATE_DEG_PER_SEC_AT_100 = 6.0   # speed=100일 때 각속도(°/s)
SEARCH_FULL_TURN_MARGIN = 1.10       # 360°의 10% 여유

# Step-and-Stare (모션블러 저감)
SEARCH_BURST_MS = 180   # 회전하는 구간(ms)
SEARCH_HOLD_MS  = 140   # 정지하여 탐지하는 구간(ms)

# 팬 스윕(서보 각도 기준) —— 요청: ±90°
PAN_SWEEP_AMPL_DEG = 90.0
SEARCH_SWEEP_HZ    = 0.35    # 넓은 각이면 살짝 더 느리게
# ----------------------------------------------------

class Follower:
    def __init__(self, car: YB_Pcb_Car):
        self.car = car
        self.pan = PAN_CENTER
        self.tilt = TILT_CENTER
        self.state = "SEARCHING"
        self.last_seen_ts = 0.0

        # 서보 좌표 스무딩(추적용)
        self.xc_smooth = 0.5
        self.yc_smooth = 0.5
        self._servo_next_ts = 0.0

        # 조향 Δ 스무딩/리미트
        self.delta_smooth = 0.0
        self._last_delta  = 0.0

        # 근접 히스테리시스 (동적 문턱)
        self.target_follow_h = INIT_TARGET_FOLLOW_H
        self.near_engage = self.target_follow_h * NEAR_ENGAGE_FACTOR
        self.near_release = self.target_follow_h * NEAR_RELEASE_FACTOR
        self.near = False

        # 거리 자동보정 상태
        self.tune_active = False
        self.tune_start_ts = 0.0
        self.tune_buf = deque(maxlen=512)

        # Searching 상태 변수
        self.search_start_ts = 0.0
        self.search_dir = +1           # +1: 좌, -1: 우
        self.spin_accum_deg = 0.0
        self._search_phase = "BURST"   # "BURST" or "HOLD"
        self._phase_ts = 0.0

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

    # ---------- 거리 자동보정 ----------
    def _maybe_start_tune(self):
        if not AUTO_TUNE_ENABLED or self.tune_active:
            return
        self.tune_active = True
        self.tune_start_ts = time.monotonic()
        self.tune_buf.clear()

    def _maybe_collect_tune(self, h_norm, move_status):
        """Cruise/Forward 중에만 샘플 수집."""
        if not self.tune_active:
            return
        now = time.monotonic()
        if now - self.tune_start_ts < TUNE_WARMUP_S:
            return
        if TUNE_MIN_H <= h_norm <= TUNE_MAX_H and move_status:
            self.tune_buf.append(h_norm)

    def _maybe_finish_tune(self):
        if not self.tune_active:
            return
        now = time.monotonic()
        if (now - self.tune_start_ts) >= (TUNE_WARMUP_S + TUNE_WINDOW_S) and len(self.tune_buf) >= TUNE_MIN_SAMPLES:
            med = float(np.median(self.tune_buf))
            # 업데이트(안정적으로 약간의 여유 둠)
            self.target_follow_h = med
            self.near_engage  = self.target_follow_h * NEAR_ENGAGE_FACTOR
            self.near_release = self.target_follow_h * NEAR_RELEASE_FACTOR
            # 범위 안전망
            self.near_engage  = float(self._clamp(self.near_engage, 0.10, 0.90))
            self.near_release = float(self._clamp(self.near_release, 0.05, self.near_engage - 0.02))
            self.tune_active = False  # 1회 보정(원하면 주기적 재보정으로 바꿀 수 있음)

    # -------- 바퀴 제어(직진 중심) --------
    def drive_wheels(self, x_dev, ymin, ymax):
        # 근접 정지 판단: 박스 높이 기반 + 히스테리시스(동적 문턱)
        h_norm = float(max(0.0, ymax - ymin))
        if self.near:
            if h_norm < self.near_release:
                self.near = False  # 근접 해제 → 주행 가능
            else:
                self.stop()
                return "Stop(near)"
        else:
            if h_norm > self.near_engage:
                self.near = True
                self.stop()
                return "Stop(near)"

        # ① 크루즈 창: 중앙 근처는 완전 직진
        if abs(x_dev) <= CRUISE_DEADZONE:
            self.delta_smooth = 0.0
            l = r = self._clamp(BASE_SPEED, MIN_SPEED, MAX_SPEED)
            self.car.Car_Run(l, r)
            return "Cruise(Ω)"

        # ②~④ Δ 계산(EMA + 레이트 리미트)
        raw_delta = float(Kx * x_dev)
        delta_lp = DELTA_EMA * self.delta_smooth + (1.0 - DELTA_EMA) * raw_delta
        step = delta_lp - self._last_delta
        if step >  MAX_DELTA_STEP: delta_lp = self._last_delta + MAX_DELTA_STEP
        elif step < -MAX_DELTA_STEP: delta_lp = self._last_delta - MAX_DELTA_STEP
        self._last_delta  = delta_lp
        self.delta_smooth = delta_lp

        # ⑤ 좌/우 속도 적용
        l = self._clamp(int(BASE_SPEED - delta_lp), MIN_SPEED, MAX_SPEED)
        r = self._clamp(int(BASE_SPEED + delta_lp), MIN_SPEED, MAX_SPEED)
        self.car.Car_Run(l, r)

        if delta_lp > 10:  return f"Fwd-R({r})"
        if delta_lp < -10: return f"Fwd-L({l})"
        return "Forward"

    # -------- 서보 제어(추적용: 이미지 좌표→각도) --------
    def aim_servos(self, x_center, y_center):
        # 추적 시엔 이미지 좌표 기반 + 데드존/EMA/주기/레이트리미트 적용
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

        return self._apply_servo_targets(pan_target, tilt_target)

    # -------- 서칭용: 목표 각도로 직접 지시(±PAN_SWEEP_AMPL_DEG 보장) --------
    def aim_servos_to_angles(self, pan_target_deg, tilt_target_deg=None):
        # 데드존/좌표 EMA를 우회하고, 동일한 완충/레이트리미트/클램프만 적용
        if tilt_target_deg is None:
            tilt_target_deg = self.tilt  # 틸트는 유지
        # 서보 갱신 주기 제한
        now = time.monotonic()
        if now < self._servo_next_ts:
            return int(self.pan), int(self.tilt)
        self._servo_next_ts = now + (1.0 / SERVO_UPDATE_HZ)
        return self._apply_servo_targets(pan_target_deg, tilt_target_deg)

    # -------- 공통: 완충(EMA) + 레이트리미트 + 클램프 + 쓰기 --------
    def _apply_servo_targets(self, pan_target, tilt_target):
        # 완충(EMA)
        new_pan  = (1 - PAN_SMOOTH)  * pan_target  + PAN_SMOOTH  * self.pan
        new_tilt = (1 - TILT_SMOOTH) * tilt_target + TILT_SMOOTH * self.tilt

        # 레이트 리미트
        def limit_step(curr, prev):
            delta = curr - prev
            if   delta >  MAX_STEP_DEG: curr = prev + MAX_STEP_DEG
            elif delta < -MAX_STEP_DEG: curr = prev - MAX_STEP_DEG
            return curr

        self.pan  = limit_step(new_pan,  self.pan)
        self.tilt = limit_step(new_tilt, self.tilt)

        # 클램프 + 쓰기
        pan_cmd  = int(self._clamp(round(self.pan),  PAN_MIN,  PAN_MAX))
        tilt_cmd = int(self._clamp(round(self.tilt), TILT_MIN, TILT_MAX))
        try:
            self.car.Ctrl_Servo(PAN_ID, pan_cmd)
            self.car.Ctrl_Servo(TILT_ID, tilt_cmd)
        except Exception as e:
            print("Servo write failed:", e)
        return pan_cmd, tilt_cmd

    # -------- 검색 패턴(한 바퀴 보장 + step-and-stare + ±90° 팬 스윕) --------
    def searching_step(self):
        now = time.monotonic()
        # 팬을 정확히 ±PAN_SWEEP_AMPL_DEG로 스윕
        t = now - self.search_start_ts
        pan_target = PAN_CENTER + PAN_SWEEP_AMPL_DEG * np.sin(2 * np.pi * SEARCH_SWEEP_HZ * t)
        # 범위 안전(0~180) 내로 들어오도록 보장
        pan_target = self._clamp(pan_target, PAN_MIN, PAN_MAX)
        self.aim_servos_to_angles(pan_target, self.tilt)

        # 각속도(°/s) 추정
        rate_100 = SPIN_RATE_DEG_PER_SEC_AT_100
        rate = rate_100 * (SEARCH_SPIN_SPEED / 100.0)

        # phase 전환
        if self._phase_ts == 0.0:
            self._phase_ts = now
        elapsed_ms = (now - self._phase_ts) * 1000.0

        if self._search_phase == "BURST":
            # 회전
            if self.search_dir > 0:
                self.car.Car_Spin_Left(SEARCH_SPIN_SPEED, SEARCH_SPIN_SPEED)
            else:
                self.car.Car_Spin_Right(SEARCH_SPIN_SPEED, SEARCH_SPIN_SPEED)
            # 누적 각도
            dt = (elapsed_ms / 1000.0)
            self.spin_accum_deg += rate * dt
            if elapsed_ms >= SEARCH_BURST_MS:
                self._search_phase = "HOLD"
                self._phase_ts = now
                self.car.Car_Stop()
        else:
            self.car.Car_Stop()
            if elapsed_ms >= SEARCH_HOLD_MS:
                self._search_phase = "BURST"
                self._phase_ts = now

        # 한 바퀴+마진 완료 시 방향 전환
        full_turn_deg = 360.0 * SEARCH_FULL_TURN_MARGIN
        if self.spin_accum_deg >= full_turn_deg:
            self.spin_accum_deg = 0.0
            self.search_dir *= -1
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

                status = follow.drive_wheels(x_dev, ymin=ymin, ymax=ymax)
                pan_cmd, tilt_cmd = follow.aim_servos(x_center, y_center)

                # ------ 거리 자동보정(주행 중일 때) ------
                h_norm = float(max(0.0, ymax - ymin))
                is_moving = status.startswith("Cruise") or status.startswith("Forward") or status.startswith("Fwd-")
                if AUTO_TUNE_ENABLED and is_moving and not follow.near:
                    follow._maybe_start_tune()
                    follow._maybe_collect_tune(h_norm, move_status=True)
                    follow._maybe_finish_tune()
                else:
                    # 정지/근접/탐색 중에는 수집하지 않음
                    pass

                # 검색 상태 변수 리셋
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
                    pan_cmd, tilt_cmd = follow.pan, follow.tilt

            # --- Rendering / HUD ---
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            label_text = "NO PERSON"
            if best:
                (xmin, ymin, xmax, ymax) = best["bbox"]
                x0, y0, x1, y1 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                h_norm = (ymax - ymin)
                label_text = f"LBL=person score={best['score']:.2f} h={h_norm:.2f}"

            hud_state = f"STATE: {follow.state} | STATUS: {status}"
            hud_dist  = f"h*={follow.target_follow_h:.2f} NE({follow.near_engage:.2f}) NR({follow.near_release:.2f})"
            cv2.putText(bgr, hud_state, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(bgr, label_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(bgr, hud_dist, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
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
