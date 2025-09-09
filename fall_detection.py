# fall_detection.py
import os, time
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# ====== 환경/튜닝 ======
USE_DISPLAY = bool(int(os.environ.get("USE_DISPLAY", "1")))  # 0이면 창 미표시
CONF_THRES  = float(os.environ.get("CONF_THRES", "0.30"))
IMG_SIZE    = int(os.environ.get("IMG_SIZE", "320"))   # 256~384 권장
FRAME_SKIP  = int(os.environ.get("FRAME_SKIP", "2"))   # N프레임 중 1프레임만 추론
MODEL_PATH  = os.environ.get("MODEL_PATH", "yolov8n-pose.pt")

# 속도·자세 임계(윈도우 버전과 동일 로직)
MIN_ELAPSED_TIME_THRESHOLD = 5
VEL_MARGIN = 80.0

# ====== 상태 ======
previous_y_values = None
first_vel_threshold = None
second_vel_threshold = None
falling_threshold = None

fallen_state = False
taking_video = False
fall_start_time = None
fall_alerted = False

# ====== 유틸 ======
def _reset_detection_state():
    global previous_y_values, fallen_state, taking_video, fall_start_time, fall_alerted
    previous_y_values = None
    fallen_state = False
    taking_video = False
    fall_start_time = None
    fall_alerted = False
    print("[state] 초기화 완료")

def frame_coordinates(result):
    """첫 번째 인물의 keypoint y좌표 배열(NaN 포함)."""
    if result.keypoints is None or result.keypoints.xy is None or result.keypoints.xy.shape[0] == 0:
        return None
    kp_xy = result.keypoints.xy[0].cpu().numpy()
    y = kp_xy[:, 1].astype(float)
    if getattr(result.keypoints, "conf", None) is not None:
        conf = result.keypoints.conf[0].cpu().numpy()
        y[conf <= 0] = np.nan
    y[y == 0.0] = np.nan
    return y

def valid_count(arr):
    if arr is None: return 0
    return int(np.count_nonzero(~np.isnan(arr)))

def nan_range(arr):
    if arr is None or valid_count(arr) == 0: return 0.0
    return float(np.nanmax(arr) - np.nanmin(arr))

def velocity_from_prev(prev_y, curr_y, fps):
    return (curr_y - prev_y) * float(fps)

def init_thresholds_from_two_frames(res1, res2, fps_used):
    """두 프레임으로 자세/속도 임계 계산."""
    global falling_threshold, first_vel_threshold, second_vel_threshold
    y1 = frame_coordinates(res1)
    y2 = frame_coordinates(res2)
    if valid_count(y1) < 6 or valid_count(y2) < 6:
        return False
    falling_threshold = (nan_range(y1) * 2.0 / 3.0) + 20.0
    v12 = velocity_from_prev(y1, y2, fps_used)
    v0 = v12[0] if not np.isnan(v12[0]) else 0.0
    v5 = v12[5] if not np.isnan(v12[5]) else 0.0
    first_vel_threshold  = abs(v0) + VEL_MARGIN
    second_vel_threshold = abs(v5) + VEL_MARGIN
    print(f"[init] falling={falling_threshold:.1f}, v0_thr={first_vel_threshold:.1f}, v5_thr={second_vel_threshold:.1f}")
    return True

def check_falling_time_common():
    global fall_start_time, fall_alerted, taking_video, fallen_state
    if fall_start_time is None: return
    if (time.time() - fall_start_time) >= MIN_ELAPSED_TIME_THRESHOLD:
        print("[fall] ALERT! duration >= threshold")
        fall_alerted = True
        taking_video = True
        fall_start_time = None
        fallen_state = False

def check_falling(y_values, fps_used, on_fall=None, announce_flag=None):
    """윈도우에서 쓰던 속도·자세 기반 FSM."""
    global previous_y_values, fallen_state, fall_start_time, fall_alerted
    global first_vel_threshold, second_vel_threshold, falling_threshold

    if previous_y_values is not None and valid_count(y_values) >= 6 and valid_count(previous_y_values) >= 6:
        v_curr = velocity_from_prev(previous_y_values, y_values, fps_used)
        first_speed  = abs(v_curr[0]) if not np.isnan(v_curr[0]) else 0.0
        second_speed = abs(v_curr[5]) if not np.isnan(v_curr[5]) else 0.0
        range_y = nan_range(y_values)

        if (falling_threshold is not None) and (range_y <= falling_threshold):
            if (first_vel_threshold is not None) and (second_vel_threshold is not None) and \
               (first_speed <= first_vel_threshold) and (second_speed <= second_vel_threshold):
                if fallen_state:
                    check_falling_time_common()
            else:
                if not fallen_state:
                    fallen_state = True
                    fall_start_time = time.time()
                    print("[fall] start (velocity-based)")
                    if on_fall and announce_flag is not None and not announce_flag.get("done", False):
                        try:
                            result = on_fall()
                            if result == "OK":
                                _reset_detection_state()
                            else:
                                announce_flag["done"] = True
                        except Exception as e:
                            print(f"[on_fall] 오류: {e}")
                            announce_flag["done"] = True
                else:
                    check_falling_time_common()
        else:
            if fall_alerted:
                pass
            else:
                fallen_state = False
            if announce_flag is not None:
                announce_flag["done"] = False
            fall_start_time = None

    previous_y_values = y_values

# ====== Picamera2 열기 / 루프 ======
def _open_picam2(width=640, height=480):
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (int(width), int(height)), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    return picam2

def run_detection(on_fall=None):
    """Picamera2에서 프레임을 받아 실시간 감지."""
    global falling_threshold, first_vel_threshold, second_vel_threshold

    # 카메라 & FPS
    WIDTH, HEIGHT = 640, 480
    picam2 = _open_picam2(WIDTH, HEIGHT)

    # FPS 추정: 첫 30프레임에서 실측(초간단)
    print("[cam] warming up & measuring FPS...")
    t0 = time.time(); cnt = 0
    while cnt < 30:
        _ = picam2.capture_array()
        cnt += 1
    fps_used = max(10.0, 30.0/(time.time() - t0))  # 대략값
    print(f"[cam] estimated FPS ≈ {fps_used:.1f}")

    # 모델
    model = YOLO(MODEL_PATH)

    if USE_DISPLAY:
        cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Feed", 960, 540)

    # ===== 부트스트랩: 유효 프레임 2장으로 임계값 =====
    res_first, res_second = None, None
    for _ in range(200):  # 최댓값 제한
        frame = picam2.capture_array()  # RGB888
        res = model.predict(frame, conf=CONF_THRES, imgsz=IMG_SIZE, verbose=False)[0]
        yvals = frame_coordinates(res)
        if valid_count(yvals) >= 6:
            if res_first is None:
                res_first = res
            else:
                res_second = res
                if init_thresholds_from_two_frames(res_first, res_second, fps_used):
                    break
                else:
                    res_first, res_second = res_second, None
        if USE_DISPLAY:
            cv2.imshow("Video Feed", res.plot())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                picam2.stop(); cv2.destroyAllWindows(); return

    # 기본 임계값(혹시 못 잡았을 때)
    if falling_threshold is None:   falling_threshold = 400.0
    if first_vel_threshold is None: first_vel_threshold = 120.0
    if second_vel_threshold is None:second_vel_threshold = 120.0
    print(f"[init-check] fall_thr={falling_threshold:.1f}, v0_thr={first_vel_threshold:.1f}, v5_thr={second_vel_threshold:.1f}, fps≈{fps_used:.1f}, imgsz={IMG_SIZE}, skip={FRAME_SKIP}")

    # ===== 본 루프 =====
    announce_flag = {"done": False}
    frame_idx = 0
    try:
        while True:
            frame = picam2.capture_array()  # RGB888
            frame_idx += 1

            # 프레임 스킵
            do_infer = (frame_idx % FRAME_SKIP == 1)

            if do_infer:
                res = model.predict(frame, conf=CONF_THRES, imgsz=IMG_SIZE, verbose=False)[0]
                y_values = frame_coordinates(res)
                if valid_count(y_values) >= 6:
                    check_falling(y_values, fps_used, on_fall=on_fall, announce_flag=announce_flag)

                if USE_DISPLAY:
                    out = res.plot()
                    if fallen_state:
                        cv2.putText(out, "fall detected", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("Video Feed", out)
            else:
                if USE_DISPLAY:
                    out = frame.copy()
                    if fallen_state:
                        cv2.putText(out, "fall detected", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("Video Feed", out)

            if USE_DISPLAY and (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    finally:
        picam2.stop()
        if USE_DISPLAY:
            cv2.destroyAllWindows()
