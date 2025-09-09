# fall_worker.py
# -*- coding: utf-8 -*-
"""
YOLO-pose fall detection worker.
- Reads latest frames from frame_bus.BUS (no camera open here)
- Low frequency to save CPU (default 0.8s)
"""

import os, time, math, numpy as np, cv2
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try: cv2.setNumThreads(1)
except Exception: pass

from ultralytics import YOLO
from frame_bus import BUS

# (옵션) 낙상시 음성 알림
FALL_TTS = int(os.getenv("FALL_TTS", "0"))
if FALL_TTS:
    try:
        from tts import text_to_speech as _tts
    except Exception:
        _tts = None
else:
    _tts = None

YOLO_MODEL  = os.getenv("YOLO_MODEL", "yolov8n-pose.pt")
YOLO_IMGSZ  = int(os.getenv("YOLO_IMGSZ", "320"))
FALL_PERIOD = float(os.getenv("FALL_PERIOD", "0.8"))  # seconds
ANGLE_THRESH_DEG = float(os.getenv("FALL_ANGLE_DEG", "25"))  # ~0°: 눕기
ASPECT_THRESH    = float(os.getenv("FALL_W_OVER_H", "1.25")) # 가로/세로
CONFIRM_FRAMES   = int(os.getenv("FALL_CONFIRM_N", "3"))

_model = None
def _lazy_load():
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL)

def _angle_deg(p1, p2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    return abs(math.degrees(math.atan2(dy, dx)))  # 90=upright, ~0=lying

def _is_fall(box_xyxy, kpts_xy):
    x0,y0,x1,y1 = box_xyxy
    w = max(1.0, x1-x0); h = max(1.0, y1-y0)
    aspect = w/h
    try:
        # COCO: 5 L-shoulder, 6 R-shoulder, 11 L-hip, 12 R-hip
        ls, rs, lh, rh = kpts_xy[5], kpts_xy[6], kpts_xy[11], kpts_xy[12]
        top = ((ls[0]+rs[0])*0.5, (ls[1]+rs[1])*0.5)
        bot = ((lh[0]+rh[0])*0.5, (lh[1]+rh[1])*0.5)
        ang = _angle_deg(top, bot)
    except Exception:
        ang = 90.0
    prone_like = (ang < ANGLE_THRESH_DEG)
    wide_like  = (aspect > ASPECT_THRESH)
    return prone_like and wide_like, ang, aspect

def yolo_fall_loop(stop_event):
    _lazy_load()
    last = 0.0; hit = 0; announced = False

    while not stop_event.is_set():
        now = time.monotonic()
        if now - last < FALL_PERIOD:
            time.sleep(0.01); continue
        last = now

        frame = BUS.latest()
        if frame is None:
            time.sleep(0.01); continue

        try:
            results = _model(frame, imgsz=YOLO_IMGSZ, device="cpu", verbose=False)
            r = results[0]

            fall = False; info = (999.0, 0.0)
            if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
                kxy = r.keypoints.xy.cpu().numpy()  # [N,17,2]
                bxy = r.boxes.xyxy.cpu().numpy()    # [N,4]
                for i in range(len(bxy)):
                    f, ang, asp = _is_fall(bxy[i], kxy[i])
                    if f: fall=True; info=(ang,asp); break
            elif r.boxes is not None and len(r.boxes) > 0:
                bxy = r.boxes.xyxy.cpu().numpy()
                for i in range(len(bxy)):
                    x0,y0,x1,y1 = bxy[i]
                    asp = max(1.0,x1-x0)/max(1.0,y1-y0)
                    if asp > ASPECT_THRESH:
                        fall=True; info=(999.0,asp); break

            hit = hit+1 if fall else max(0, hit-1)
            if hit >= CONFIRM_FRAMES and not announced:
                print(f"[FALL] DETECTED (angle≈{info[0]:.1f}°, w/h≈{info[1]:.2f})", flush=True)
                announced = True
                if _tts:
                    try: _tts("넘어짐이 감지되었습니다. 괜찮으신가요?")
                    except Exception as e: print(f"[FALL][TTS] error: {e}", flush=True)
            elif hit == 0 and announced:
                announced = False

        except Exception as e:
            print(f"[FALL] error: {e}", flush=True)
            time.sleep(0.05)
