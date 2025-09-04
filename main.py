#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI (ribbon) camera smoke test using Picamera2 + libcamera.
Keys:
  q : quit
  s : save snapshot to ./captures/
  r : start/stop recording to ./records/ (H.264 mp4)
"""

import time
import os
from pathlib import Path
import cv2

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def overlay_fps(frame, fps, res_text):
    text = f"{res_text} | FPS: {fps:4.1f}"
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

def main():
    snapshots_dir = Path("./captures"); ensure_dir(snapshots_dir)
    records_dir   = Path("./records");  ensure_dir(records_dir)

    picam = Picamera2()

    # Configure a reasonable 1280x720 preview
    preview_config = picam.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)},
        queue=True  # enable frame queueing for smoother capture
    )
    picam.configure(preview_config)

    # Optional camera controls (uncomment / tweak as needed)
    # picam.set_controls({"AeExposureMode": 0})  # 0=Normal, 3=Short, etc.
    # picam.set_controls({"AfMode": 1})          # 1=Continuous, 2=Manual
    # picam.set_controls({"AwbEnable": True})

    picam.start()
    win = "CSI Camera Test (q=quit, s=snapshot, r=record)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    # Recording state
    recording = False
    encoder = None
    ff_out = None

    last = time.time()
    count = 0
    fps = 0.0

    try:
        while True:
            frame = picam.capture_array()  # ndarray in RGB888
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # FPS
            count += 1
            now = time.time()
            dt = now - last
            if dt >= 0.5:
                inst = count / dt
                fps = 0.9 * fps + 0.1 * inst if fps else inst
                count = 0
                last = now

            h, w = frame.shape[:2]
            overlay_fps(frame, fps, f"{w}x{h}")
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = snapshots_dir / f"snapshot_{ts}.jpg"
                cv2.imwrite(str(out), frame)
                print(f"[SAVE] {out}")

            elif key == ord('r'):
                if not recording:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    mp4_path = records_dir / f"record_{ts}.mp4"
                    encoder = H264Encoder(bitrate=6_000_000)
                    ff_out = FfmpegOutput(str(mp4_path))
                    picam.start_recording(encoder, ff_out)
                    recording = True
                    print(f"[REC] start -> {mp4_path}")
                else:
                    picam.stop_recording()
                    recording = False
                    print("[REC] stop")
    finally:
        if recording:
            picam.stop_recording()
        picam.stop()
        cv2.destroyAllWindows()
        print("Camera test ended.")

if __name__ == "__main__":
    main()
