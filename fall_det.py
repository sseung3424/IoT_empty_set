# ===== fall_det.py =====
# 실시간 낙상감지 + 웹 스트리밍(HTTP MJPEG)
# 접속: http://<라즈베리파이_IP>:8000/  (예: http://10.210.24.159:8000/)
import os
import threading
import atexit
from flask import Flask, Response, render_template_string

from fall_detection import (
    run_detection,
    _reset_detection_state as reset_detection_state,
    mjpeg_generator,    # ← fall_detection.py에서 stream_publish(...)가 호출되고 있어야 함
)
from send_alarm import say_are_you_ok, cleanup as buzzer_cleanup

# ---------------- Alarm flow ----------------
def _guard_flow():
    """낙상 최초 감지 시 부저 울림(또는 OK면 상태 리셋)."""
    try:
        result = say_are_you_ok()  # "ALERT" 또는 "OK"
        if result == "OK":
            reset_detection_state()
    except Exception as e:
        print(f"[guard] 오류: {e}")

def on_fall_async():
    threading.Thread(target=_guard_flow, daemon=True).start()

# ---------------- Web server (HTTP/HTTPS) ----------------
app = Flask(__name__)

INDEX = """
<!doctype html><html><head><meta charset="utf-8"><title>Pi Live</title>
<style>
  body{margin:0;background:#111;color:#eee;font-family:sans-serif}
  .wrap{max-width:960px;margin:0 auto;padding:8px}
  img{width:100%;max-width:960px;border-radius:12px}
</style></head>
<body><div class="wrap">
  <h2>Live</h2>
  <img src="/video.mjpg"/>
</div></body></html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX)

@app.route("/video.mjpg")
def video():
    # multipart/x-mixed-replace (MJPEG)
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def start_server(host="0.0.0.0", port=8000, cert=None, key=None):
    """
    cert/key가 있으면 HTTPS, 없으면 HTTP로 실행.
    """
    context = None
    if cert and key and os.path.exists(cert) and os.path.exists(key):
        context = (cert, key)
        print(f"[web] HTTPS serving at https://{host}:{port}")
    else:
        print(f"[web] HTTP serving at  http://{host}:{port}")
    # threaded=True: 감지 루프와 병렬 동작
    app.run(host=host, port=port, ssl_context=context, threaded=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    # 종료 시 GPIO 정리
    atexit.register(buzzer_cleanup)

    # 1) 웹 서버를 '먼저' 백그라운드로 시작
    # 기본: HTTP 8000
    threading.Thread(
        target=start_server,
        kwargs={"host":"0.0.0.0", "port":8000},  # ← HTTPS 쓰려면 아래 줄로 교체
        # kwargs={"host":"0.0.0.0", "port":8443, "cert":"cert.pem", "key":"key.pem"},
        daemon=True
    ).start()

    # 2) 실시간 감지 루프 시작(블로킹)
    run_detection(on_fall=on_fall_async)
