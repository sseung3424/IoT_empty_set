# main.py
import threading
import atexit
from fall_detection import run_detection, _reset_detection_state as reset_detection_state
from send_alarm import say_are_you_ok, cleanup as buzzer_cleanup
from web_stream import start_https_server


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

if __name__ == "__main__":
    # 종료 시 GPIO 정리
    atexit.register(buzzer_cleanup)

    # Picamera2를 사용한 실시간 감지 시작
    run_detection(on_fall=on_fall_async)
    # HTTPS MJPEG 서버 시작 (https://<Pi-IP>:8443)
    threading.Thread(target=start_https_server,
                    kwargs={"host":"0.0.0.0","port":8443,"cert":"cert.pem","key":"key.pem"},
                    daemon=True).start()
