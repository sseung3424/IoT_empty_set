# main.py
import os
import time
import threading
import signal
import sys
import os as _os
from dotenv import load_dotenv

load_dotenv()

# ===== 설정 =====
DEBUG            = int(os.environ.get("DEBUG", "1"))              # 디버그 로그
MANUAL_INPUT     = int(os.environ.get("MANUAL_INPUT", "0"))       # 1: STT 대신 키보드 입력
HARD_EXIT_TIMEOUT= float(os.environ.get("HARD_EXIT_TIMEOUT", "2"))# 우아한 종료 대기 후 강제 종료(초)
TRACKING_ON      = int(os.environ.get("TRACKING_ON", "1"))        # 0: tracking 비활성화

# ===== conv 모듈 =====
from stt import speech_to_text
from tts import text_to_speech
from llm import ask_gemini

# ===== tracking (수정 금지: 기존 tracking.main 사용) =====
import tracking


def log(msg: str):
    if DEBUG:
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] {msg}")
    else:
        print(msg)


def conv_loop(stop_event: threading.Event):
    """STT→LLM→TTS 루프. 블로킹될 수 있으니 별도 데몬 스레드에서 동작."""
    log("conv thread start (say 'exit/quit/stop' to end)")
    while not stop_event.is_set():
        try:
            # 1) 입력
            if MANUAL_INPUT:
                try:
                    user_text = input("User> ").strip()
                except EOFError:
                    user_text = ""
            else:
                user_text = speech_to_text() or ""

            if not user_text:
                continue

            log(f"[User] {user_text}")

            # 종료 명령
            if user_text.strip().lower() in ("exit", "quit", "stop"):
                log("exit command detected by conv")
                stop_event.set()
                break

            # 2) LLM
            reply = ask_gemini(user_text)
            log(f"[LLM] {reply}")

            # 3) 출력(TTS)
            try:
                text_to_speech(reply)
            except Exception as e:
                log(f"TTS error: {e} (skip)")

        except Exception as e:
            log(f"conv error: {e}")
            time.sleep(0.2)

    log("conv thread terminated")


def start_threads(stop_event: threading.Event):
    """conv / tracking 데몬 스레드 시작."""
    threads = []

    conv_thread = threading.Thread(target=conv_loop, args=(stop_event,), daemon=True, name="conv")
    conv_thread.start()
    threads.append(conv_thread)
    log("conv thread started (daemon)")

    if TRACKING_ON:
        track_thread = threading.Thread(target=tracking.main, daemon=True, name="tracking")
        track_thread.start()
        threads.append(track_thread)
        log("tracking thread started (daemon)")
    else:
        log("tracking disabled (TRACKING_ON=0)")

    return threads


def main():
    stop_event = threading.Event()
    shutting_down = {"count": 0}  # Ctrl+C 누른 횟수 기록

    def _graceful_or_hard_exit():
        """우아한 종료 대기 후 강제 종료."""
        log(f"graceful shutdown: waiting up to {HARD_EXIT_TIMEOUT:.1f}s...")
        t0 = time.time()
        while time.time() - t0 < HARD_EXIT_TIMEOUT:
            time.sleep(0.1)
        log("force exiting now")
        _os._exit(0)  # 하드 종료 (블로킹 STT/카메라가 있어도 무조건 종료)

    def _on_signal(signum, frame):
        shutting_down["count"] += 1
        log(f"signal {signum} received ({shutting_down['count']}) → shutting down")
        stop_event.set()
        # 첫 번째 신호: 우아한 종료 타이머 시작
        if shutting_down["count"] == 1:
            killer = threading.Thread(target=_graceful_or_hard_exit, daemon=True)
            killer.start()
        else:
            # 두 번째부터는 즉시 하드 종료
            log("immediate hard exit")
            _os._exit(1)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    threads = start_threads(stop_event)
    print("=== CONV + TRACKING (daemon threads) ===")
    print(" - 음성 'exit/quit/stop' → 종료 / Ctrl+C → 종료(2초 내 강제 종료)")
    if MANUAL_INPUT:
        print(" - MANUAL_INPUT=1: 키보드로 STT 대체 (User> 프롬프트)")

    # 메인 스레드는 종료 관리만 수행
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        log("main exiting (threads are daemons)")
        # 데몬 스레드는 프로세스 종료와 함께 내려감
        # 여기서 join을 강제하지 않음 (블로킹 방지)
        pass


if __name__ == "__main__":
    main()
