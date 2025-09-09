# main.py
import os
import time
import threading
import signal
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

# ====== 설정(환경변수) ======
DEBUG         = int(os.environ.get("DEBUG", "1"))           # 1: 상세 로그
MANUAL_INPUT  = int(os.environ.get("MANUAL_INPUT", "0"))    # 1: STT 대신 키보드 입력
TRACKING_ON   = int(os.environ.get("TRACKING_ON", "1"))     # 0: tracking 비활성화

# ====== 모듈 ======
from stt import speech_to_text
from tts import text_to_speech
from llm import ask_gemini
import tracking  # tracking.py는 수정하지 않음 (tracking.main 사용)

# ====== 로깅/측정 유틸 ======
def log(msg: str):
    if DEBUG:
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] {msg}")
    else:
        print(msg)

@contextmanager
def step(name: str):
    t0 = time.time()
    try:
        yield
    finally:
        if DEBUG:
            dt = (time.time() - t0) * 1000
            log(f"{name} done in {dt:.1f} ms")

# ====== 대화 루프 (메인 스레드에서 실행) ======
def conv_loop(stop_event: threading.Event):
    log("conv loop start (say 'exit/quit/stop' to end)")
    while not stop_event.is_set():
        try:
            # 1) 입력
            if MANUAL_INPUT:
                user_text = input("User> ").strip()
                if not user_text:
                    continue
            else:
                with step("STT"):
                    user_text = speech_to_text() or ""
            if not user_text:
                continue

            log(f"[User] {user_text}")

            # 종료 명령
            if user_text.lower() in ("exit", "quit", "stop"):
                log("exit command detected")
                stop_event.set()
                break

            # 2) LLM
            with step("LLM"):
                reply = ask_gemini(user_text)

            log(f"[LLM] {reply}")

            # 3) 출력(TTS)
            try:
                with step("TTS"):
                    text_to_speech(reply)
            except Exception as e:
                log(f"TTS error: {e} (skip)")

        except KeyboardInterrupt:
            log("KeyboardInterrupt in conv loop")
            stop_event.set()
            break
        except Exception as e:
            # conv 단계 어디서든 예외를 삼키지 않고 보여줌
            log(f"conv error: {e}")
            time.sleep(0.2)

    log("conv loop terminated")

# ====== 메인 ======
def main():
    stop_event = threading.Event()

    # 시그널 → 종료
    def _on_signal(signum, frame):
        log(f"signal {signum} → shutting down")
        stop_event.set()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # tracking은 백그라운드에서 실행(디버깅 방해 ↓)
    if TRACKING_ON:
        track_thread = threading.Thread(target=tracking.main, daemon=True)
        track_thread.start()
        log("tracking thread started (daemon)")
    else:
        log("tracking disabled (TRACKING_ON=0)")

    # conv는 메인 스레드에서 실행 → 디버깅 편의성 ↑
    try:
        conv_loop(stop_event)
    finally:
        log("waiting background threads...")
        # tracking은 daemon=True라 프로세스 종료와 함께 내려감
        log("all done")

if __name__ == "__main__":
    main()
