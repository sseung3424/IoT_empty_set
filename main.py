# conv, tracking, fall_det 세개 메인 멀티 스레딩하기 -> 과제

# main.py
import threading
import time
import signal
from dotenv import load_dotenv

load_dotenv()

# conv
from stt import speech_to_text
from tts import text_to_speech
from llm import ask_gemini

# tracking: 파일은 그대로 두고, 내부의 main()을 호출
import tracking


def conv_loop(stop_event: threading.Event):
    """STT → LLM → TTS 루프. 'exit/quit/stop' 음성 입력 시 전체 종료."""
    print("[conv] start (say 'exit/quit/stop' to end)")
    while not stop_event.is_set():
        try:
            user_text = speech_to_text()
            if not user_text:
                continue

            print(f"[User] {user_text}")
            if user_text.strip().lower() in ("exit", "quit", "stop"):
                print("[conv] exit command detected")
                stop_event.set()
                break

            reply = ask_gemini(user_text)
            print(f"[LLM] {reply}")
            text_to_speech(reply)

        except Exception as e:
            print(f"[conv] error: {e}")
            time.sleep(0.3)
    print("[conv] terminated")


def main():
    stop_event = threading.Event()

    # Ctrl+C 등 시그널 → stop_event
    def _on_signal(signum, frame):
        print(f"[main] signal {signum} → shutting down")
        stop_event.set()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # 스레드 생성
    conv_thread = threading.Thread(target=conv_loop, args=(stop_event,), daemon=True)
    # tracking.py는 수정하지 않았으므로, 그 안의 main()을 그대로 호출
    track_thread = threading.Thread(target=tracking.main, daemon=True)

    # 시작
    conv_thread.start()
    track_thread.start()

    print("=== CONV + TRACKING (threads) ===")
    print(" - 음성 'exit/quit/stop' → 전체 종료")
    print(" - tracking 창 ESC → tracking만 종료(다시 실행하려면 프로그램 재시작)")

    # 메인스레드는 종료 신호 대기
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        print("[main] waiting threads...")
        conv_thread.join(timeout=2.0)
        # track_thread는 daemon=True이므로 프로세스 종료 시 함께 종료됨
        print("[main] all done")


if __name__ == "__main__":
    main()
