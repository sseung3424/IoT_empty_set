# main.py
import os
import time
import threading
import traceback
from queue import Queue
from dotenv import load_dotenv

load_dotenv()

# ======================= 환경 변수 (디버깅 편의) =======================
# 실행 모드: conv(기본) | tracking | both
MODE         = os.environ.get("MODE", "conv").lower()

# STT 대신 키보드 입력 사용: 1=사용(디버깅 기본), 0=마이크 사용
MANUAL_INPUT = int(os.environ.get("MANUAL_INPUT", "1"))

# STT 타임아웃(초). 0이면 타임아웃 미적용
STT_TIMEOUT  = float(os.environ.get("STT_TIMEOUT", "10"))

# LLM/TTS 건너뛰기 (속도/단계별 디버깅)
SKIP_LLM     = int(os.environ.get("SKIP_LLM", "0"))
SKIP_TTS     = int(os.environ.get("SKIP_TTS", "0"))

# 출력 로그 간단/상세
DEBUG        = int(os.environ.get("DEBUG", "1"))

# ======================= conv 모듈 =======================
from stt import speech_to_text
from tts import text_to_speech
from llm import ask_gemini

# ======================= tracking (수정 금지) =======================
import tracking  # tracking.main() 그대로 사용

# ======================= 유틸 =======================
EXIT_WORDS = {"exit", "quit", "stop", "종료", "끝", "그만"}

def log(msg: str):
    if DEBUG:
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] {msg}")
    else:
        print(msg)

def stt_with_timeout(timeout_s: float):
    """
    STT가 블로킹될 수 있으므로, 별도 스레드로 감싸서 timeout 적용.
    timeout 발생 시 빈 문자열 반환.
    예외 발생 시 예외를 다시 던져서 디버깅에 바로 보이게 함.
    """
    q: Queue = Queue(maxsize=1)
    err_q: Queue = Queue(maxsize=1)

    def _worker():
        try:
            txt = speech_to_text()
            q.put(txt)
        except Exception as e:
            err_q.put(e)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout_s)

    if th.is_alive():
        log(f"[STT] timeout ({timeout_s:.1f}s) → skip this turn")
        return ""  # 다음 루프로 넘어가서 계속 디버깅 가능

    if not err_q.empty():
        raise err_q.get()

    return q.get() if not q.empty() else ""

def read_user_text():
    """수동/마이크 입력을 통합 처리."""
    if MANUAL_INPUT:
        try:
            return input("User> ").strip()
        except EOFError:
            return ""
    else:
        if STT_TIMEOUT > 0:
            return (stt_with_timeout(STT_TIMEOUT) or "").strip()
        else:
            return (speech_to_text() or "").strip()

def is_exit(text: str) -> bool:
    return text.lower() in EXIT_WORDS

# ======================= conv 루프 (메인 스레드에서 실행) =======================
def conv_loop():
    log("conv loop start (exit/quit/stop/종료/끝/그만)")
    while True:
        try:
            user_text = read_user_text()
            if not user_text:
                continue

            log(f"[User] {user_text}")
            if is_exit(user_text):
                log("exit command detected → bye")
                break

            # LLM
            if SKIP_LLM:
                reply = f"(echo) {user_text}"
            else:
                t0 = time.time()
                reply = ask_gemini(user_text)
                log(f"LLM done in {(time.time()-t0)*1000:.1f} ms")

            log(f"[LLM] {reply}")

            # TTS
            if not SKIP_TTS:
                t0 = time.time()
                try:
                    text_to_speech(reply)
                    log(f"TTS done in {(time.time()-t0)*1000:.1f} ms")
                except Exception:
                    log("TTS error:")
                    traceback.print_exc()
            else:
                log("TTS skipped (SKIP_TTS=1)")

        except KeyboardInterrupt:
            log("KeyboardInterrupt → exit conv")
            break
        except Exception:
            log("conv error:")
            traceback.print_exc()
            # 오류가 나도 루프를 계속 돌아 디버깅에 유리
            time.sleep(0.2)

    log("conv loop terminated")

# ======================= tracking 백그라운드 =======================
def start_tracking_daemon_if_needed():
    if MODE in ("tracking", "both"):
        t = threading.Thread(target=tracking.main, daemon=True, name="tracking")
        t.start()
        log("tracking thread started (daemon)")
    else:
        log("tracking disabled (MODE≠tracking/both)")

# ======================= 엔트리포인트 =======================
def main():
    # tracking은 백그라운드(데몬)로 시작
    start_tracking_daemon_if_needed()

    # conv는 메인 스레드에서 실행 → 예외/트레이스가 콘솔에 바로 출력
    if MODE in ("conv", "both"):
        conv_loop()
    else:
        log("MODE=tracking → press Ctrl+C to quit")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            log("KeyboardInterrupt → exit")

    log("all done")

if __name__ == "__main__":
    main()
