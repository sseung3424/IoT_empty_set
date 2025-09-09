# main.py
# -*- coding: utf-8 -*-
"""
Run three workers concurrently:
1) Tracking (camera+TFLite) -> publishes frames to BUS
2) Fall detection (YOLO-pose, low frequency) -> reads from BUS
3) Conversation (STT -> LLM -> TTS)
"""

import os, time, threading, signal, traceback
from queue import Queue
from dotenv import load_dotenv

# 공존성 향상
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

load_dotenv()

# ===== 옵션 =====
DEBUG             = int(os.environ.get("DEBUG", "1"))
MANUAL_INPUT      = int(os.environ.get("MANUAL_INPUT", "1"))   # 1: 키보드, 0: 마이크(STT)
STT_TIMEOUT       = float(os.environ.get("STT_TIMEOUT", "10")) # 0이면 타임아웃 미사용
HARD_EXIT_TIMEOUT = float(os.environ.get("HARD_EXIT_TIMEOUT", "2"))

# ===== 모듈 =====
from stt import speech_to_text
from tts import text_to_speech
from llm import ask_gemini

from tracking import run_tracking           # <- 위에서 만든 함수
from fall_worker import yolo_fall_loop      # <- 낙상 워커

EXIT_WORDS = {"exit", "quit", "stop", "종료", "끝", "그만"}

def log(msg: str):
    if DEBUG:
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] {msg}", flush=True)
    else:
        print(msg, flush=True)

# ---------------- STT 타임아웃 래퍼(필요시) ----------------
def stt_with_timeout(timeout_s: float) -> str:
    if timeout_s <= 0:
        return speech_to_text() or ""
    q: Queue = Queue(maxsize=1)
    errq: Queue = Queue(maxsize=1)
    def _w():
        try:
            q.put(speech_to_text() or "")
        except Exception as e:
            errq.put(e)
    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        log(f"[STT] timeout {timeout_s:.1f}s → skip this turn")
        return ""
    if not errq.empty():
        raise errq.get()
    return q.get() if not q.empty() else ""

# ---------------- 대화 루프 ----------------
def conv_loop(stop_event: threading.Event):
    log("conv start (type/say 'exit/quit/stop/종료/끝/그만' to end)")
    while not stop_event.is_set():
        try:
            if MANUAL_INPUT:
                try: user_text = input("User> ").strip()
                except EOFError: user_text = ""
            else:
                user_text = stt_with_timeout(STT_TIMEOUT).strip()

            if not user_text:
                continue

            log(f"[User] {user_text}")
            if user_text.lower() in EXIT_WORDS:
                log("[conv] exit command detected")
                stop_event.set(); break

            reply = ask_gemini(user_text)
            log(f"[LLM] {reply}")

            try: text_to_speech(reply)
            except Exception as e: log(f"[TTS] error: {e}")

        except Exception:
            log("[conv] error:"); traceback.print_exc(); time.sleep(0.2)

    log("conv terminated")

# ---------------- 엔트리 ----------------
def main():
    stop_event = threading.Event()
    shutting = {"count": 0}

    def _signal_handler(signum, frame):
        shutting["count"] += 1
        log(f"[main] signal {signum} ({shutting['count']})")
        stop_event.set()
        if shutting["count"] == 1:
            def _killer():
                t0 = time.time()
                while time.time() - t0 < HARD_EXIT_TIMEOUT and any(t.is_alive() for t in threads):
                    time.sleep(0.1)
                os._exit(0)
            threading.Thread(target=_killer, daemon=True).start()
        else:
            os._exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 1) 트래킹 스레드 (카메라 + 프레임 publish)
    track_thread = threading.Thread(target=run_tracking, args=(stop_event,), daemon=True, name="tracking")
    track_thread.start(); log("tracking thread started")

    # 2) 낙상 감지 스레드 (YOLO-pose)
    fall_thread = threading.Thread(target=yolo_fall_loop, args=(stop_event,), daemon=True, name="fall")
    fall_thread.start(); log("fall thread started")

    # 3) 대화 스레드 (STT→LLM→TTS)
    conv_thread = threading.Thread(target=conv_loop, args=(stop_event,), daemon=True, name="conv")
    conv_thread.start(); log("conv thread started")

    global threads
    threads = [track_thread, fall_thread, conv_thread]

    try:
        while not stop_event.is_set():
            if any(not t.is_alive() for t in threads):
                log("[main] a thread died → shutting down")
                stop_event.set(); break
            time.sleep(0.2)
    finally:
        log("[main] joining threads...")
        for t in threads:
            t.join(timeout=1.5)
        log("[main] bye")

if __name__ == "__main__":
    main()
