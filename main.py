# main.py
# -*- coding: utf-8 -*-
"""
Run two loops concurrently with a GUI-safe structure:
1) Voice chatbot: STT -> LLM -> TTS  (worker thread)
2) Person follower: follower_mod.main()  (MAIN thread for OpenCV GUI)

Improvements:
- Explicit logging of Listening / timeout / recognized / TTS done.
- Longer STT timeout (12s) and option to disable.
- Line-buffered stdout so prints from threads appear immediately.
- Tiny sleep in patched waitKey to yield CPU to other threads.
"""

import os
import sys
import time
import threading
import queue
from dotenv import load_dotenv

# -------- Load environment (.env must contain GEMINI_API_KEY) --------
load_dotenv()

# Make stdout line-buffered so thread prints are visible immediately
try:
    sys.stdout.reconfigure(line_buffering=True)  # Py3.7+
except Exception:
    pass

def log(msg: str):
    print(msg, flush=True)

# -------- Import local modules (voice chatbot) --------
from stt import speech_to_text        # your STT function
from tts import text_to_speech        # your TTS function
from llm import ask_gemini            # your LLM function

# -------- Import follower module --------
# Set this to the actual module filename (without .py)
# e.g., "tracking" or "follow_person_final" or "fall_det"
FOLLOWER_MODULE = "fall_det"  # <- CHANGE THIS if your file is tracking.py etc.

import cv2
import importlib
follower_mod = importlib.import_module(FOLLOWER_MODULE)

# -------- STT timeout control --------
USE_STT_TIMEOUT = True
STT_TIMEOUT_SEC = 12.0  # seconds (increase from 4s)

def stt_with_timeout(timeout_sec: float) -> str | None:
    """
    Run speech_to_text() in a small worker thread and return within timeout_sec.
    Returns None on timeout or any STT error to keep the loop responsive.
    """
    q: "queue.Queue[str | None]" = queue.Queue(maxsize=1)

    def _run():
        try:
            q.put(speech_to_text())
        except Exception as e:
            log(f"[STT] Exception: {e}")
            q.put(None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        return q.get(timeout=timeout_sec)
    except queue.Empty:
        return None

def voice_chatbot_loop(stop_event: threading.Event):
    """STT -> LLM -> TTS loop (cooperates with stop_event)."""
    log("=== Voice Chatbot (STT → LLM → TTS) ===")
    while not stop_event.is_set():
        # 1) STT (with optional timeout to prevent long blocking)
        log("[Voice] Listening...")
        if USE_STT_TIMEOUT:
            user_text = stt_with_timeout(STT_TIMEOUT_SEC)
            if user_text is None:
                log("[Voice] STT timeout or no input detected.")
                time.sleep(0.1)
                continue
        else:
            try:
                user_text = speech_to_text()
            except Exception as e:
                log(f"[STT] Error: {e}")
                time.sleep(0.2)
                continue

        if not user_text:
            log("[Voice] Empty input.")
            time.sleep(0.1)
            continue

        log(f"[User] {user_text}")

        # Exit keywords from user voice
        if user_text.strip().lower() in ("exit", "quit", "stop"):
            log("[INFO] Exit requested by user. Shutting down...")
            stop_event.set()
            break

        # 2) LLM
        try:
            response = ask_gemini(user_text)
        except Exception as e:
            log(f"[LLM] Error: {e}")
            response = "Sorry, I had an issue generating a reply."
        log(f"[Gemini] {response}")

        # 3) TTS
        try:
            text_to_speech(response)
            log("[TTS] Playback done.")
        except Exception as e:
            log(f"[TTS] Error: {e}")

    log("[Voice] Loop ended.")

def _make_patched_waitKey(stop_event: threading.Event):
    """
    Create a patched cv2.waitKey that:
    - returns ESC (27) when stop_event is set (to break follower loop),
    - yields a tiny bit of time to other threads each call.
    """
    orig_waitKey = cv2.waitKey

    def patched_waitKey(delay: int):
        if stop_event.is_set():
            return 27
        ret = orig_waitKey(delay)
        # Tiny yield helps the voice thread get scheduled if CPU is saturated
        time.sleep(0.001)
        return ret

    return patched_waitKey

def run_follower_in_main(stop_event: threading.Event):
    """
    Run follower_mod.main() on the MAIN thread (GUI-safe).
    We patch cv2.waitKey so that when stop_event is set, follower exits cleanly.
    """
    log(f"=== Follower ({FOLLOWER_MODULE}.main) on MAIN thread ===")
    patched = _make_patched_waitKey(stop_event)
    orig_waitKey = cv2.waitKey
    cv2.waitKey = patched
    try:
        follower_mod.main()  # follower's loop should call imshow + waitKey(1)
    except Exception as e:
        log(f"[Follower] Error: {e}")
    finally:
        cv2.waitKey = orig_waitKey
        log("[Follower] Loop ended.")

def main():
    log("=== Multi-Runner: Chatbot (thread) + Follower (MAIN) ===")
    log("Tips:")
    log(" - Say 'exit' / 'quit' / 'stop' (via mic) to end both loops.")
    log(" - Or press Ctrl+C in the terminal.\n")

    stop_event = threading.Event()

    # Start voice chatbot as a background thread
    t_voice = threading.Thread(target=voice_chatbot_loop, args=(stop_event,), daemon=True)
    t_voice.start()
    log("[MAIN] Voice thread started.")

    try:
        # Run follower (camera/GUI) on MAIN thread for HighGUI/Qt safety
        run_follower_in_main(stop_event)
    except KeyboardInterrupt:
        log("\n[MAIN] KeyboardInterrupt. Stopping...")
        stop_event.set()
    finally:
        # Give voice thread a moment to exit gracefully
        for _ in range(50):
            if not t_voice.is_alive():
                break
            time.sleep(0.1)
        log("[MAIN] All done. Bye.")

if __name__ == "__main__":
    main()
