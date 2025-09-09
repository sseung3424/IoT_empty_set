# main.py
# -*- coding: utf-8 -*-
"""
Run two loops concurrently with a GUI-safe structure:
1) Voice chatbot: STT -> LLM -> TTS  (worker thread)
2) Person follower: human_follower.main()  (MAIN thread for OpenCV GUI)

- Uses a shared stop_event for graceful shutdown.
- Patches cv2.waitKey in the follower (MAIN) so it exits cleanly when stop_event is set.
- Adds an STT timeout wrapper to avoid long blocking on mic input.
"""

import time
import threading
import queue
from dotenv import load_dotenv

# -------- Load environment (.env must contain GEMINI_API_KEY & Google creds) --------
load_dotenv()

# -------- Import local modules (voice chatbot) --------
from stt import speech_to_text        # your STT (sounddevice + Google STT)
from tts import text_to_speech        # your TTS (Google TTS -> aplay)
from llm import ask_gemini            # your LLM (Gemini)

# -------- Import follower module --------
# Assumes human_follower.py is in the same directory
import cv2
import importlib
follower_mod = importlib.import_module("human_follower")

# -------- STT timeout control --------
USE_STT_TIMEOUT = True
STT_TIMEOUT_SEC = 4.0  # seconds


def stt_with_timeout(timeout_sec: float) -> str | None:
    """
    Run speech_to_text() in a small worker thread and return within timeout_sec.
    Returns None on timeout or any STT error to keep the loop responsive.
    """
    q: "queue.Queue[str | None]" = queue.Queue(maxsize=1)

    def _run():
        try:
            q.put(speech_to_text())
        except Exception:
            q.put(None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        return q.get(timeout=timeout_sec)
    except queue.Empty:
        return None


def voice_chatbot_loop(stop_event: threading.Event):
    """STT -> LLM -> TTS loop (cooperates with stop_event)."""
    print("=== Voice Chatbot (STT → LLM → TTS) ===")
    while not stop_event.is_set():
        # 1) STT (with optional timeout to prevent long blocking)
        if USE_STT_TIMEOUT:
            user_text = stt_with_timeout(STT_TIMEOUT_SEC)
        else:
            user_text = speech_to_text()

        if not user_text:
            # No speech / timeout / error -> keep loop responsive
            time.sleep(0.2)
            continue

        print(f"[User] {user_text}")

        # Exit keywords from user voice
        if user_text.strip().lower() in ("exit", "quit", "stop"):
            print("[INFO] Exit requested by user. Shutting down...")
            stop_event.set()
            break

        # 2) LLM
        response = ask_gemini(user_text)
        print(f"[Gemini] {response}")

        # 3) TTS (Google TTS -> aplay, releases device after playback)
        try:
            text_to_speech(response)
        except Exception as e:
            print(f"[TTS] Error: {e}")

    print("[Voice] Loop ended.")


def _make_patched_waitKey(stop_event: threading.Event):
    """
    Create a patched cv2.waitKey that returns ESC (27) when stop_event is set.
    This lets the follower's while-loop break gracefully.
    """
    orig_waitKey = cv2.waitKey

    def patched_waitKey(delay: int):
        if stop_event.is_set():
            # Simulate ESC to break the follower loop
            return 27
        return orig_waitKey(delay)

    return patched_waitKey


def run_follower_in_main(stop_event: threading.Event):
    """
    Run human_follower.main() on the MAIN thread (GUI-safe).
    We patch cv2.waitKey so that when stop_event is set, follower exits cleanly.
    """
    print("=== Follower (human_follower) on MAIN thread ===")
    patched = _make_patched_waitKey(stop_event)
    orig_waitKey = cv2.waitKey
    cv2.waitKey = patched
    try:
        # follower's loop should call imshow + waitKey(1) regularly
        follower_mod.main()
    except Exception as e:
        print(f"[Follower] Error: {e}")
    finally:
        cv2.waitKey = orig_waitKey
        print("[Follower] Loop ended.")


def main():
    print("=== Multi-Runner: Chatbot (thread) + Follower (MAIN) ===")
    print("Tips:")
    print(" - Say 'exit' / 'quit' / 'stop' (via mic) to end both loops.")
    print(" - Or press Ctrl+C in the terminal.\n")

    stop_event = threading.Event()

    # Start voice chatbot as a background thread
    t_voice = threading.Thread(target=voice_chatbot_loop, args=(stop_event,), daemon=True)
    t_voice.start()

    try:
        # Run follower (camera/GUI) on MAIN thread for HighGUI/Qt safety
        run_follower_in_main(stop_event)
    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt. Stopping...")
        stop_event.set()
    finally:
        # Give voice thread a moment to exit gracefully
        for _ in range(50):
            if not t_voice.is_alive():
                break
            time.sleep(0.1)
        print("[MAIN] All done. Bye.")


if __name__ == "__main__":
    main()
