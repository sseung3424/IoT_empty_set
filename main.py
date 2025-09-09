# main.py
# -*- coding: utf-8 -*-
"""
Run two loops concurrently:
1) Voice chatbot: STT -> LLM -> TTS
2) Person follower: human_follower.main()

- Uses a shared stop_event for graceful shutdown.
- Monkey-patches cv2.waitKey in the follower thread so it exits cleanly
  when stop_event is set (simulates ESC key).
"""

import os
import threading
import time
from dotenv import load_dotenv

# -------- Load environment (.env must contain GEMINI_API_KEY) --------
load_dotenv()

# -------- Import local modules (voice chatbot) --------
from stt import speech_to_text        # your STT function
from tts import text_to_speech        # your TTS function
from llm import ask_gemini            # your LLM function

# -------- Import follower module --------
# Assumes human_follower.py is in the same directory
import cv2
import importlib
follower_mod = importlib.import_module("human_follower")


def voice_chatbot_loop(stop_event: threading.Event):
    """STT -> LLM -> TTS loop (cooperates with stop_event)."""
    print("=== Voice Chatbot (STT → LLM → TTS) ===")
    while not stop_event.is_set():
        # 1) STT
        user_text = speech_to_text()
        if not user_text:
            print("[INFO] No speech detected. Try again.")
            # Small sleep to avoid tight loop on mic errors
            time.sleep(0.2)
            continue

        print(f"[User] {user_text}")

        # Exit keywords from user voice
        if user_text.strip().lower() in ("exit", "quit", "stop"):
            print("[INFO] Exit requested by user. Shutting down all loops...")
            stop_event.set()
            break

        # 2) LLM
        response = ask_gemini(user_text)
        print(f"[Gemini] {response}")

        # 3) TTS
        text_to_speech(response)

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


def follower_loop(stop_event: threading.Event):
    """Run human_follower.main() with cv2.waitKey patched for cooperative exit."""
    print("=== Follower Thread (human_follower) ===")
    # Patch cv2.waitKey in this thread's context
    patched = _make_patched_waitKey(stop_event)
    orig_waitKey = cv2.waitKey
    cv2.waitKey = patched
    try:
        # Run the follower main loop (blocks until ESC or error)
        follower_mod.main()
    except Exception as e:
        print(f"[Follower] Error: {e}")
    finally:
        # Restore original waitKey to avoid side effects
        cv2.waitKey = orig_waitKey
        print("[Follower] Loop ended.")


def main():
    print("=== Multi-Runner: Chatbot + Follower ===")
    print("Tips:")
    print(" - Say 'exit' / 'quit' / 'stop' (via mic) to end both loops.")
    print(" - Or press Ctrl+C in the terminal.\n")

    stop_event = threading.Event()

    # Start follower first (camera/motors), then voice bot (mic/speaker)
    t_follower = threading.Thread(target=follower_loop, args=(stop_event,), daemon=True)
    t_voice    = threading.Thread(target=voice_chatbot_loop, args=(stop_event,), daemon=True)

    t_follower.start()
    t_voice.start()

    try:
        # Wait for either thread to finish; keep main alive
        while t_follower.is_alive() or t_voice.is_alive():
            time.sleep(0.3)
            if stop_event.is_set():
                break
    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt. Stopping...")
        stop_event.set()
    finally:
        # Give threads a moment to exit gracefully
        for _ in range(30):
            if not (t_follower.is_alive() or t_voice.is_alive()):
                break
            time.sleep(0.1)

        print("[MAIN] All done. Bye.")

if __name__ == "__main__":
    main()
