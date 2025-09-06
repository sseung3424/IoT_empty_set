# main.py
import os
from dotenv import load_dotenv

load_dotenv()

# Import local modules
from stt import speech_to_text
from tts import text_to_speech
from llm import ask_gemini

def main():
    # Load environment variables (.env must contain GEMINI_API_KEY)

    print("=== Voice Chatbot (STT → LLM → TTS) ===")

    while True:
        # Step 1: Speech to text
        user_text = speech_to_text()
        if not user_text:
            print("[INFO] No speech detected. Try again.")
            continue

        print(f"[User] {user_text}")

        # Exit condition
        if user_text.strip().lower() in ("exit", "quit", "stop"):
            print("Exiting chatbot.")
            break

        # Step 2: Ask Gemini
        response = ask_gemini(user_text)
        print(f"[Gemini] {response}")

        # Step 3: Text to speech
        text_to_speech(response)

if __name__ == "__main__":
    main()
