# llm.py
import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Ensure main.py calls load_dotenv() and .env has the key.")

genai.configure(api_key=GEMINI_API_KEY)
_gemini = genai.GenerativeModel("gemini-1.5-flash-latest")

def ask_gemini(user_input: str) -> str:
    try:
        resp = _gemini.generate_content([
            "You are a kind chatbot for people who live alone or elderly people living by themselves. Please answer in a warm tone, in Korean, without using emojis, and keep your replies from being too long.",
            user_input
        ])
        return resp.text.strip()
    except Exception as e:
        return f"[ERROR] {e}"
