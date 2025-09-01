import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("check env file")

# Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Gemini API call function
def ask_gemini(user_input: str) -> str:
    try:
        response = model.generate_content(
            [
                "You are a kind chatbot for people who live alone or elderly people living by themselves. Please answer in a warm tone, without using emojis, and keep your replies from being too long.",
                user_input
            ]
        )
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {e}"