import os
from dotenv import load_dotenv
import google.generativeai as genai

# 환경 변수 불러오기 (api key는 .env 파일에 따로 저장해야함)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("환경 변수 GEMINI_API_KEY를 찾을 수 없습니다. .env 파일 확인!")

# Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Gemini API 호출 함수
def ask_gemini(user_input: str) -> str:
    try:
        response = model.generate_content(
            [
                "당신은 혼자 사는 사람이나 독거 노인을 위한 친절한 챗봇입니다. 이모티콘 사용하지 말고 너무 길게는 말고 따뜻한 어조로 대답해주세요.",
                user_input
            ]
        )
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {e}"