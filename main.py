from tts import text_to_speech
from llm import ask_gemini

def main():
    print("챗봇 - 종료하려면 '종료' 입력")
    while True:
        user_msg = input("사용자: ").strip()
        if user_msg.lower() == "종료":
            print("챗봇을 종료합니다.")
            break
        if not user_msg:
            continue

        print("챗봇: 생각중이에요...")
        reply = ask_gemini(user_msg)
        print("\n".join(reply.splitlines()))
        print()

        # 에러가 아니면 음성 출력
        if not reply.startswith("[ERROR]"):
            text_to_speech(reply)

if __name__ == "__main__":
    main()
