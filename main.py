from tts import text_to_speech
from llm import ask_gemini

def main():
    print("chatbot - enter 'exit' for stop")
    while True:
        user_msg = input("user: ").strip()
        if user_msg.lower() == "exit":
            print("chatbot stops")
            break
        if not user_msg:
            continue

        print("chatbot: thinking...")
        reply = ask_gemini(user_msg)
        print("\n".join(reply.splitlines()))
        print()

        if not reply.startswith("[ERROR]"):
            text_to_speech(reply)

if __name__ == "__main__":
    main()
