# main.py
import cv2
import threading
from fall_det import FallDetector, cleanup
from tts import text_to_speech
from stt import speech_to_text
from llm import ask_gemini
from tracking import run_tracking_thread  # <-- tracking thread entrypoint

def fall_detection_loop(stop_event: threading.Event):
    """
    Video capture + fall detection loop.
    Uses camera index 0 by default.
    """
    cap = cv2.VideoCapture(0)
    detector = FallDetector()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            label, fall = detector.process_frame(frame)
            cv2.putText(frame, label, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if fall:
                print("[ALERT] Fall detected!")

            cv2.imshow("Fall Detection", frame)
            # Press 'q' to close this window and stop only this loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def chatbot_loop(stop_event: threading.Event):
    """
    Voice chatbot loop.
    """
    print("chatbot - say 'exit' to stop")
    while not stop_event.is_set():
        print("user: (say)")
        user_msg = speech_to_text()
        if not user_msg:
            continue

        print("user:", user_msg)
        if user_msg.lower() == "exit":
            print("chatbot stops")
            break

        print("chatbot: thinking...")
        reply = ask_gemini(user_msg)
        print("chatbot:", "\n".join(reply.splitlines()))
        print()

        if not reply.startswith("[ERROR]"):
            text_to_speech(reply)

if __name__ == "__main__":
    # Shared stop signal for all threads
    stop_event = threading.Event()

    # Start fall detection thread (camera 0)
    fall_thread = threading.Thread(
        target=fall_detection_loop,
        args=(stop_event,),
        daemon=True
    )
    fall_thread.start()

    # Start tracking thread (set camera_index accordingly)
    # If you only have one camera, either:
    #  - change camera_index to 0 and stop running fall detection on the same device, or
    #  - refactor to share frames from a single capture thread.
    track_thread = threading.Thread(
        target=run_tracking_thread,
        kwargs={"stop_event": stop_event, "camera_index": 1, "show_window": False},
        daemon=True
    )
    track_thread.start()

    # Run chatbot in the main thread to keep Ctrl+C responsive
    try:
        chatbot_loop(stop_event)
    finally:
        # Signal all threads to stop and wait briefly
        stop_event.set()
        fall_thread.join(timeout=1.0)
        track_thread.join(timeout=1.0)
        cleanup()
        print("all threads stopped")
