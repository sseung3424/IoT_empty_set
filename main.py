# main.py
from dotenv import load_dotenv
load_dotenv()

import cv2
import threading
import time

from fall_det import FallDetector, cleanup
from tts import text_to_speech
from stt import speech_to_text
from llm import ask_gemini

# Robot control (motors/servo via I2C)
from yb_car import YB_Pcb_Car

# Tracking module (interface can vary across your codebase)
try:
    import tracking as tracking_mod
except Exception:
    tracking_mod = None

def init_tracker(car):
    """
    Try to initialize the tracking interface.
    Supported patterns:
      1) Class-based: tracking.PersonTracker([car]) with update(frame) or process_frame(frame)
      2) Function-based: tracking.process_frame(frame, [car])
    Returns: (tracker_obj, track_fn)
    """
    tracker_obj, track_fn = None, None
    if tracking_mod:
        if hasattr(tracking_mod, "PersonTracker"):
            try:
                tracker_obj = tracking_mod.PersonTracker(car=car)
            except TypeError:
                tracker_obj = tracking_mod.PersonTracker()
        elif hasattr(tracking_mod, "process_frame"):
            track_fn = tracking_mod.process_frame
    return tracker_obj, track_fn


def chatbot_loop(stop_event: threading.Event):
    """
    Voice chatbot loop.
    - STT for input
    - LLM for response
    - TTS for output
    - Say 'exit' to stop everything
    """
    print("chatbot - say 'exit' to stop")
    while not stop_event.is_set():
        try:
            user_msg = speech_to_text()
        except Exception as e:
            print(f"[STT ERROR] {e}")
            time.sleep(0.2)
            continue

        if not user_msg:
            time.sleep(0.05)
            continue

        print("user:", user_msg)

        if user_msg.strip().lower() == "exit":
            print("chatbot stops")
            stop_event.set()
            break

        print("chatbot: thinking...")
        try:
            reply = ask_gemini(user_msg)  # should return a string
        except Exception as e:
            reply = f"[ERROR] LLM failed: {e}"

        print("chatbot:\n" + (reply or ""))

        if reply and not reply.startswith("[ERROR]"):
            try:
                text_to_speech(reply)
            except Exception as e:
                print(f"[TTS ERROR] {e}")


def unified_camera_loop(stop_event: threading.Event,
                        camera_index: int = 0,
                        width: int = 640, height: int = 480, fps: int = 30,
                        show_window: bool = False):
    """
    Unified camera loop:
      - Single capture for both fall detection and tracking (no device conflicts)
      - Optional GUI window via `show_window`
      - Tracking is expected to command the robot through YB_Pcb_Car
    """
    car = YB_Pcb_Car()

    tracker_obj, track_fn = init_tracker(car)
    detector = FallDetector()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Camera {camera_index} open failed.")
        return

    # Reasonable performance settings for Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    time.sleep(0.5)  # warm-up

    if tracking_mod is None:
        print("[WARN] tracking module not found -> tracking disabled")
    elif tracker_obj is None and track_fn is None:
        print("[WARN] tracking interface not found -> tracking disabled")
    else:
        print("[INFO] tracking enabled")

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera read failed; retrying...")
                time.sleep(0.03)
                continue

            # 1) Fall detection
            label, fall = "NA", False
            try:
                label, fall = detector.process_frame(frame)
                if fall:
                    print("[ALERT] Fall detected!")
                    # Example safety action: stop robot
                    # car.Car_Stop()
            except Exception as e:
                print(f"[FallDet ERROR] {e}")

            # 2) Tracking (if available)
            try:
                if tracker_obj is not None:
                    if hasattr(tracker_obj, "update"):
                        tracker_obj.update(frame)
                    elif hasattr(tracker_obj, "process_frame"):
                        tracker_obj.process_frame(frame)
                    elif track_fn:
                        track_fn(frame, car)
                elif track_fn is not None:
                    try:
                        track_fn(frame, car)  # prefer passing car
                    except TypeError:
                        track_fn(frame)       # fallback
            except Exception as e:
                print(f"[Tracking ERROR] {e}")

            # 3) Optional GUI window for debugging
            if show_window:
                try:
                    cv2.putText(frame, str(label), (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Robot Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                except cv2.error:
                    # In headless environments, imshow may fail -> ignore
                    pass

            # Tiny sleep to avoid 100% CPU pegging
            time.sleep(0.001)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        try:
            cleanup()  # fall_det cleanup (should be idempotent)
        except Exception as e:
            print(f"[CLEANUP WARN] {e}")
        try:
            car.Car_Stop()  # ensure safe stop
        except Exception:
            pass
        print("camera loop stopped")


if __name__ == "__main__":
    # Shared stop signal across threads
    stop_event = threading.Event()

    # Single camera -> unified loop handles both fall detection & tracking
    cam_thread = threading.Thread(
        target=unified_camera_loop,
        kwargs={
            "stop_event": stop_event,
            "camera_index": 10,
            "width": 640, "height": 480, "fps": 30,
            "show_window": False  # set True when you connect a display or use VNC
        },
        daemon=True
    )
    cam_thread.start()

    # Run chatbot on main thread (Ctrl+C responsive)
    try:
        chatbot_loop(stop_event)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        try:
            cam_thread.join(timeout=1.0)
        except RuntimeError:
            pass
        print("all threads stopped")
