# vision_node.py
import os, time, json
import paho.mqtt.client as mqtt
import cv2

from dotenv import load_dotenv
load_dotenv()

from fall_detection import run_detection_from_capture

BROKER_HOST = os.getenv("BROKER_HOST", "127.0.0.1")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))
TOPIC_FALL  = "vision/fall_event"
TOPIC_STATUS= "vision/status"

# Robot-Pi가 쏘는 TCP 스트림 주소 (필요 시 IP/포트 맞추기)
ROBOT_STREAM_URL = os.getenv("ROBOT_STREAM_URL", "tcp://robotpi.local:8888")

cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Vision_Pi")
cli.will_set("status/visionpi", json.dumps({"online":False}), qos=1, retain=True)
cli.connect(BROKER_HOST, BROKER_PORT, 60)
cli.loop_start()
cli.publish("status/visionpi", json.dumps({"online":True}), qos=1, retain=True)

def on_fall_callback():
    """fall_detection.py에서 낙상 시 호출되는 콜백. 'OK'를 리턴하면 내부 FSM이 상태를 초기화."""
    payload = {"ts": time.time(), "type": "fall", "conf": 0.9}
    cli.publish(TOPIC_FALL, json.dumps(payload), qos=1)
    print("[Vision] fall_event published:", payload)
    return "OK"

def main():
    print("[Vision] opening stream:", ROBOT_STREAM_URL)
    cap = cv2.VideoCapture(ROBOT_STREAM_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Failed to open stream. Check Robot-Pi rpicam-vid and URL.")

    try:
        run_detection_from_capture(cap, on_fall=on_fall_callback)
    finally:
        cap.release()
        cli.publish("status/visionpi", json.dumps({"online":False}), qos=1, retain=True)
        cli.loop_stop(); cli.disconnect()

if __name__ == "__main__":
    main()
