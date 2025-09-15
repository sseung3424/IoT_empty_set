# robot_node.py
import os, time, json
import paho.mqtt.client as mqtt

# === 환경 ===
BROKER_HOST = os.getenv("BROKER_HOST", "127.0.0.1")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))

TOPIC_TELE   = "robot/telemetry"
TOPIC_FALL   = "vision/fall_event"

# === 기존 추적 코드 가져오기 ===
# follow_person_final.py를 그대로 재사용합니다.
import threading
import cv2
from follow_person_final import main as follow_main
from follow_person_final import Follower, YB_Pcb_Car, load_labels, make_interpreter_cpu, set_input_tensor, get_output
from picamera2 import Picamera2

# 텔레메트리 공유용 (간단한 샘플)
_last_telemetry = {"status":"Idle","pan":0,"tilt":0,"fps":0.0}

# === MQTT ===
def on_connect(c,u,f,rc,props=None):
    print("[MQTT] connected rc=", rc)
    c.subscribe(TOPIC_FALL, qos=1)

def on_message(c,u,msg):
    try:
        data = json.loads(msg.payload.decode())
    except Exception:
        print("[MQTT] invalid json:", msg.payload[:64])
        return

    if msg.topic == TOPIC_FALL and data.get("type")=="fall" and float(data.get("conf",0))>=0.7:
        print("[EVENT] FALL detected → STOP MOTORS")
        try:
            car = _robot_ctx.get("car")
            if car: car.Car_Stop()
        except Exception as e:
            print("[ERR] stop:", e)
        # (선택) 여기서 tts 사용해 음성 경고도 가능

cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Robot_Pi")
cli.on_connect = on_connect
cli.on_message = on_message
cli.will_set("status/robotpi", json.dumps({"online":False}), qos=1, retain=True)
cli.connect(BROKER_HOST, BROKER_PORT, 60)
cli.loop_start()

_robot_ctx = {}

def run_follow_loop():
    """
    follow_person_final.py의 로직을 최대한 유지하면서
    주기적으로 텔레메트리만 발행합니다.
    """
    global _last_telemetry

    # 아래 내용은 follow_person_final.main()을 인라인으로 구현(원본 유지)
    import os, time, numpy as np
    from follow_person_final import (
        MODEL_DIR, MODEL_CPU, LABELS_TXT, CONF_THRESH, TARGET_LABEL,
        LOST_TIMEOUT, SEARCH_SPEED, Follower, YB_Pcb_Car, set_input_tensor, get_output, load_labels, make_interpreter_cpu
    )

    labels = load_labels(os.path.join(MODEL_DIR, LABELS_TXT))
    interpreter = make_interpreter_cpu(os.path.join(MODEL_DIR, MODEL_CPU))
    interpreter.allocate_tensors()
    car = YB_Pcb_Car()
    _robot_ctx["car"] = car
    follow = Follower(car)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    WINDOW = "Follower"
    t_fps = time.time(); frames = 0; fps = 0.0
    try:
        while True:
            rgb = picam2.capture_array()
            h, w, _ = rgb.shape
            set_input_tensor(interpreter, rgb)
            interpreter.invoke()
            detections = get_output(interpreter, CONF_THRESH)
            best = max([d for d in detections if labels.get(d["class_id"]) == TARGET_LABEL], 
                       key=lambda x: x["score"], default=None)

            status = "Idle"
            pan_cmd, tilt_cmd = follow.pan, follow.tilt
            
            if best is not None:
                follow.state = "TRACKING"
                follow.last_seen_ts = time.monotonic()
                xmin, ymin, xmax, ymax = best["bbox"]
                x_center = (xmin + xmax) * 0.5
                y_center = (ymin + ymax) * 0.5
                x_dev = 0.5 - x_center
                status = follow.drive_wheels(x_dev, y_max=ymax)
                pan_cmd, tilt_cmd = follow.aim_servos(x_center, y_center)
            else:
                if follow.state == "TRACKING" and time.monotonic() - follow.last_seen_ts > LOST_TIMEOUT:
                    print("Target lost too long → SEARCH")
                    follow.state = "SEARCHING"
                    follow.stop()
                if follow.state == "SEARCHING":
                    status = "Searching..."
                    pan_cmd, tilt_cmd = follow.aim_servos(0.5, 0.5)
                    car.Car_Spin_Left(SEARCH_SPEED, SEARCH_SPEED)

            # FPS 계산 및 텔레메트리 발행
            frames += 1
            if frames >= 20:
                now = time.time()
                fps = frames / (now - t_fps + 1e-6)
                t_fps, frames = now, 0
            _last_telemetry.update({"status":status, "pan":int(pan_cmd), "tilt":int(tilt_cmd), "fps":round(fps,1)})
            cli.publish(TOPIC_TELE, json.dumps(_last_telemetry), qos=0)

            # 디스플레이는 원본 코드처럼 유지 (원하면 끄기)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if best:
                (xmin, ymin, xmax, ymax) = best["bbox"]
                x0, y0, x1, y1 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
                cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(bgr, f"STATUS: {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(bgr, f"PAN={int(pan_cmd)} TILT={int(tilt_cmd)} FPS={fps:.1f}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow(WINDOW, bgr)
            if (cv2.waitKey(1) & 0xFF) == 27: # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("[Robot] cleanup")
        try:
            follow.stop()
        except: pass
        try:
            picam2.stop()
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 텔레메트리 초기값
    cli.publish("status/robotpi", json.dumps({"online":True}), qos=1, retain=True)
    try:
        run_follow_loop()
    finally:
        cli.publish("status/robotpi", json.dumps({"online":False}), qos=1, retain=True)
        cli.loop_stop(); cli.disconnect()
