# pip install tflite-runtime opencv-python
import os, time, cv2, numpy as np
import tflite_runtime.interpreter as tflite
from collections import deque

# ---------------- Camera config (Yahboom style) ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

TEST_CAM_ONLY = False  # set True to test camera only (like Yahboom sample)

if TEST_CAM_ONLY:
    t_start = time.time()
    fps = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(frame, 50, 150)

        fps += 1
        mfps = fps / (time.time() - t_start + 1e-9)
        cv2.putText(frame, "FPS " + str(int(mfps)), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        cv2.imshow('Canny', canny)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

# ---------------- TFLite MoveNet setup ----------------
MODEL_PATH = "movenet_singlepose_lightning.tflite"  # put your tflite model here
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

# ---------------- Fall detection state ----------------
HIST = 30
centroid_hist = deque(maxlen=HIST)
angle_hist = deque(maxlen=HIST)
aspect_hist = deque(maxlen=HIST)
ts_hist = deque(maxlen=HIST)

state = "Standing"
fall_time = None

def preprocess(frame):
    """Resize & normalize frame for MoveNet."""
    ih, iw = frame.shape[:2]
    inp = cv2.resize(frame, (256, 256))
    inp = inp.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)
    return inp, (iw, ih)

def run_movenet(frame):
    """Run MoveNet and return keypoints as (17, 3) => (x, y, score) in pixel coords."""
    inp, (iw, ih) = preprocess(frame)
    interpreter.set_tensor(in_details[0]['index'], inp)
    interpreter.invoke()
    kp = interpreter.get_tensor(out_details[0]['index'])
    keypoints = []
    for j in range(17):
        y = kp[0, 0, j, 0] * ih
        x = kp[0, 0, j, 1] * iw
        s = kp[0, 0, j, 2]
        keypoints.append((x, y, s))
    return np.array(keypoints)

def torso_angle_and_bbox(kp):
    """Compute torso angle to vertical and bbox aspect ratio from confident keypoints."""
    def pick(idx): return kp[idx][:2], kp[idx][2]
    (ls, ls_s), (rs, rs_s) = pick(5), pick(6)
    (lh, lh_s), (rh, rh_s) = pick(11), pick(12)

    conf_ok = min(ls_s, rs_s, lh_s, rh_s) > 0.3
    if conf_ok:
        shoulder = (ls + rs) / 2
        hip = (lh + rh) / 2
    else:
        (lk, lk_s), (rk, rk_s) = pick(13), pick(14)
        shoulder = (lh + rh) / 2
        if min(lk_s, rk_s) > 0.2:
            hip = (lk + rk) / 2
        else:
            hip = shoulder

    vec = hip - shoulder
    v = vec / (np.linalg.norm(vec) + 1e-6)
    cosang = v[1]  # dot with (0,1)
    theta = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    pts = np.array([p[:2] for p in kp if p[2] > 0.2])
    if len(pts) >= 3:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        aspect = h / (w + 1e-6)
        cx, cy = x + w / 2, y + h / 2
    else:
        aspect, (cx, cy) = 1.0, (0.0, 0.0)
    return theta, aspect, (cx, cy)

def update_state(theta, aspect, cy, tstamp, img_h=480):
    """Finite state machine for fall detection with simple thresholds."""
    global state, fall_time

    angle_hist.append(theta)
    aspect_hist.append(aspect)
    centroid_hist.append(cy)
    ts_hist.append(tstamp)

    if len(centroid_hist) >= 3:
        dt = ts_hist[-1] - ts_hist[-3]
        vy = (centroid_hist[-1] - centroid_hist[-3]) / dt if dt > 0 else 0.0
    else:
        vy = 0.0

    ANG = theta > 60.0
    FLAT = aspect < 0.6
    FAST = abs(vy) > 0.6 * img_h  # pixels/sec threshold
    STILL = False
    if len(ts_hist) >= 5:
        dur = ts_hist[-1] - ts_hist[0]
        if dur >= 1.8:
            dy = max(centroid_hist) - min(centroid_hist)
            STILL = dy < 0.02 * img_h  # <2% of image height

    if state == "Standing":
        if ANG and FAST:
            state = "Suspicious"
            fall_time = tstamp
    elif state == "Suspicious":
        if (ANG and FLAT) and STILL and (tstamp - fall_time >= 2.0):
            state = "Fallen"
        elif not ANG and not FAST:
            state = "Standing"
    elif state == "Fallen":
        if (not ANG) and aspect > 0.9 and (not STILL):
            state = "Standing"
    return state

# ---------------- Main loop (Yahboom-style HUD/FPS/ESC) ----------------
t_start = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    kps = run_movenet(frame)
    theta, aspect, (cx, cy) = torso_angle_and_bbox(kps)
    cur_state = update_state(theta, aspect, cy, t0, img_h=int(cap.get(4) or 480))

    # Optional: draw minimal bbox from keypoints for visualization
    pts = np.array([p[:2] for p in kps if p[2] > 0.2])
    if len(pts) >= 3:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # FPS text (same style as Yahboom)
    fps += 1
    mfps = fps / (time.time() - t_start + 1e-9)
    cv2.putText(frame, "FPS " + str(int(mfps)), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # State HUD
    cv2.putText(frame, f"state={cur_state}  theta={theta:.1f}  r(h/w)={aspect:.2f}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
