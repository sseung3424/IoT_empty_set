# follow_user.py
# -*- coding: utf-8 -*-
"""
Raspberry Pi + YDLIDAR + YB_Pcb_Car: Automatic User Following
Goal: Maintain distance = 1.0 m, angle = 0° (straight tracking)
- LiDAR: /dev/ttyUSB0 (change PORT if needed)
- Target estimation: robust minimum (close-cut / sector gating / smoothing)
- Control: Distance PI + Angle P → Left/Right PWM
- Output: ANG=0~360° (for display), ERRθ=±180° (for control), R[m]
"""

import time, math
from collections import deque
import ydlidar
from YB_Pcb_Car import YB_Pcb_Car

# ========================== LiDAR Configuration ==========================
PORT = "/dev/ttyUSB0"   # On Windows use "COMx"
COMBOS = [
    (115200, True,  ydlidar.TYPE_TRIANGLE),
    (230400, True,  ydlidar.TYPE_TRIANGLE),
    (230400, False, ydlidar.TYPE_TRIANGLE),
    (128000, True,  ydlidar.TYPE_TRIANGLE),
]

CLOCKWISE        = False   # True if LiDAR angle increases clockwise
ANGLE_OFFSET_DEG = 0.0     # Mount offset (0° = front correction)
R_MIN_M          = 0.25    # Cut-off near LiDAR housing
R_MAX_M          = 20.0
WARMUP_SEC       = 0.8
FIRST_TRIES      = 20

# Target candidate gating
INIT_SECTOR   = 80.0   # Initial frame: ±INIT_SECTOR° around front
TRACK_SECTOR  = 40.0   # After: ±TRACK_SECTOR° around last angle
ROBUST_P      = 0.20   # Lower p-quantile for initial guess
ROBUST_BAND_M = 0.20   # Only search min within band around quantile
R_BLIND_M     = 0.25   # Blind zone cut
R_FOCUS_MAX   = 2.5    # Maximum distance of interest (limit candidates)

# ========================== Driving Control ==========================
SET_R        = 1.00    # Target distance [m]
KP_R         = 140.0   # Distance P gain [PWM/m]
KI_R         = 30.0    # Distance I gain [PWM/(m·s)]
KP_TH_DEG    = 1.6     # Angle P gain [PWM/deg] (±180° range)
INTEGRATOR_CLAMP = 120.0

MAX_PWM      = 170     # Wheel max PWM (0~255 recommended)
MIN_MOVE_PWM = 50      # Min PWM to overcome motor deadzone
V_DEAD_R     = 0.03    # Distance deadband [m]
TH_DEAD_DEG  = 3.0     # Angle deadband [deg]
STEER_DIR    = +1      # Steering direction (+1 default, -1 if reversed)

NO_TARGET_STOP_MS = 800   # Stop if no target detected for ms
OBST_STOP_R       = 0.20  # Immediate stop if too close obstacle

PRINT_EVERY_N = 5    # Log interval (frames)

# ========================== Utility ==========================
def norm_deg(a): return ((a + 180.0) % 360.0) - 180.0
def to_0_360(a): return (a % 360.0 + 360.0) % 360.0
def rad2deg_if_needed(a): return math.degrees(a) if abs(a) <= 6.5 else a

def circ_mean_deg(deg_list):
    if not deg_list: return None
    s = sum(math.sin(math.radians(d)) for d in deg_list)
    c = sum(math.cos(math.radians(d)) for d in deg_list)
    return norm_deg(math.degrees(math.atan2(s, c)))

def safe_point(p):
    a = None
    for name in ("angle","degree","theta","azimuth"):
        if hasattr(p, name):
            a = float(getattr(p, name)); break
    if a is None: return None
    r = None
    for name in ("range","distance","dist","radius"):
        if hasattr(p, name):
            r = float(getattr(p, name)); break
    if r is None: return None
    a_deg = norm_deg(rad2deg_if_needed(a))
    return a_deg, r

def infer_units_from_values(vals):
    vals = [r for r in vals if r is not None and 0 < r < 1e9]
    if not vals: return 'm'
    vals.sort(); p95 = vals[min(len(vals)-1, int(len(vals)*0.95))]
    return 'mm' if p95 > 50.0 else 'm'

def infer_units_from_samples(samples):
    vals = [r for (_, r) in samples if r is not None and 0 < r < 1e9]
    if not vals: return 'm'
    vals.sort(); p95 = vals[min(len(vals)-1, int(len(vals)*0.95))]
    return 'mm' if p95 > 50.0 else 'm'

# ========================== Scan Normalization ==========================
class ScanNormalizer:
    def __init__(self):
        self.range_unit = None
    def _apply_angle(self, a_deg):
        if CLOCKWISE: a_deg = -a_deg
        return norm_deg(a_deg + ANGLE_OFFSET_DEG)
    def _from_arrays(self, scan):
        try:
            ranges = list(scan.ranges); angles = list(scan.angles)
        except Exception:
            return None
        if not ranges or not angles or len(ranges) != len(angles):
            return None
        if self.range_unit is None:
            self.range_unit = infer_units_from_values(ranges)
            print(f"[unit] inferred from arrays: {self.range_unit}")
        to_m = (1/1000.0) if self.range_unit == 'mm' else 1.0
        out = []
        for r, a in zip(ranges, angles):
            if r is None or not (r == r) or r <= 0 or r > 1e9: continue
            a_deg = self._apply_angle(norm_deg(rad2deg_if_needed(a)))
            r_m = r * to_m
            if r_m < R_MIN_M or r_m > R_MAX_M: continue
            out.append((a_deg, r_m))
        return out
    def _from_points(self, scan):
        try: pts = scan.points
        except Exception: return None
        if not pts: return []
        raw, probe = [], []
        for p in pts:
            sp = safe_point(p)
            if sp is None: continue
            a_deg, r_raw = sp
            if not (0 < r_raw < 1e9): continue
            raw.append((a_deg, r_raw)); probe.append((a_deg, r_raw))
        if self.range_unit is None:
            self.range_unit = infer_units_from_samples(probe)
            print(f"[unit] inferred from points: {self.range_unit}")
        to_m = (1/1000.0) if self.range_unit == 'mm' else 1.0
        out = []
        for a_deg, r_raw in raw:
            a_deg = self._apply_angle(a_deg)
            r_m = r_raw * to_m
            if r_m < R_MIN_M or r_m > R_MAX_M: continue
            out.append((a_deg, r_m))
        return out
    def normalize(self, scan):
        data = self._from_arrays(scan)
        if data is None: data = self._from_points(scan)
        return data if data is not None else []

# ========================== Target Candidate (Robust Minimum) ==========================
def robust_min_in_sector(points, center_deg=None, half_span_deg=None):
    """
    points: list[(θdeg, r_m)]
    center=None → ±INIT_SECTOR° around front, else ±half_span around center
    Return: (r_min, θ_at_min) or None
    """
    if not points: return None

    # Sector selection
    if center_deg is None:
        lo, hi = -INIT_SECTOR, +INIT_SECTOR
        sel = [(th, r) for (th, r) in points if lo <= th <= hi]
    else:
        lo, hi = center_deg - half_span_deg, center_deg + half_span_deg
        if lo <= -180 or hi >= 180:
            sel = []
            for th, r in points:
                d = norm_deg(th - center_deg)
                if -half_span_deg <= d <= +half_span_deg:
                    sel.append((th, r))
        else:
            sel = [(th, r) for (th, r) in points if lo <= th <= hi]
    if not sel: return None

    # Distance limit & blind cut
    sel = [(th, r) for (th, r) in sel if (r >= R_BLIND_M and r <= R_FOCUS_MAX)]
    if not sel: return None

    # Quantile approximation → restrict to band
    rs = sorted(r for (_, r) in sel)
    k = max(0, min(len(rs)-1, int(len(rs)*ROBUST_P)))
    q = rs[k]
    band_lo = max(R_MIN_M, q - ROBUST_BAND_M)
    band_hi = min(R_FOCUS_MAX, q + ROBUST_BAND_M)
    cand = [(th, r) for (th, r) in sel if band_lo <= r <= band_hi] or sel

    th_min, r_min = None, math.inf
    for th, r in cand:
        if r < r_min:
            r_min, th_min = r, th
    return (r_min, th_min) if th_min is not None else None

class RollingPolar:
    def __init__(self, len_r=5, len_th=7):
        self.buf_r = deque(maxlen=max(1, int(len_r)))
        self.buf_th = deque(maxlen=max(1, int(len_th)))
    def update(self, r, th):
        self.buf_r.append(r); self.buf_th.append(th)
        sr = sorted(self.buf_r)
        r_med = sr[len(sr)//2] if len(sr)%2 else 0.5*(sr[len(sr)//2-1] + sr[len(sr)//2])
        th_mean = circ_mean_deg(list(self.buf_th))
        return r_med, th_mean

# ========================== Vehicle Control Wrapper ==========================
class CarController:
    def __init__(self):
        self.car = YB_Pcb_Car()
        self.ri = 0.0  # Distance integrator
        self.last_target_ms = 0

    def stop(self):
        try: self.car.Car_Stop()
        except: pass

    def drive_pwm(self, v_pwm, w_pwm):
        """v_pwm: forward/back (+forward), w_pwm: steering (+left turn), both in PWM scale"""
        w_pwm *= STEER_DIR
        base = abs(v_pwm)
        turn = w_pwm

        # Deadzone compensation
        if base > 0:
            base = max(MIN_MOVE_PWM, min(MAX_PWM, base))
        turn = max(-MAX_PWM, min(MAX_PWM, turn))

        left  = int(max(0, min(MAX_PWM, base - turn)))
        right = int(max(0, min(MAX_PWM, base + turn)))

        if base < MIN_MOVE_PWM and abs(turn) > MIN_MOVE_PWM*0.6:
            # Almost in-place rotation (add small forward motion)
            left  = int(max(0, min(MAX_PWM, MIN_MOVE_PWM - turn)))
            right = int(max(0, min(MAX_PWM, MIN_MOVE_PWM + turn)))
            v_pwm = max(v_pwm, 0)

        if v_pwm >= 0:
            self.car.Car_Run(left, right)
        else:
            self.car.Car_Back(left, right)

    def control_to(self, r, th_deg):
        """
        r [m], th_deg ∈ [-180,180): Control inputs
        Goal: r→SET_R, th→0
        """
        # Distance PI
        e_r = (r - SET_R)
        self.ri += e_r * LOOP_DT
        self.ri = max(-INTEGRATOR_CLAMP/KI_R if KI_R>0 else 0.0,
                      min(INTEGRATOR_CLAMP/KI_R if KI_R>0 else 0.0,
                          self.ri))
        v_cmd = KP_R*e_r + (KI_R*self.ri if KI_R>0 else 0.0)

        # Angle P
        e_th = th_deg
        w_cmd = KP_TH_DEG * e_th

        # Deadbands
        if abs(e_r) < V_DEAD_R:  v_cmd = 0.0
        if abs(e_th) < TH_DEAD_DEG: w_cmd = 0.0

        self.drive_pwm(v_cmd, w_cmd)

# ========================== Main Loop ==========================
LOOP_DT = 0.05  # Control period [s] ≈ 20 Hz

def run_once(port, baud, single_channel, lidar_type):
    print(f"\n=== LiDAR {port} @ {baud} | single_channel={single_channel} | type={lidar_type} ===")
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, baud)
    laser.setlidaropt(ydlidar.LidarPropLidarType, lidar_type)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    for opt,val in [
        ("LidarPropSingleChannel", bool(single_channel)),
        ("LidarPropIgnoreBaseInfo", True),
        ("LidarPropHeartBeat", False),
        ("LidarPropSupportMotorDtrCtrl", False),
        ("LidarPropFixedResolution", False),
    ]:
        try: laser.setlidaropt(getattr(ydlidar, opt), val)
        except Exception: pass

    if not laser.initialize():
        print("❌ initialize failed")
        try: laser.turnOff(); laser.disconnecting()
        except Exception: pass
        return False

    if not laser.turnOn():
        print("❌ turnOn failed")
        try: laser.turnOff(); laser.disconnecting()
        except Exception: pass
        return False

    time.sleep(WARMUP_SEC)
    scan = ydlidar.LaserScan()
    norm = ScanNormalizer()
    filt = RollingPolar()
    car  = CarController()

    # First frame
    for _ in range(FIRST_TRIES):
        if laser.doProcessSimple(scan): break
        time.sleep(0.03)

    last_center = None
    last_seen_ms = 0
    frame = 0
    try:
        while True:
            t0 = time.time()
            if not laser.doProcessSimple(scan):
                car.stop()
                time.sleep(LOOP_DT)
                continue

            pts = norm.normalize(scan)   # [(θdeg, r_m)]
            # Safety stop if very close obstacle in ±20°
            if any(r < OBST_STOP_R for (_, r) in pts if abs(_) <= 20):
                car.stop(); time.sleep(LOOP_DT); continue

            # Target candidate
            if last_center is None:
                cand = robust_min_in_sector(pts, center_deg=None, half_span_deg=None)
            else:
                cand = robust_min_in_sector(pts, center_deg=last_center, half_span_deg=TRACK_SECTOR)
                if cand is None:  # fallback
                    cand = robust_min_in_sector(pts, center_deg=None, half_span_deg=None)

            if cand is None:
                # Stop if no detection for too long
                if (time.time()*1000 - last_seen_ms) > NO_TARGET_STOP_MS:
                    car.stop()
                time.sleep(LOOP_DT)
                continue

            r_raw, th_raw = cand
            r_s, th_s = filt.update(r_raw, th_raw)  # smoothing
            last_center = th_s
            last_seen_ms = time.time()*1000

            th_ctrl = norm_deg(th_s)
            th_disp = to_0_360(th_s)

            car.control_to(r_s, th_ctrl)

            # Log
            frame += 1
            if frame % PRINT_EVERY_N == 0:
                print(f"[trk] R={r_s:4.2f} m | ANG={th_disp:6.1f}° | errθ={th_ctrl:+6.1f}°")

            # Loop time control
            t_elapsed = time.time() - t0
            if t_elapsed < LOOP_DT:
                time.sleep(LOOP_DT - t_elapsed)

    except KeyboardInterrupt:
        print("⏹ Interrupted")
    finally:
        try: car.stop()
        except: pass
        try: laser.turnOff()
        except Exception: pass
        try: laser.disconnecting()
        except Exception: pass
        print("LiDAR & Motors stopped.")
    return True

def main():
    try: ydlidar.os_init()
    except Exception: pass

    for baud, sc, ltype in COMBOS:
        ok = run_once(PORT, baud, sc, ltype)
        if ok:
            try: ydlidar.os_cleanup()
            except Exception: pass
            return

    print("🚫 All combinations failed: check port/power/cable/model/options")
    try: ydlidar.os_cleanup()
    except Exception: pass

if __name__ == "__main__":
    main()
