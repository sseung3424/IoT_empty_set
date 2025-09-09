# send_alarm.py
import os
import time
import RPi.GPIO as GPIO

# ====== 설정 (환경변수로 변경 가능) ======
# 모드: BOARD(물리핀 번호) 또는 BCM
GPIO_MODE = os.environ.get("GPIO_MODE", "BOARD").upper()  # "BOARD" or "BCM"

# 핀: BOARD 모드 기본 32(=BCM 12), BCM 모드 기본 18
BUZZER_PIN = int(os.environ.get("BUZZER_PIN",
                                "32" if GPIO_MODE == "BOARD" else "18"))

# PWM 주파수(Hz)와 패턴
BUZZER_FREQ = int(os.environ.get("BUZZER_FREQ", "440"))    # 기본 A4
BEEP_COUNT  = int(os.environ.get("BEEP_COUNT", "3"))       # 몇 번 울릴지
BEEP_ON_MS  = int(os.environ.get("BEEP_ON_MS", "180"))     # 각 비프 길이(ms)
BEEP_OFF_MS = int(os.environ.get("BEEP_OFF_MS", "120"))    # 비프 사이 간격(ms)
DUTY        = int(os.environ.get("DUTY", "50"))            # 듀티(%) 0~100

# 경고 패턴을 톤 스윕으로 바꾸고 싶다면 아래를 1로
USE_SWEEP   = int(os.environ.get("USE_SWEEP", "0"))        # 0:고정톤, 1:스윕

# ====== GPIO 초기화 ======
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD if GPIO_MODE == "BOARD" else GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

_pwm = GPIO.PWM(BUZZER_PIN, BUZZER_FREQ)
_pwm_started = False

def _ensure_pwm_started():
    global _pwm_started
    if not _pwm_started:
        _pwm.start(0)
        _pwm_started = True

def _beep_once(freq_hz: int, on_ms: int, duty: int):
    _ensure_pwm_started()
    _pwm.ChangeFrequency(freq_hz)
    _pwm.ChangeDutyCycle(max(0, min(100, duty)))
    time.sleep(on_ms / 1000.0)
    _pwm.ChangeDutyCycle(0)

def _beep_sweep(start_hz: int, end_hz: int, duration_ms: int, steps: int = 20):
    """start→end로 주파수 스윕."""
    _ensure_pwm_started()
    step_time = max(1, duration_ms // steps) / 1000.0
    if steps <= 1:
        _pwm.ChangeFrequency(start_hz)
        _pwm.ChangeDutyCycle(DUTY)
        time.sleep(duration_ms / 1000.0)
        _pwm.ChangeDutyCycle(0)
        return
    df = (end_hz - start_hz) / float(steps - 1)
    _pwm.ChangeDutyCycle(DUTY)
    f = start_hz
    for _ in range(steps):
        _pwm.ChangeFrequency(int(f))
        time.sleep(step_time)
        f += df
    _pwm.ChangeDutyCycle(0)

def say_are_you_ok():
    """
    낙상 '처음' 감지 시 호출.
    - 기본: 짧은 비프 3회
    - USE_SWEEP=1 이면 톤 스윕 1회
    """
    print(f"detected → buzzer on (pin={BUZZER_PIN}, mode={GPIO_MODE})")
    try:
        if USE_SWEEP:
            # 예: 600→1600Hz, 총 600ms 스윕
            _beep_sweep(start_hz=600, end_hz=1600, duration_ms=600, steps=24)
        else:
            for _ in range(BEEP_COUNT):
                _beep_once(BUZZER_FREQ, BEEP_ON_MS, DUTY)
                time.sleep(BEEP_OFF_MS / 1000.0)
    except Exception as e:
        print("[buzzer] 오류:", e)
    return "ALERT"

def cleanup():
    """프로그램 종료 시 GPIO 정리."""
    try:
        if _pwm_started:
            _pwm.stop()
    finally:
        GPIO.cleanup()
