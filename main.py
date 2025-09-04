#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buzzer test on Raspberry Pi (GPIO.BOARD mode).
- Pin: 32 (physical pin number)
- Function: sweep duty cycle up and down
- Stop with Ctrl+C
"""

import time
import RPi.GPIO as GPIO

# GPIO setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
BUZZER_PIN = 32
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Create PWM instance on BUZZER_PIN at 440Hz
p = GPIO.PWM(BUZZER_PIN, 440)
p.start(50)  # initial duty cycle 50%

print("Buzzer test start. Press Ctrl+C to stop.")

try:
    while True:
        # duty cycle up
        for dc in range(0, 101, 5):
            p.ChangeDutyCycle(dc)
            time.sleep(0.05)
        # duty cycle down
        for dc in range(100, -1, -5):
            p.ChangeDutyCycle(dc)
            time.sleep(0.05)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected.")

finally:
    p.stop()
    GPIO.cleanup()
    print("Buzzer test ended. GPIO cleaned up.")
