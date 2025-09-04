# main.py
# -*- coding: utf-8 -*-
"""
Simple drive test for YB_Pcb_Car motor controller over I2C.
- Keys: w/a/s/d = move, x = stop, q = quit
- +/- : increase/decrease speed
- j/k : pan servo (id=0) left/right
- i/m : tilt servo (id=1) up/down
"""

import time
import sys
import termios
import tty

from YB_Pcb_Car import YB_Pcb_Car


# ---------- tiny terminal helper (Linux/macOS) ----------
def getch():
    """Read a single character from stdin without Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# ---------- clamp helper ----------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    car = YB_Pcb_Car()

    # Default speeds (0~255 is typical for many Yahboom boards)
    left_speed = 120
    right_speed = 120

    # Servo state (0~180)
    pan = 90   # id=0
    tilt = 90  # id=1

    print("\nYB_Pcb_Car drive tester")
    print("------------------------------------------------")
    print("w: forward   s: back     a: left     d: right")
    print("x: stop      q: quit")
    print("+: faster    -: slower   (both wheels)")
    print("j/k: pan(0)  i/m: tilt(1)  (servo 0 and 1)")
    print("------------------------------------------------")
    print(f"Start speed L/R = {left_speed}/{right_speed}, pan={pan}, tilt={tilt}\n")

    # Initialize servos to neutral
    try:
        car.Ctrl_Servo(0, pan)
        car.Ctrl_Servo(1, tilt)
    except Exception as e:
        print("Servo init failed (continue anyway):", e)

    try:
        while True:
            c = getch()

            if c in ("q", "Q"):
                # quit
                car.Car_Stop()
                print("Quit.")
                break

            elif c in ("x", "X"):
                # stop
                car.Car_Stop()
                print("STOP")

            elif c in ("w", "W"):
                # forward
                car.Car_Run(left_speed, right_speed)
                print(f"FORWARD: L={left_speed} R={right_speed}")

            elif c in ("s", "S"):
                # back
                car.Car_Back(left_speed, right_speed)
                print(f"BACK:    L={left_speed} R={right_speed}")

            elif c in ("a", "A"):
                # turn left (left slower, right faster) – or use Car_Left API
                # Using provided API for clarity:
                car.Car_Left(left_speed, right_speed)
                print(f"LEFT:    L={left_speed} R={right_speed}")

            elif c in ("d", "D"):
                # turn right
                car.Car_Right(left_speed, right_speed)
                print(f"RIGHT:   L={left_speed} R={right_speed}")

            elif c == "+":
                left_speed = clamp(left_speed + 10, 0, 255)
                right_speed = clamp(right_speed + 10, 0, 255)
                print(f"SPEED UP -> L={left_speed} R={right_speed}")

            elif c == "-":
                left_speed = clamp(left_speed - 10, 0, 255)
                right_speed = clamp(right_speed - 10, 0, 255)
                print(f"SPEED DOWN -> L={left_speed} R={right_speed}")

            # --- servo controls ---
            elif c in ("j", "J"):  # pan left
                pan = clamp(pan - 5, 0, 180)
                car.Ctrl_Servo(0, pan)
                print(f"PAN(0) -> {pan}")

            elif c in ("k", "K"):  # pan right
                pan = clamp(pan + 5, 0, 180)
                car.Ctrl_Servo(0, pan)
                print(f"PAN(0) -> {pan}")

            elif c in ("i", "I"):  # tilt up
                tilt = clamp(tilt + 5, 0, 180)
                car.Ctrl_Servo(1, tilt)
                print(f"TILT(1) -> {tilt}")

            elif c in ("m", "M"):  # tilt down
                tilt = clamp(tilt - 5, 0, 180)
                car.Ctrl_Servo(1, tilt)
                print(f"TILT(1) -> {tilt}")

            # ignore other keys silently

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    finally:
        try:
            car.Car_Stop()
        except:
            pass
        print("Motors stopped. Bye.")


if __name__ == "__main__":
    # Small countdown so you can put the robot on the ground safely
    for t in range(3, 0, -1):
        print(f"Starting in {t}...")
        time.sleep(1)
    main()
