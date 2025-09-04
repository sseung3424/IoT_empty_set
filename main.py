#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speaker test (USB headset preferred).
- Plays 440 Hz tone (1s) then L/R channel check (1.5s).
"""

import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 16000  # 16 kHz is enough for tests

def find_output_device():
    """Pick an output device that looks like a USB headset if available."""
    devices = sd.query_devices()
    # Prefer devices that look like USB/Headset/Audio
    for i, d in enumerate(devices):
        name = d["name"].lower()
        if d["max_output_channels"] > 0 and any(k in name for k in ["usb", "headset", "audio"]):
            return i
    # Fallback: first device with output channels
    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            return i
    return None

def play_tone(frequency=440.0, seconds=1.0):
    """Play a mono sine tone to both channels."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    sd.play(stereo, SAMPLE_RATE, blocking=True)

def lr_check(frequency=660.0):
    """Play left-only, right-only, then both (0.5s each)."""
    seg = int(0.5 * SAMPLE_RATE)
    t = np.linspace(0, 0.5, seg, endpoint=False)
    wave = 0.2 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    left = np.zeros((seg, 2), dtype=np.float32);  left[:, 0] = wave
    right = np.zeros((seg, 2), dtype=np.float32); right[:, 1] = wave
    both = np.stack([wave, wave], axis=1)

    sd.play(np.concatenate([left, right, both], axis=0), SAMPLE_RATE, blocking=True)

def main():
    out_idx = find_output_device()
    if out_idx is not None:
        sd.default.device = (None, out_idx)
        print(f"[Output] Using device {out_idx}: {sd.query_devices(out_idx)['name']}")
    else:
        print("[Output] Using system default.")

    print("Playing 440 Hz...")
    play_tone(440.0, 1.0)
    time.sleep(0.2)

    print("L/R check (left → right → both)...")
    lr_check(660.0)

    print("Done.")

if __name__ == "__main__":
    main()
