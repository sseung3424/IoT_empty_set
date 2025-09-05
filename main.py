#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speaker test for USB audio device (AB13X USB Audio).
- Plays a 440 Hz tone (1 s) then L/R channel check (0.5 s each, then both).
- Uses hardware parameters: 48 kHz, stereo, 16-bit PCM.
"""

import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 48000        # hardware-supported sample rate
OUTPUT_DEV  = "hw:3,0"     # set to "plughw:3,0" if you prefer ALSA conversion

# Configure sounddevice defaults to match the USB headset
sd.default.samplerate = SAMPLE_RATE
sd.default.channels   = 2            # stereo
sd.default.dtype      = "int16"
sd.default.device     = (None, OUTPUT_DEV)

def _to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 PCM."""
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)

def play_tone(frequency: float = 440.0, seconds: float = 1.0, gain: float = 0.2):
    """Play a stereo sine tone (same signal on L/R)."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    tone = gain * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    sd.play(_to_int16(stereo), SAMPLE_RATE, blocking=True, device=OUTPUT_DEV)

def lr_check(frequency: float = 660.0, seg_seconds: float = 0.5, gain: float = 0.2):
    """Play left-only, right-only, then both (0.5 s each by default)."""
    n = int(seg_seconds * SAMPLE_RATE)
    t = np.linspace(0, seg_seconds, n, endpoint=False)
    wave = (gain * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    left  = np.stack([wave, np.zeros_like(wave)], axis=1)
    right = np.stack([np.zeros_like(wave), wave], axis=1)
    both  = np.stack([wave, wave], axis=1)

    seq = np.concatenate([left, right, both], axis=0)
    sd.play(_to_int16(seq), SAMPLE_RATE, blocking=True, device=OUTPUT_DEV)

def main():
    print(f"[Output] Using device: {OUTPUT_DEV}")
    print("Playing 440 Hz tone...")
    play_tone(440.0, 1.0)
    time.sleep(0.2)

    print("L/R check (left → right → both)...")
    lr_check(660.0, 0.5)

    print("Done.")

if __name__ == "__main__":
    main()
