#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microphone test (USB headset preferred).
- Records 5 seconds, plays back, and saves as recorded.wav.
- Prints simple RMS level (dBFS).
"""

import numpy as np
import sounddevice as sd
import wave

SAMPLE_RATE = 8000
DURATION = 5.0
WAV_PATH = "recorded.wav"

def find_input_device():
    """Pick an input device that looks like a USB headset if available."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        name = d["name"].lower()
        if d["max_input_channels"] > 0 and any(k in name for k in ["usb", "headset", "audio"]):
            return i
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            return i
    return None

def record(seconds=DURATION):
    print(f"Recording {seconds:.1f}s ...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Done.")
    return audio.squeeze(-1)

def rms_dbfs(x: np.ndarray) -> float:
    """Return RMS level in dBFS for float32 [-1,1] audio."""
    eps = 1e-12
    rms = np.sqrt(np.mean(np.square(x)) + eps)
    return 20.0 * np.log10(rms + eps)

def save_wav(path, audio_float32):
    """Save float32 [-1,1] audio to 16-bit PCM WAV."""
    audio_int16 = np.clip(audio_float32 * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    print(f"Saved: {path}")

def main():
    in_idx = find_input_device()
    if in_idx is not None:
        sd.default.device = (in_idx, None)
        print(f"[Input] Using device {in_idx}: {sd.query_devices(in_idx)['name']}")
    else:
        print("[Input] Using system default.")

    audio = record(DURATION)
    level = rms_dbfs(audio)
    print(f"RMS level: {level:.1f} dBFS (speak into the mic and check it changes)")

    print("Playing back...")
    sd.play(audio, SAMPLE_RATE, blocking=True)

    save_wav(WAV_PATH, audio)
    print("Done.")

if __name__ == "__main__":
    main()
