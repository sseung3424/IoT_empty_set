#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microphone test for USB audio device (AB13X USB Audio).
- Records 5 seconds, plays back, and saves as recorded.wav
- Uses hardware parameters: 48 kHz, mono, 16-bit PCM
- Computes simple RMS level in dBFS
"""

import numpy as np
import sounddevice as sd
import wave

SAMPLE_RATE = 48000      # hardware-supported sample rate
DURATION = 5.0           # recording duration in seconds
WAV_PATH = "recorded.wav"

# Device setup: input is hw:3,0 (USB audio), output can be system default or USB
INPUT_HW = "hw:3,0"
OUTPUT_DEV = None        # set to "hw:3,0" or "plughw:3,0" if you want playback on the USB headset

# Configure global defaults for sounddevice
sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 1
sd.default.dtype = "int16"
sd.default.device = (INPUT_HW, OUTPUT_DEV)

def record(seconds=DURATION):
    """Record from the USB microphone and return audio as float32 in range [-1, 1]."""
    print(f"Recording {seconds:.1f}s @ {SAMPLE_RATE} Hz, int16, mono ...")
    audio = sd.rec(int(seconds * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype="int16",
                   device=INPUT_HW)
    sd.wait()
    print("Done.")
    # Convert int16 to float32 for analysis/playback
    return (audio.astype(np.float32) / 32768.0).squeeze(-1)

def rms_dbfs(x: np.ndarray) -> float:
    """Compute RMS level in dBFS from float32 audio."""
    eps = 1e-12
    rms = np.sqrt(np.mean(x * x) + eps)
    return 20.0 * np.log10(rms + eps)

def save_wav(path, audio_float32):
    """Save float32 audio [-1,1] as 16-bit PCM WAV file."""
    audio_int16 = np.clip(audio_float32 * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    print(f"Saved: {path}")

def main():
    # Record
    audio = record(DURATION)
    level = rms_dbfs(audio)
    print(f"RMS level: {level:.1f} dBFS (speak into the mic to see changes)")

    # Playback
    print("Playing back...")
    sd.play(audio, SAMPLE_RATE, blocking=True, device=OUTPUT_DEV)

    # Save file
    save_wav(WAV_PATH, audio)
    print("Finished.")

if __name__ == "__main__":
    main()
