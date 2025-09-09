# tts.py
from google.cloud import texttospeech
import wave
import numpy as np
import subprocess
import os
import tempfile

# ====== Audio I/O config ======
SAMPLE_RATE = 48000          # USB 헤드셋에 맞춰 48 kHz 요청
CHANNELS = 1                 # Google TTS는 mono로 받아도 충분
APLAY_DEVICE = "plughw:3,0"  # 또는 ~/.asoundrc를 썼다면 "speaker" 로 바꿔도 됨 (권장: "speaker")

tts_client = texttospeech.TextToSpeechClient()


def _write_wav(path: str, pcm16_mono: bytes, sample_rate: int):
    """Write mono PCM16 little-endian bytes to a WAV file."""
    # Ensure little-endian int16
    data = np.frombuffer(pcm16_mono, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def text_to_speech(text: str) -> None:
    """Synthesize Korean speech and play via ALSA aplay (avoids PortAudio locks)."""
    if not text:
        return

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Standard-A",
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,   # 48 kHz 요청
    )

    # Call Google TTS
    resp = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    # Save to a temp wav and play with aplay (non-blocking device ownership after playback)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        _write_wav(tmp_path, resp.audio_content, SAMPLE_RATE)
        # -D로 ALSA PCM 지정; ~/.asoundrc 설정 시 "speaker"가 가장 편함
        subprocess.run(["aplay", "-q", "-D", APLAY_DEVICE, tmp_path], check=False)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
