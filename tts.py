# tts.py
from google.cloud import texttospeech
import numpy as np
import sounddevice as sd

# ====== Audio I/O config (match your working output test) ======
SAMPLE_RATE = 48000              # 48 kHz to match USB device
CHANNELS_OUT = 2                 # play as stereo (duplicate mono to L/R)
DTYPE = "int16"
OUTPUT_DEV = "hw:3,0"            # set to your USB output device (e.g., "plughw:3,0")

# Pre-configure sounddevice defaults
sd.default.samplerate = SAMPLE_RATE
sd.default.channels = CHANNELS_OUT
sd.default.dtype = DTYPE
sd.default.device = (None, OUTPUT_DEV)

# ====== Google TTS client ======
tts_client = texttospeech.TextToSpeechClient()

def _mono_pcm16_to_stereo_int16(raw_bytes: bytes) -> np.ndarray:
    """Convert mono PCM16 little-endian bytes to stereo int16 numpy array."""
    mono = np.frombuffer(raw_bytes, dtype=np.int16)
    # shape -> (N, 1) then duplicate to (N, 2)
    stereo = np.stack([mono, mono], axis=1)
    return stereo

def text_to_speech(text: str) -> None:
    """Synthesize Korean speech with Google TTS and play via sounddevice."""
    if not text:
        return

    # Build synthesis request: LINEAR16 @ 48 kHz, mono
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Standard-A",
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,   # request 48 kHz output
    )

    # Call Google TTS
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Convert LINEAR16 mono -> stereo int16
    stereo_i16 = _mono_pcm16_to_stereo_int16(response.audio_content)

    # Play blocking on the selected ALSA device
    sd.play(stereo_i16, SAMPLE_RATE, blocking=True, device=OUTPUT_DEV)
