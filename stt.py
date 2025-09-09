# stt.py
from google.cloud import speech
import sounddevice as sd
import queue
import threading
import sys

# ====== Audio I/O config ======
SAMPLE_RATE = 48000          # USB 헤드셋이 48 kHz 지원 일반적
CHANNELS = 1                 # mono
DTYPE = "int16"              # Google STT는 LINEAR16 기대
BLOCK_MS = 100               # ~100 ms per chunk
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_MS / 1000)

# 우선순위:
# 1) DEVICE_NAME에 "USB", "Headset" 같이 부분 문자열로 매칭 시도
# 2) 못 찾으면 INPUT_DEVICE_INDEX 사용
# 3) 둘 다 없으면 PortAudio 기본 입력 사용(None)
DEVICE_NAME = "USB"          # 장치 이름 일부(예: "USB", "Headset", "C-Media", 필요시 수정)
INPUT_DEVICE_INDEX = None    # 정확한 인덱스 알고 있으면 정수로 지정

# ====== Google STT client ======
stt_client = speech.SpeechClient()


def _resolve_input_device():
    """
    Return a device index or None.
    Prefer substring match on DEVICE_NAME; fallback to INPUT_DEVICE_INDEX; else None.
    """
    try:
        devices = sd.query_devices()
    except Exception as e:
        print("[STT] sd.query_devices() failed:", e, file=sys.stderr)
        return INPUT_DEVICE_INDEX

    # Try name substring match among input-capable devices
    if DEVICE_NAME:
        name_l = DEVICE_NAME.lower()
        for i, info in enumerate(devices):
            if info.get("max_input_channels", 0) > 0:
                if name_l in str(info.get("name", "")).lower():
                    return i

    return INPUT_DEVICE_INDEX  # could be None


def _request_generator(q: "queue.Queue[bytes]"):
    """Yield StreamingRecognizeRequest objects from an audio-bytes queue."""
    while True:
        chunk = q.get()
        if chunk is None:
            return
        yield speech.StreamingRecognizeRequest(audio_content=chunk)


def speech_to_text() -> str:
    """
    Capture microphone via sounddevice and stream to Google Cloud STT.
    Stops on the first final result and returns recognized text.
    Returns "" on timeout/error/no-speech.
    """
    audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=20)
    stop_event = threading.Event()

    # Configure sounddevice defaults (optional)
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS
    sd.default.dtype = DTYPE

    # Resolve input device once per call (robust to hotplug)
    in_dev = _resolve_input_device()
    if in_dev is not None:
        try:
            dev_info = sd.query_devices(in_dev)
            if dev_info.get("max_input_channels", 0) <= 0:
                print(f"[STT] Device index {in_dev} has no input channels; using default.", file=sys.stderr)
                in_dev = None
        except Exception:
            in_dev = None

    def audio_callback(indata, frames, time_, status):
        if status:
            print("[AudioStatus]", status)
        try:
            audio_q.put_nowait(indata.tobytes())
        except queue.Full:
            pass  # drop oldest if recognizer is slow

    recog_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    stream_config = speech.StreamingRecognitionConfig(
        config=recog_config,
        interim_results=False,
        single_utterance=True,   # 한 문장 후 스트림 종료
    )

    print("Listening...")
    final_text = ""

    try:
        with sd.InputStream(device=in_dev,
                            samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype=DTYPE,
                            blocksize=BLOCK_FRAMES,
                            callback=audio_callback):
            requests = _request_generator(audio_q)
            responses = stt_client.streaming_recognize(stream_config, requests)

            for response in responses:
                for result in response.results:
                    if result.is_final:
                        final_text = result.alternatives[0].transcript
                        print("Recognized:", final_text)
                        stop_event.set()
                        break
                if stop_event.is_set():
                    break
    except Exception as e:
        print("[ERROR] STT open/stream:", e, file=sys.stderr)
    finally:
        audio_q.put(None)  # stop generator

    return final_text
