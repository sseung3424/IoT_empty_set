# stt.py
from google.cloud import speech
import sounddevice as sd
import queue
import threading

# ====== Audio I/O config (match your working test) ======
SAMPLE_RATE = 48000          # USB device supports 48 kHz
CHANNELS = 1                 # mono
DTYPE = "int16"              # LINEAR16 expected by Google STT
BLOCK_MS = 100               # ~100 ms per chunk
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_MS / 1000)

# Choose your input device: ALSA name like "hw:3,0" (as in your test) or an index/int.
INPUT_DEVICE = "hw:3,0"      # set to your USB mic device (or an integer index)

# ====== Google STT client ======
stt_client = speech.SpeechClient()

def _request_generator(q: "queue.Queue[bytes]"):
    """Yield StreamingRecognizeRequest objects from an audio-bytes queue."""
    while True:
        chunk = q.get()
        if chunk is None:  # sentinel -> stop
            return
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

def speech_to_text() -> str:
    """
    Capture microphone via sounddevice and stream to Google Cloud STT.
    Stops on the first final result and returns recognized text.
    """
    audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=20)
    stop_event = threading.Event()

    # Configure sounddevice defaults (optional)
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS
    sd.default.dtype = DTYPE

    # Audio callback pushes raw int16 bytes into the queue
    def audio_callback(indata, frames, time, status):
        # status handling is intentionally minimal; print but keep streaming
        if status:
            print("[AudioStatus]", status)
        try:
            audio_q.put_nowait(indata.tobytes())
        except queue.Full:
            # drop if the recognizer is slower than input (prevents unbounded growth)
            pass

    # Build Google STT configs
    recog_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    stream_config = speech.StreamingRecognitionConfig(
        config=recog_config,
        interim_results=False,   # return only final results
        single_utterance=True,   # end after a spoken phrase
    )

    print("Listening...")
    final_text = ""

    # Open input stream with fixed blocksize for stable chunking
    with sd.InputStream(device=INPUT_DEVICE,
                        samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype=DTYPE,
                        blocksize=BLOCK_FRAMES,
                        callback=audio_callback):
        # Start streaming to Google
        requests = _request_generator(audio_q)
        responses = stt_client.streaming_recognize(stream_config, requests)

        try:
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
            print("[ERROR] STT:", e)
        finally:
            # stop generator/stream by sending sentinel
            audio_q.put(None)

    return final_text
