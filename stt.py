# stt.py  — streaming + fallback(synchronous) with strong debug
from google.cloud import speech
import sounddevice as sd
import numpy as np
import queue, threading, time, sys

# ====== Audio I/O config ======
SAMPLE_RATE = 48000      # ~/.asoundrc에서 mic를 48k로 설정한 상태 권장
CHANNELS    = 1
DTYPE       = "int16"
BLOCK_MS    = 100
BLOCK_FRAMES= int(SAMPLE_RATE * BLOCK_MS / 1000)

# ★ 여기만 환경에 맞게 조정
INPUT_DEVICE = "hw:3,0"   # 필요시 "plughw:3,0" 도 시도 가능

# ====== Google STT client ======
stt_client = speech.SpeechClient()

def _request_generator(q: "queue.Queue[bytes]"):
    while True:
        chunk = q.get()
        if chunk is None:
            return
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

def _recognize_sync(pcm16_mono: bytes, sample_rate: int) -> str:
    """Fallback: buffered 3s audio → synchronous recognize."""
    print("[STT][fallback] synchronous recognize...")
    audio = speech.RecognitionAudio(content=pcm16_mono)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    resp = stt_client.recognize(config=config, audio=audio)
    for res in resp.results:
        if res.alternatives:
            txt = res.alternatives[0].transcript
            print("Recognized:", txt)
            return txt
    print("[STT][fallback] no text")
    return ""

def speech_to_text() -> str:
    """
    Try streaming STT first.
    If no final result within STREAM_MAX_SEC, fallback to 3s buffered sync STT.
    """
    print("[STT] enter")
    audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=30)
    stop_event = threading.Event()
    final_text = ""

    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels   = CHANNELS
    sd.default.dtype      = DTYPE

    last_rms_print = 0.0
    level_seen = False

    def audio_callback(indata, frames, t, status):
        nonlocal last_rms_print, level_seen
        if status:
            print("[STT][AudioStatus]", status, file=sys.stderr)
        # 0.5s마다 RMS 출력 (입력 레벨 확인)
        now = time.time()
        if now - last_rms_print > 0.5:
            rms = float(np.sqrt(np.mean(indata.astype(np.int16).astype(np.float32)**2)))
            print(f"[STT][Mic RMS] {rms:.1f}")
            if rms > 50:  # 대충 소음/발화가 있으면 true
                level_seen = True
            last_rms_print = now
        try:
            audio_q.put_nowait(indata.tobytes())
        except queue.Full:
            pass

    # ========== 1) Streaming 먼저 시도 ==========
    print(f"[STT] Listening... (streaming, device={INPUT_DEVICE})")
    recog_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    stream_config = speech.StreamingRecognitionConfig(
        config=recog_config,
        interim_results=True,     # 중간 결과도 로깅
        single_utterance=True,    # 말하고 약간 침묵 → final
    )

    STREAM_MAX_SEC = 8.0         # 8초 안에 final 없으면 폴백
    stream_start = time.time()

    try:
        with sd.InputStream(
            device=INPUT_DEVICE,                  # ★★★ 강제 지정
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_FRAMES,
            callback=audio_callback
        ):
            print("[STT] mic stream OPENED")
            requests = _request_generator(audio_q)
            responses = stt_client.streaming_recognize(stream_config, requests)

            for response in responses:
                for result in response.results:
                    if result.is_final:
                        final_text = result.alternatives[0].transcript
                        print("Recognized:", final_text)
                        stop_event.set()
                        break
                    else:
                        if result.alternatives:
                            print("[STT][interim]:", result.alternatives[0].transcript)
                if stop_event.is_set():
                    break
                if (time.time() - stream_start) > STREAM_MAX_SEC:
                    print("[STT] streaming timeout → fallback")
                    break
    except Exception as e:
        print("[ERROR] STT streaming:", e, file=sys.stderr)
    finally:
        audio_q.put(None)

    if final_text:
        return final_text

    # ========== 2) Fallback: 3초 버퍼 녹음 후 동기식 인식 ==========
    print("[STT] fallback record 3s...")
    rec = sd.rec(int(SAMPLE_RATE * 3),
                 samplerate=SAMPLE_RATE,
                 channels=CHANNELS,
                 dtype=DTYPE,
                 device=INPUT_DEVICE)              # ★★★ 폴백도 같은 장치로
    sd.wait()
    pcm = rec.tobytes()
    return _recognize_sync(pcm, SAMPLE_RATE)
