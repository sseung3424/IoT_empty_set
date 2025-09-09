# stt.py — ALSA arecord -D mic (RAW) → Google streaming, with keepalive silence
from google.cloud import speech
import subprocess, threading, queue, sys, time, os, signal

# ====== Audio I/O config ======
SAMPLE_RATE = 48000
CHANNELS    = 1
BYTES_PER_SAMPLE = 2            # S16_LE
CHUNK_MS    = 20                # 더 짧게: 20ms
CHUNK_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * (CHUNK_MS/1000.0))
SILENCE_CHUNK = b"\x00" * CHUNK_BYTES

# ====== Google STT client ======
stt_client = speech.SpeechClient()

def _request_generator(q: "queue.Queue[bytes]", stop_event: threading.Event):
    """
    Generator that yields audio chunks in (near) real-time.
    - 즉시 무음 한 청크를 보내 API 'no audio' 타임아웃을 피함
    - 큐가 잠깐 비면 무음 keepalive를 보냄(실시간 유지)
    """
    # 1) 빠른 초기 전송(무음 1청크)
    yield speech.StreamingRecognizeRequest(audio_content=SILENCE_CHUNK)

    last_send = time.time()
    idle_keepalive_sec = 0.2  # 200ms 이상 비면 무음 전송
    while not stop_event.is_set():
        try:
            chunk = q.get(timeout=idle_keepalive_sec)
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)
            last_send = time.time()
        except queue.Empty:
            # 큐가 비면 무음 keepalive
            yield speech.StreamingRecognizeRequest(audio_content=SILENCE_CHUNK)
            last_send = time.time()

def speech_to_text() -> str:
    """
    Capture mic via ALSA (arecord -D mic -t raw), stream to Google STT.
    Returns first final transcript or "" on timeout/no-speech.
    """
    print("[STT] Listening... (arecord -D mic, RAW)")
    # ~/.asoundrc 의 'mic' PCM을 RAW로 캡처
    cmd = [
        "arecord", "-q",
        "-D", "mic",
        "-r", str(SAMPLE_RATE),
        "-f", "S16_LE",
        "-c", str(CHANNELS),
        "-t", "raw"                 # ★ WAV 헤더 제거
    ]

    # 파이프 오픈 (bufsize=0로 파이썬 버퍼링 최소화)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    q: "queue.Queue[bytes]" = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    final_text = ""

    def reader():
        """arecord stdout에서 CHUNK_BYTES 단위로 읽어 큐에 넣음."""
        try:
            buf = b""
            while not stop_event.is_set():
                need = CHUNK_BYTES - len(buf)
                data = proc.stdout.read(need)
                if not data:
                    break
                buf += data
                if len(buf) >= CHUNK_BYTES:
                    q.put(buf[:CHUNK_BYTES])
                    buf = buf[CHUNK_BYTES:]
        except Exception as e:
            print("[STT] reader err:", e, file=sys.stderr)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    recog_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    stream_config = speech.StreamingRecognitionConfig(
        config=recog_config,
        interim_results=True,     # 중간 결과 로깅
        single_utterance=True,    # 발화 + 잠깐 침묵 후 final
    )

    try:
        requests  = _request_generator(q, stop_event)
        responses = stt_client.streaming_recognize(stream_config, requests)
        start = time.time()
        max_secs = 12.0

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
            if time.time() - start > max_secs:
                print("[STT] streaming timeout (no final)")
                break
    except Exception as e:
        print("[ERROR] STT streaming:", e, file=sys.stderr)
    finally:
        q.put(None)
        stop_event.set()
        # arecord 종료
        try:
            proc.terminate()
            try:
                proc.wait(timeout=0.4)
            except subprocess.TimeoutExpired:
                os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass

    return final_text