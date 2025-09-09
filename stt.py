# stt.py — capture via ALSA arecord -D mic → Google streaming
from google.cloud import speech
import subprocess, threading, queue, sys, time, os, signal

# ====== Audio I/O config ======
SAMPLE_RATE = 48000        # ~/.asoundrc의 mic 설정에 맞춤(48k)
CHANNELS    = 1
BYTES_PER_SAMPLE = 2       # S16_LE
CHUNK_MS    = 100
CHUNK_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * (CHUNK_MS/1000.0))

# ====== Google STT client ======
stt_client = speech.SpeechClient()

def _request_generator(q: "queue.Queue[bytes]"):
    while True:
        chunk = q.get()
        if chunk is None:
            return
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

def speech_to_text() -> str:
    """
    Capture mic via ALSA (arecord -D mic), stream to Google STT.
    Returns first final transcript or "" on timeout/no-speech.
    """
    print("[STT] Listening... (arecord -D mic)")
    # ~/.asoundrc 에서 만든 'mic' PCM을 직접 사용
    cmd = [
        "arecord", "-q",
        "-D", "mic",
        "-r", str(SAMPLE_RATE),
        "-f", "S16_LE",
        "-c", str(CHANNELS)
    ]
    # 파이프 열기
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)
    stop_event = threading.Event()
    final_text = ""

    def reader():
        """arecord stdout에서 고정 크기 청크로 읽어 큐에 넣음."""
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
        interim_results=True,     # 중간 결과도 확인
        single_utterance=True,    # 발화 + 잠깐 침묵 → final
    )

    try:
        requests = _request_generator(q)
        responses = stt_client.streaming_recognize(stream_config, requests)
        start = time.time()
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
            if time.time() - start > 10.0:  # 10초 안에 final 없으면 종료
                print("[STT] streaming timeout")
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
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass

    return final_text
