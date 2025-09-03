# stt.py
from google.cloud import speech
import pyaudio
import queue
import threading

stt_client = speech.SpeechClient()

RATE = 16000
CHUNK = int(RATE / 10)
MIC_INDEX = 2  # None - basic mic, configure index for selecting USB mic

def stream_generator(q):
    while True:
        chunk = q.get()
        if chunk is None:
            return
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

def speech_to_text():
    audio_interface = pyaudio.PyAudio()
    audio_stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=MIC_INDEX,  # select USB mic
        frames_per_buffer=CHUNK,
    )

    q = queue.Queue()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )

    def fill_buffer():
        while True:
            try:
                data = audio_stream.read(CHUNK, exception_on_overflow=False)
                q.put(data)
            except Exception as e:
                print("[ERROR] Mic read:", e)
                break

    threading.Thread(target=fill_buffer, daemon=True).start()

    requests = stream_generator(q)
    responses = stt_client.streaming_recognize(streaming_config, requests)

    print("Listening...")

    final_text = ""
    try:
        for response in responses:
            for result in response.results:
                if result.is_final:
                    final_text = result.alternatives[0].transcript
                    q.put(None)
                    audio_stream.stop_stream()
                    audio_stream.close()
                    audio_interface.terminate()
                    print("Recognized:", final_text)
                    return final_text
    except Exception as e:
        print("[ERROR] STT:", e)
        q.put(None)
        audio_stream.stop_stream()
        audio_stream.close()
        audio_interface.terminate()
        return ""
