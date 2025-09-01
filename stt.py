from google.cloud import speech
import pyaudio
import queue

stt_client = speech.SpeechClient()

RATE = 16000
CHUNK = int(RATE / 10)

def stream_generator(q):
    while True:
        chunk = q.get()
        if chunk is None:
            return
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

def listen_and_recognize():
    audio_interface = pyaudio.PyAudio()
    audio_stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
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
            data = audio_stream.read(CHUNK)
            q.put(data)

    import threading
    threading.Thread(target=fill_buffer, daemon=True).start()

    requests = stream_generator(q)
    responses = stt_client.streaming_recognize(streaming_config, requests)

    print("say, I'm hearing")
    try:
        for response in responses:
            for result in response.results:
                if result.is_final:
                    print("final recognition:", result.alternatives[0].transcript)
    except KeyboardInterrupt:
        print("\naudio recogniction stops")
        q.put(None)
        audio_stream.stop_stream()
        audio_stream.close()
        audio_interface.terminate()

if __name__ == "__main__":
    listen_and_recognize()
