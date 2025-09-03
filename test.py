# list_audio.py
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    print(i, info["name"], "| host:", info["hostApi"], "| rate:", int(info["defaultSampleRate"]))
pa.terminate()
