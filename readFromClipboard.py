import os
import torch
import pyperclip
from pyperclip import PyperclipWindowsException
import time
import wave
import pyaudio
import re
from TTS.api import TTS
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue
from queue import Empty
from io import BytesIO
import numpy as np
import scipy.io.wavfile
import keyboard #pip install keyboard
from functools import partial


logging.basicConfig(level=logging.INFO)

audioQueue = Queue()
SAMPLE_RATE = 20000

class TaskManager:
    def __init__(self):
        self.executor = None
        self.stop_event = threading.Event()
        self.tts = None
        self.current_text = "" 

    def start_tasks(self):
        self.stop_event.clear()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.executor.submit(audio_generator, self.tts, self.current_text, self.stop_event)
        self.executor.submit(audio_player, self.stop_event)
        self.executor.submit(listen_for_ctrl_x, self)
        logging.debug("Started new tasks")

    def stop_tasks(self):
        if self.executor:
            self.stop_event.set()
            self.executor.shutdown(wait=True)
            logging.debug("Stopped previous tasks")
    
    def setTTS(self, tts):
        self.tts = tts
    def setText(self, text):
        self.current_text = text

AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)
SLEEP_TIME_MS = 100/1000.0 

def extract_field_names(obj):
    if hasattr(obj, '__dict__'):
        return {key: "unserializable" for key in obj.__dict__.keys()}
    else:
        return "Object has no __dict__"

def initialize_tts():
    """Initialize CUDA device and TTS model."""
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if torch.cuda.is_available():
    #     logging.debug("CUDA installed successfully\n")
    # else:
    #     logging.debug("CUDA not properly installed. Stopping process...")
    #     quit()

    device = torch.device("cpu")

    # view_models = input("View models? [y/n]\n")
    # if view_models.lower() == "y":
    #     tts_manager = TTS().list_models()
    #     all_models = tts_manager.list_models()
    #     logging.debug("TTS models:\n", all_models, "\n")

    # model = input("Enter model name:\n")  # e.g., tts_models/multilingual/multi-dataset/xtts_v2
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
    # tts = TTS("tts_models/en/jenny/jenny").to(device)
    tts = TTS("tts_models/en/ljspeech/vits").to(device)
    return tts

def audio_generator(tts, text, stop_event):
    global SAMPLE_RATE
    text = text.replace('\n\r', ' ').replace('\n', ' ')
    lines = split_text_into_lines(text)
    for line in lines:
        if stop_event.is_set():
            break
        else:
            wav, SAMPLE_RATE = tts.tts(text=line, split_sentences=False)
            audioQueue.put(wav) #speaker_wav="male.wav", language="en", 
        time.sleep(SLEEP_TIME_MS)

def split_text_into_lines(text):
    # Replace newlines with spaces
    text = text.replace('\n', ' ')

    # Use regular expression to split after '.' or '?' or ';'.
    # The r'\.' and r'\?' and r'\;' match the literal characters.
    # The '+' allows for one or more whitespace characters after the delimiter.
    parts = re.split(r'(?<=[.?;])\s+', text, flags=re.IGNORECASE)
    
    # Remove empty strings from the list (caused by consecutive delimiters or delimiters at the start/end)
    parts = [part for part in parts if part.strip()]

    return parts

def play_output_audio(wav_data, stop_event):
    global SAMPLE_RATE
    # Open the WAV file
    try:
        # Step 1: Create a BytesIO buffer and write the WAV data into it
        buffer = BytesIO()
        sample_rate = SAMPLE_RATE  # or whatever your sample rate is

        # Normalize and convert to int16
        wav_norm = wav_data * (32767 / max(0.01, np.max(np.abs(wav_data))))
        wav_norm = wav_norm.astype(np.int16)

        # Write to buffer
        scipy.io.wavfile.write(buffer, sample_rate, wav_norm)
        buffer.seek(0)

        # Step 2: Open the buffer as a WAV file
        wf = wave.open(buffer, 'rb')

        # Step 3: Play using PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Step 4: Stream the audio
        chunk = 1024
        data = wf.readframes(chunk)
        while data and not stop_event.is_set():
            stream.write(data)
            data = wf.readframes(chunk)

        # Step 5: Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
    except KeyboardInterrupt:
        logging.info("\n👋 Listener stopped by user.")

def audio_player(stop_event):
    while not stop_event.is_set():
        if not audioQueue.empty():
            wav_bytes = audioQueue.get()
            for item in wav_bytes:
                if not (type(item) == int or type(item) == np.float32):
                    print(f"{item} is of type {type(item)}")

            # if tensor convert to numpy
            if torch.is_tensor(wav_bytes):
                wav_bytes = wav_bytes.cpu().numpy()
            if isinstance(wav_bytes, list):
                wav_bytes = np.array(wav_bytes)
            play_output_audio(wav_bytes, stop_event)
            time.sleep(SLEEP_TIME_MS)
        else:
            time.sleep(SLEEP_TIME_MS)

def clipboard_listener(manager):
    recent_text = ""
    logging.info("Listening for clipboard changes... Press Ctrl+C to stop.\n")
    
    try:
        while True:
            try:
                current_text = pyperclip.paste()
                if current_text != recent_text and isinstance(current_text, str) and current_text.strip():
                    logging.info("New clipboard text detected")
                    manager.stop_tasks()
                    manager.setText(current_text)
                    manager.start_tasks()
                    logging.debug(current_text)
                    recent_text = current_text
            except PyperclipWindowsException as e:
                print("Clipboard access failed:", e)

            time.sleep(1)  # Check every second
    except KeyboardInterrupt:
        manager.stop_tasks()
        logging.info("\n👋 Listener stopped by user.")

def end_current_play(manager):
    print("Ctrl+X detected! Performing action...")
    if (not manager.stop_event.is_set()):
        print("Stopping..")
        manager.stop_tasks()
        while not audioQueue.empty():
            try:
                audioQueue.get_nowait()
            except Empty:
                pass
            audioQueue.task_done()

# Keyboard listener function
def listen_for_ctrl_x(manager):
    print("Listening for Ctrl+X...")
    keyboard.add_hotkey('ctrl+x', partial(end_current_play, manager))
    while not manager.stop_event.is_set():
        time.sleep(1)
    

def main():
    manager = TaskManager()
    tts = initialize_tts()
    manager.setTTS(tts)
    manager.start_tasks()
    
    try:
        clipboard_listener(manager)
    except KeyboardInterrupt:
        manager.stop_tasks()
        logging.info("Stopped by user")

if __name__ == "__main__":
    main()
