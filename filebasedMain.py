import os
import sys
import torch
import pyperclip
import time
import wave
import pyaudio
import torchaudio
import re
from TTS.api import TTS
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
import glob

logging.basicConfig(level=logging.DEBUG) #

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
TEN_MS = 10/1000.0

def initialize_tts():
    """Initialize CUDA device and TTS model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logging.debug("CUDA installed successfully\n")
    else:
        logging.debug("CUDA not properly installed. Stopping process...")
        quit()

    # view_models = input("View models? [y/n]\n")
    # if view_models.lower() == "y":
    #     tts_manager = TTS().list_models()
    #     all_models = tts_manager.list_models()
    #     logging.debug("TTS models:\n", all_models, "\n")

    # model = input("Enter model name:\n")  # e.g., tts_models/multilingual/multi-dataset/xtts_v2
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
    # tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True).to(device)
    return tts

def synthesize_speech(tts, text):
    """Generate audio from text using the TTS model."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_path = os.path.join(AUDIO_DIR, f"{timestamp}.wav")
    logging.debug("at synthesize_speech")
    tts.tts_to_file(
        text=text,
        speaker_wav="male.wav",
        language="en",
        file_path= file_path
    )
    logging.debug("Speech synthesis complete. Saved to %s", file_path)

def audio_generator(tts, text, stop_event):
    text = text.replace('\n\r', ' ').replace('\n', ' ')
    lines = split_text_into_lines(text)
    for line in lines:
        if stop_event.is_set():
            break
        else:
            synthesize_speech(tts, line)
        time.sleep(TEN_MS)

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

def play_output_audio(outputFile):
    # Open the WAV file
    try:
        wf = wave.open(outputFile, 'rb')
        p = pyaudio.PyAudio()

        # Open stream
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Read and play the audio
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
    except KeyboardInterrupt:
        logging.info("\n👋 Listener stopped by user.")

def audio_player(stop_event):
    while not stop_event.is_set():
        wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")), key=os.path.getctime)
        if wav_files:
            oldest_file = wav_files[0]
            play_output_audio(oldest_file)
            os.remove(oldest_file)
            logging.debug(f"Deleted: {oldest_file}")
            time.sleep(TEN_MS)
        else:
            time.sleep(2)

def clipboard_listener(manager):
    recent_text = ""
    logging.info("Listening for clipboard changes... Press Ctrl+C to stop.\n")
    
    try:
        while True:
            current_text = pyperclip.paste()
            if current_text != recent_text and isinstance(current_text, str) and current_text.strip():
                logging.info("New clipboard text detected")
                manager.stop_tasks()
                manager.setText(current_text)
                manager.start_tasks()
                # synthesize_speech(tts, current_text)
                # play_output_audio()
                logging.debug(current_text)
                recent_text = current_text
            time.sleep(1)  # Check every second
    except KeyboardInterrupt:
        manager.stop_tasks()
        logging.info("\n👋 Listener stopped by user.")

def clear_old_wavs():
    try:
        for filename in os.listdir(AUDIO_DIR):
            if filename.endswith(".wav"):
                file_path = os.path.join(AUDIO_DIR, filename)
                try:
                    os.remove(file_path)
                    logging.debug(f"Deleted: {file_path}")
                except OSError as e:
                    logging.error(f"Error deleting {file_path}: {e}")

    except FileNotFoundError:
        logging.error(f"Folder not found: {AUDIO_DIR}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    clear_old_wavs()
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
