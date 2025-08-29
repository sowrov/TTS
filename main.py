import os
import sys
import torch
import pyperclip
import time
import wave
import pyaudio
import torchaudio
from TTS.api import TTS

outputFile = "output.wav"

def initialize_tts():
    """Initialize CUDA device and TTS model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("CUDA installed successfully\n")
    else:
        print("CUDA not properly installed. Stopping process...")
        quit()

    # view_models = input("View models? [y/n]\n")
    # if view_models.lower() == "y":
    #     tts_manager = TTS().list_models()
    #     all_models = tts_manager.list_models()
    #     print("TTS models:\n", all_models, "\n")

    # model = input("Enter model name:\n")  # e.g., tts_models/multilingual/multi-dataset/xtts_v2
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
    return tts

def synthesize_speech(tts, text):
    """Generate audio from text using the TTS model."""
    tts.tts_to_file(
        text=text,
        speaker_wav="male.wav",
        language="en",
        file_path= outputFile
    )
    print("Speech synthesis complete. Saved to %s", outputFile)

def play_output_audio():
    # Open the WAV file
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


def clipboard_listener(tts):
    recent_text = ""
    print("Listening for clipboard changes... Press Ctrl+C to stop.\n")
    
    try:
        while True:
            current_text = pyperclip.paste()
            if current_text != recent_text and isinstance(current_text, str) and current_text.strip():
                print("New clipboard text detected")
                synthesize_speech(tts, current_text)
                play_output_audio()
                # print(current_text)
                # print("-" * 40)

                recent_text = current_text
            time.sleep(1)  # Check every second
    except KeyboardInterrupt:
        print("\n👋 Listener stopped by user.")

def main():
    """Main function to read file and trigger TTS."""
    # Get file name from command-line argument or prompt
    # if len(sys.argv) > 1:
    #     file_path = sys.argv[1]
    # else:
    #     file_path = input("Enter path to your text file:\n")

    # if not os.path.isfile(file_path):
    #     print("Error: File does not exist.")
    #     return
    # if not os.access(file_path, os.R_OK):
    #     print("Error: File is not readable.")
    #     return

    # with open(file_path, 'r', encoding='utf-8') as f:
    #     text = f.read()

    tts = initialize_tts()
    
    clipboard_listener(tts)

if __name__ == "__main__":
    main()
