# voice_assistant.py
import vosk
import sys
import sounddevice as sd
import queue
import json
import subprocess
import pyttsx3
import re
import os
from transformers import pipeline

# ----------------------------------------
# Load universal pretrained model (thinking brain)
model_llm = pipeline("text-generation", model="codellama/CodeLlama-7b-Instruct-hf")

# ----------------------------------------
# Initialize Vosk speech recognition
q = queue.Queue()
model_vosk = vosk.Model("model")  # ensure you have a Vosk model folder named "model"
samplerate = 16000
device = None

def callback(indata, frames, time, status):
    q.put(bytes(indata))

# ----------------------------------------
# Initialize pyttsx3 TTS (voice output)
engine = pyttsx3.init()

# ----------------------------------------
# Intent parser (dynamic, not hard-coded dictionary)
def interpret_command(text):
    text = text.lower()

    # --- File operations ---
    if "open folder" in text:
        folder = text.split("open folder")[-1].strip()
        return f"xdg-open {folder}"

    elif "rename folder" in text:
        match = re.search(r"rename folder (\w+) to (\w+)", text)
        if match:
            old_name, new_name = match.groups()
            return f"mv {old_name} {new_name}"

    elif "delete file" in text:
        file = text.split("delete file")[-1].strip()
        return f"rm {file}"

    elif "list files" in text:
        return "ls -la"

    # --- System control ---
    elif "shutdown" in text:
        return "shutdown now"
    elif "restart" in text:
        return "reboot"

    # --- Networking ---
    elif "ping" in text:
        target = text.split("ping")[-1].strip()
        return f"ping -c 4 {target}"

    # --- Default fallback ---
    else:
        return None

def run_command(cmd):
    if cmd:
        os.system(cmd)
        return f"Executed: {cmd}"
    else:
        return "Sorry, I didn’t understand that command."

# ----------------------------------------
# Main loop
def main():
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device,
                           dtype='int16', channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model_vosk, samplerate)
        print("Listening...")

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print(f"You said: {text}")

                    # Step 1: Try to interpret as OS command
                    cmd = interpret_command(text)
                    response = run_command(cmd)

                    # Step 2: If not an OS command, let LLM respond
                    if not cmd:
                        llm_output = model_llm(text, max_length=100, do_sample=True)[0]["generated_text"]
                        response = llm_output

                    print(f"Assistant: {response}")

                    # Speak response (safe fallback)
                    try:
                        engine.say(response)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"(TTS failed: {e})")

if __name__ == "__main__":
    main()

