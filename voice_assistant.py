import sys
import json
import requests
import vosk
import pyaudio
import pyttsx3
import torch
from diffusers import StableDiffusionPipeline
from multimodal_handler import process_file

# 1. Text to Speech
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    print(f"\n[SCode AI]: {text}\n")
    engine.say(text)
    engine.runAndWait()

# 2. Local LLM Query
OLLAMA_URL = "http://localhost:11434/api/chat"

def query_scode_ai(prompt, context=""):
    full_prompt = prompt
    if context:
        full_prompt = f"Context from file:\n{context}\n\nUser Question: {prompt}"

    payload = {
        "model": "qwen2.5-coder:7b",
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if res.status_code == 200:
            return res.json().get("message", {}).get("content", "")
        return f"Ollama error: {res.status_code}"
    except Exception as e:
        return f"Connection failed: {e}"

# 3. CPU Image Generation
def generate_image(prompt):
    try:
        speak("Generating image locally on CPU...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float32,
            safety_checker=None
        )
        pipe.to("cpu")
        pipe.enable_attention_slicing()
        
        img = pipe(prompt, num_inference_steps=12).images[0]
        img.save("output.png")
        return "Image generated and saved as output.png."
    except Exception as e:
        return f"Image generation failed: {e}"

# 4. Listener & Multi-intent Processing
def listen_and_process():
    model = vosk.Model("vosk-model-small-en-us-0.15")
    rec = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    print("\n=== SCode AI Multimodal System Active ===")
    speak("SCode AI ready for text, files, and image generation.")

    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                user_text = json.loads(rec.Result()).get("text", "").strip()
                if not user_text:
                    continue

                print(f"[You]: {user_text}")
                text_lower = user_text.lower()

                # Intent 1: Image Generation
                if any(k in text_lower for k in ["generate image", "draw", "create picture"]):
                    prompt = text_lower.replace("generate image", "").replace("draw", "").strip()
                    res = generate_image(prompt)
                    speak(res)

                # Intent 2: File / Image / Audio Processing
                elif "read file" in text_lower or "analyze file" in text_lower or "read document" in text_lower:
                    speak("Please enter the full path to the file:")
                    file_path = input("File Path: ").strip()
                    file_content = process_file(file_path)
                    
                    speak("File processed. Ask your question about this file:")
                    user_question = input("Question: ").strip()
                    
                    reply = query_scode_ai(user_question, context=file_content)
                    speak(reply)

                # Intent 3: General Query
                else:
                    reply = query_scode_ai(user_text)
                    speak(reply)

    except KeyboardInterrupt:
        print("\nStopping SCode AI...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    listen_and_process()
