#!/bin/bash
source ~/scode-env/bin/activate

# Fine-tune SCODE AI
python3 train.py --model CodeLlama-7B --dataset data.json --output scode_ai_weights

# Serve SCODE AI with voice + OS commands
python3 voice_assistant.py --model scode_ai_weights --speech vosk --tts coqui

