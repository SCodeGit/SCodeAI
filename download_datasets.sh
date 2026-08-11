#!/bin/bash

# Activate virtual environment
source ~/hfenv/bin/activate

# Create output directories
mkdir -p ./datasets/codealpaca
mkdir -p ./datasets/evol_instruct
mkdir -p ./datasets/magicoder
mkdir -p ./datasets/glaive_code
mkdir -p ./datasets/humaneval

echo "=== Starting Dataset Downloads ==="

# 1. CodeAlpaca 20K (General Code Instructions)
echo "Downloading CodeAlpaca 20K..."
hf download HuggingFaceH4/CodeAlpaca_20K --repo-type dataset --local-dir ./datasets/codealpaca

# 2. Evol-Instruct-Code-80k (Complex Multi-step Coding Instructions)
echo "Downloading Evol-Instruct-Code-80k..."
hf download nickrosh/Evol-Instruct-Code-80k --repo-type dataset --local-dir ./datasets/evol_instruct

# 3. Magicoder-Evol-Instruct-110K (High-Quality Synthetic Code Instructions)
echo "Downloading Magicoder-Evol-Instruct-110K..."
hf download ise-uiuc/Magicoder-Evol-Instruct-110K --repo-type dataset --local-dir ./datasets/magicoder

# 4. Glaive Code Assistant v2 (Conversational & Function-Calling Code Data)
echo "Downloading Glaive Code Assistant v2..."
hf download glaiveai/glaive-code-assistant-v2 --repo-type dataset --local-dir ./datasets/glaive_code

# 5. HumanEval (Evaluation Benchmark)
echo "Downloading OpenAI HumanEval..."
hf download openai/openai_humaneval --repo-type dataset --local-dir ./datasets/humaneval

echo "=== All Downloads Complete ==="
