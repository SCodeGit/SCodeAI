import sys
import json
import urllib.request

MODEL_NAME = "scode-ai:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

def chat():
    print("=" * 60)
    print(" 🚀 SCode AI Interactive Terminal ")
    print(" Type 'exit' or 'quit' to close.")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("\033[1;34mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            payload = json.dumps({"model": MODEL_NAME, "prompt": user_input}).encode("utf-8")
            req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})

            print("\033[1;32mSCode AI:\033[0m ", end="", flush=True)

            with urllib.request.urlopen(req) as response:
                for line in response:
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        print(data.get("response", ""), end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n\033[1;31mError connecting to Ollama:\033[0m {e}\n")

if __name__ == "__main__":
    chat()
