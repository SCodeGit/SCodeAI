import os
import json
import glob

OUTPUT_FILE = "./datasets/combined_scode_dataset.jsonl"
DATASETS_DIR = "./datasets"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
processed_records = []

def process_item(instruction, input_text, output):
    inst = str(instruction or "").strip()
    inp = str(input_text or "").strip()
    out = str(output or "").strip()
    
    if inst and out:
        return {
            "instruction": inst,
            "input": inp,
            "output": out
        }
    return None

print("=== Starting Dataset Merging & Formatting ===")

# Search all subdirectories for json, jsonl, and parquet metadata files
json_files = glob.glob(f"{DATASETS_DIR}/**/*.json*", recursive=True)

for file_path in json_files:
    # Skip output file if re-running
    if "combined_scode_dataset" in file_path:
        continue
        
    print(f"Reading {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Handle both JSON array files and JSONL files
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]

            for item in data:
                if not isinstance(item, dict):
                    continue
                    
                # Standardize keys across datasets
                inst = item.get("instruction") or item.get("question") or item.get("prompt")
                inp = item.get("input") or ""
                out = item.get("output") or item.get("response") or item.get("answer") or item.get("solution")
                
                rec = process_item(inst, inp, out)
                if rec:
                    processed_records.append(rec)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

print(f"\nWriting {len(processed_records)} total records to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in processed_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

