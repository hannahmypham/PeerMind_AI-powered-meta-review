import argparse
import json
import csv
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from peft import PeftModel, PeftConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_path", type=str, default="data/processed/test.jsonl")
parser.add_argument("--output_path", type=str, default="")
args = parser.parse_args()

MODEL_PATH = args.model_path
TEST_PATH = args.test_path
run_name = Path(MODEL_PATH).name
OUTPUT_PATH = args.output_path or f"data/predictions/{run_name}_predictions.csv"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Model: {MODEL_PATH}")

# Load LoRA config to get base model name
peft_config = PeftConfig.from_pretrained(MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    peft_config.base_model_name_or_path
)

model = PeftModel.from_pretrained(base_model, MODEL_PATH)

# Optional but recommended: merge LoRA weights for inference
model = model.merge_and_unload()
model.to(device)
model.eval()

Path("data/predictions").mkdir(parents=True, exist_ok=True)

def extract_decision(text):
    text = (text or "").strip()
    if "DECISION:" in text:
        after = text.split("DECISION:", 1)[1].strip()
        first_line = after.split("\n", 1)[0].strip()
        up = first_line.upper()
        if up.startswith("ACCEPT"):
            return "ACCEPT"
        if up.startswith("REJECT"):
            return "REJECT"
        return first_line
    return "UNKNOWN"

rows = []

with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        ex = json.loads(line)

        paper_id = ex["paper_id"]           # ✅ guaranteed
        input_text = ex["input_text"]
        true_output = ex["target_text"]

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=768
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                length_penalty=0.8,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
            )

        pred_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        rows.append({
            "paper_id": paper_id,
            "input_text": input_text,
            "true_decision": extract_decision(true_output),
            "pred_decision": extract_decision(pred_text),
            "true_meta_review": true_output,
            "pred_meta_review": pred_text
        })

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved predictions to {OUTPUT_PATH}")