"""
Downloads a properly fine-tuned sentiment model from HuggingFace
(distilbert-base-uncased-finetuned-sst-2-english) and exports it as .pt

Usage:
    pip install transformers torch safetensors
    python convert_to_pt.py
"""
import torch
import shutil
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This is a properly fine-tuned sentiment model from HuggingFace Hub
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = "model_output"

print(f">>> Downloading fine-tuned model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Save HuggingFace files (config.json, tokenizer.json etc)
print(">>> Saving model files...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Export TorchScript .pt for DJL
print(">>> Exporting TorchScript .pt ...")
dummy = tokenizer("sample text", return_tensors="pt", padding=True, truncation=True, max_length=128)
with torch.no_grad():
    traced = torch.jit.trace(model, (dummy["input_ids"], dummy["attention_mask"]), strict=False)

torch.jit.save(traced, os.path.join(OUTPUT_DIR, "model_output.pt"))
print(">>> Done! model_output/ is ready.")
print(">>> Files:", os.listdir(OUTPUT_DIR))