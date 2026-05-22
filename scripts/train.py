import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset

print(">>> Commencing DistilBERT Fine-Tuning...")

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load preprocessed data
df_train = pd.read_csv("data/processed/train.csv")
df_test = pd.read_csv("data/processed/test.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

class SentimentDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(list(df["text"]), truncation=True, padding=True, max_length=128)
        self.labels = list(df["label"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(df_train)
eval_dataset = SentimentDataset(df_test)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save in HuggingFace format (tokenizer.json + config.json + model weights)
# DJL DeferredTranslatorFactory reads this layout natively
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Also export TorchScript .pt for DJL's PyTorch engine
dummy_input = tokenizer("test", return_tensors="pt")
input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]

traced = torch.jit.trace(model, (input_ids, attention_mask), strict=False)
torch.jit.save(traced, os.path.join(OUTPUT_DIR, "model_output.pt"))

print(f">>> Training complete. Model artifacts saved to {OUTPUT_DIR}/")
print(">>> Files:", os.listdir(OUTPUT_DIR))