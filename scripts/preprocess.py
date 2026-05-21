import pandas as pd
import os

print(">>> Initializing Python-based NLP Preprocessing Phase...")

# Ensure destination directory exists
os.makedirs("data/processed", exist_ok=True)

# Placeholder: In a full pipeline, load your raw data here.
# For now, we simulate clean splits for training.
train_data = {"text": ["I love this implementation!", "This code is totally broken."], "label": [1, 0]}
test_data = {"text": ["Super clean architecture.", "Terrible error parsing strings."], "label": [1, 0]}

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

df_train.to_csv("data/processed/train.csv", index=False)
df_test.to_csv("data/processed/test.csv", index=False)

print(">>> Preprocessing complete! Training sets saved inside data/processed/.")
