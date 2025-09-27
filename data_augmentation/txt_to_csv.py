import pandas as pd

texts = []
labels = []

# 1. Load the txt file as Python code
dataset_file = "dataset_generated.txt"
with open(dataset_file, "r", encoding="utf-8") as f:
    data = f.read()

# This will define `texts` and `labels` in the current scope
exec(data)

# 2. Create a DataFrame
df = pd.DataFrame({"text": texts, "label": labels})

# 3. Save as CSV
csv_file = "dataset_generated.csv"
df.to_csv(csv_file, index=False, encoding="utf-8")

print(f"Saved {len(df)} samples to {csv_file}")
