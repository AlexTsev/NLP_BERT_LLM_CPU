# 1. Imports
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# 2. Load pre-trained model & tokenizer (English BERT)
#model_name = "bert-base-uncased"  # switch to English

# 2. Load pre-trained model & tokenizer (GreekBERT)
model_name = "nlpaueb/bert-base-greek-uncased-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Get the path relative to this script
csv_file = os.path.join("..", "data_augmentation", "dataset_generated.csv")

# Load the CSV
df = pd.read_csv(csv_file)
texts = df['text'].tolist()
labels = df['label'].tolist()

# -----------------------------
# 3. Stratified train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# -----------------------------
# 4. Δες τα token IDs και attention mask για λίγα δείγματα
sample_texts = train_texts[:3]
sample_labels = train_labels[:3]

encoded_sample = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="tf", max_length=64)

print("=== Sample texts and labels ===")
for t, l in zip(sample_texts, sample_labels):
    print(f"Text: {t} -> Label: {l}")

print("\n=== Token IDs ===")
print(encoded_sample["input_ids"].numpy())

print("\n=== Attention Mask ===")
print(encoded_sample["attention_mask"].numpy())

print("\n=== NLP Model Started Fine-Tuning BERT Training ===")

# -----------------------------
# 5. Tokenization function
def encode(texts, labels):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="tf", max_length=64)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, tf.convert_to_tensor(labels)

train_dataset = tf.data.Dataset.from_tensor_slices(encode(train_texts, train_labels)).shuffle(len(train_texts)).batch(4)
val_dataset = tf.data.Dataset.from_tensor_slices(encode(val_texts, val_labels)).batch(4)

# -----------------------------
# 6. Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# -----------------------------
# 7. Train model with validation
model_hist = model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# -----------------------------
# 8. Inference
def predict(text):
    enc = tokenizer(text, padding=True, truncation=True, return_tensors="tf", max_length=64)
    logits = model(enc)["logits"]
    pred = tf.argmax(logits, axis=1).numpy()[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[pred]

# Test predictions
for t in texts[:10]:  # βάζω τα πρώτα 10 για να μην γεμίσει η κονσόλα
    print(f"Text: {t} -> Sentiment: {predict(t)}")

# -----------------------------
# 9. Plot Loss
plt.plot(model_hist.history['loss'], label='Train Loss')
plt.plot(model_hist.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Accuracy
plt.plot(model_hist.history['accuracy'], label='Train Accuracy')
plt.plot(model_hist.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()