# file: bert_classification_example.py
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
#from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
from matplotlib import cm

# Προεκπαιδευμένο BERT για text classification
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
#model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

texts = [
    "The bank provides excellent customer service.",
    "The user experience was frustrating.",
    "The application works as expected.",
    "The product was completely useless and disappointed me.",
    "The service was good but not outstanding."
]

# Map 5-star rating to sentiment labels
def map_star_to_sentiment(star_label):
    star = int(star_label[0])  # '4 stars' -> 4
    mapping = {
        1: "Very Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Very Positive"
    }
    return mapping.get(star, "Unknown")

# Assign distinct colors for each sentiment
sentiment_colors = {
    "Very Negative": "darkred",
    "Negative": "red",
    "Neutral": "orange",
    "Positive": "green",
    "Very Positive": "#006400"
}

# Collect predictions
predictions = []
for text in texts:
    result = classifier(text)[0]
    label_text = map_star_to_sentiment(result['label'])
    predictions.append((text, label_text))
    print(f"Sentence: {text} -> Sentiment: {label_text} ({result['label']}, score={result['score']:.3f})")

# -----------------------------
# Plot sentences with predictions
plt.figure(figsize=(8, 6))
plt.axis("off")

y_pos = 0.8
for text, sentiment in predictions:
    plt.text(0.01, y_pos, f"Sentence:\n{text}", fontsize=12, color="black", wrap=True)
    y_pos -= 0.12
    plt.text(0.02, y_pos, f"Prediction: {sentiment}", fontsize=12, color=sentiment_colors.get(sentiment, "blue"), wrap=True)
    y_pos -= 0.15

plt.title("Sentiment Classification (5 Stars)", fontsize=14)
plt.tight_layout()
plt.savefig("../plot_bert_sentiment.png", dpi=150)
plt.show()