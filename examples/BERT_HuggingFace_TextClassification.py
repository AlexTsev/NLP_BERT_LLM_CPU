# file: bert_classification_example.py
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
#from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Προεκπαιδευμένο BERT για text classification
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
#model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

texts = [
    "Η τράπεζα προσφέρει εξαιρετική εξυπηρέτηση.",
    "Η εμπειρία του χρήστη ήταν πολύ κακή.",
    "Η εφαρμογή λειτουργεί όπως αναμένεται."
]

for text in texts:
    result = classifier(text)
    print(text, "->", result)
