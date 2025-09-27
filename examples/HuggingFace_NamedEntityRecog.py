# file: huggingface_ner_example.py
from transformers import pipeline

ner_pipeline = pipeline("ner", grouped_entities=True, model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "This Company is a leading FinTech company in Athens, Greece."

entities = ner_pipeline(text)
print(entities)
