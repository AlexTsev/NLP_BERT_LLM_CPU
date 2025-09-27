# file: llm_summarization_example.py
from transformers import pipeline

# Pretrained LLM για summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
This Company provides software solutions to banking and financial institutions. 
They focus on AI and machine learning to optimize processes and deliver insights.
"""

summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
print("Original:", text)
print("Summary:", summary[0]['summary_text'])
