# file: llm_summarization_example.py
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import matplotlib.pyplot as plt
from textwrap import fill

# -----------------------------
# Load TF-native summarization model
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)  # TF weights, no PyTorch needed

# Create summarization pipeline with TensorFlow model
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")

text = """
Random Company is a leading provider of innovative software solutions for the banking and financial sector. 
Founded in 2010, the company has expanded globally, serving clients in Europe, North America, and Asia. 
Their main focus is on artificial intelligence and machine learning technologies, which help financial institutions optimize their operations, detect fraud, and gain actionable insights from vast amounts of data. 
Random Company's suite of products includes predictive analytics, risk management tools, customer experience platforms, and automated compliance systems. 
They have won multiple awards for innovation and excellence in technology, and their team of experts collaborates closely with clients to tailor solutions to each organization’s specific needs. 
The company also invests heavily in research and development to stay ahead in a competitive market and continues to release new features that leverage the latest AI advancements.
"""

# -----------------------------
# Chunk text for summarization
max_chunk_words = 200  # safer for plotting too
words = text.split()
chunks = []
start = 0
while start < len(words):
    end = start + max_chunk_words
    chunks.append(" ".join(words[start:end]))
    start = end

# -----------------------------
# Summarize each chunk
chunk_summaries = []
for chunk in chunks:
    inputs = tokenizer("summarize: " + chunk, return_tensors="tf", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, min_length=40, do_sample=False)
    chunk_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    chunk_summaries.append(chunk_summary)

# Final combined summary
summary_text = " ".join(chunk_summaries)

# -----------------------------
# Plot original text and summary
plt.figure(figsize=(8, 6))
plt.axis("off")

# Wrap text to fit nicely in the figure
wrapped_original = fill(text, width=80)
wrapped_summary = fill(summary_text, width=80)

# Original text in red
# Summary in green
# Use transform=plt.gcf().transFigure to place text relative to the full figure
plt.text(0.05, 0.3, f"Original:\n{text}", fontsize=12, color="red", wrap=True, transform=plt.gcf().transFigure)
plt.text(0.05, 0.1, f"Summary:\n{summary_text}", fontsize=12, color="green", wrap=True, transform=plt.gcf().transFigure)
plt.suptitle("Text Summarization", fontsize=14)  # suptitle for figure title
plt.tight_layout()

plt.savefig("../plot_summarization.png", dpi=150)
plt.show()