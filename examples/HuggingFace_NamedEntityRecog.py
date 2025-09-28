# file: huggingface_ner_example.py
from transformers import pipeline
import matplotlib.pyplot as plt
from matplotlib import cm  # colormap utilities
import random   # to pick random colors

ner_pipeline = pipeline("ner", grouped_entities=True, model="dbmdz/bert-large-cased-finetuned-conll03-english")

sentences = [
    "This company Random is a leading fintech company in Athens, Greece.",
    "Barack Obama visited the Olympic Games in Paris."
]


entity_colors = {
    "ORG": "tab:blue",
    "LOC": "tab:green",
    "PER": "tab:red",
    "MISC": "tab:orange"
}

# Prepare text for plotting
# Build a string with entity highlights: e.g. "Company [ORG]"
plt.figure(figsize=(9, 5))
plt.axis("off")
plt.title("Named Entity Recognition", fontsize=15)

y_pos = 0.9  # vertical start position

for idx, text in enumerate(sentences, start=1):
    # Run NER
    entities = ner_pipeline(text)
    print(f"Sentence {idx} Entities:", entities)

    # Show the sentence itself
    plt.text(
        0.01, y_pos,
        f"Sentence {idx}: {text}",
        fontsize=12,
        wrap=True
    )
    y_pos -= 0.08

    # Show the entities, one per line, color-coded by label
    for ent in entities:
        ent_text = ent["word"]
        ent_label = ent["entity_group"]
        color = entity_colors.get(ent_label, "black")
        plt.text(
            0.03, y_pos,
            f"{ent_text}  [{ent_label}]",
            fontsize=12,
            color=color,
            wrap=True
        )
        y_pos -= 0.06

    # Add a small gap before the next sentence
    y_pos -= 0.06

# Save the figure
plt.tight_layout()
plt.savefig("../plot_ner_entities.png", dpi=150)
plt.show()