import nlpaug.augmenter.word as naw
import nltk

# Download necessary NLTK resources
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

print("Resources installed successfully!")

# -------------------------
# 1. Original Dataset
# -------------------------
texts = [
    # Positive (2)
    "The app is fantastic", "The service was excellent", "The product works perfectly",
    "My experience was wonderful", "The bank is very reliable", "The service is excellent",
    "The staff was polite", "The quality is excellent", "The delivery was very fast",
    "Customer support is flawless", "The in-store experience was wonderful", "The transaction was quick and secure",
    "The website is user-friendly", "The app works without issues", "The product exceeded my expectations",
    "The user experience is excellent", "Communication with the company was effective",
    "The prices are fair and competitive",
    "Delivery was on time", "The product is top quality",

    # Neutral (1)
    "The product is okay", "The service is satisfactory", "The bank has good services",
    "The app is functional", "The store was relatively organized", "The payment was completed without issues",
    "The process was standard", "Delivery occurred as expected", "The service was average",
    "The app had no errors", "The experience was neutral", "The product meets basic needs",
    "Communication was consistent", "The registration process was simple", "The website works properly",
    "The app is stable", "Payment completed without problems", "The staff was neutral",
    "The company provides basic information", "The experience was as expected",

    # Negative (0)
    "The service was unacceptable", "The product was broken", "The app keeps freezing",
    "My experience was bad", "The bank delayed the service", "The staff was rude",
    "Delivery was very late", "The product quality is poor", "The website does not work properly",
    "Customer support is inadequate", "Payment failed multiple times", "The app is difficult to use",
    "Communication with the company was disappointing", "The store was messy", "The user experience is bad",
    "Prices are excessive for the quality", "Delivery was significantly delayed",
    "The product does not meet expectations",
    "The app had many errors", "The service was very poor"
]

labels = [
    # Positive
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    # Neutral
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    # Negative
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

# -------------------------
# 2. Augmentation Setup
# -------------------------
# Synonym augmentation with 30% word replacement probability
aug = naw.SynonymAug(aug_src='wordnet', lang='eng', aug_p=0.3)

aug_texts, aug_labels = [], []

# -------------------------
# 3. Generate Augmented Dataset
# -------------------------
for text, label in zip(texts, labels):
    aug_texts.append(text)
    aug_labels.append(label)

    # Generate up to 24 augmentations
    augmented_list = aug.augment(text, n=24)

    # Remove duplicates
    unique_augmented = list(set(augmented_list))

    for aug_text in unique_augmented:
        aug_texts.append(aug_text)
        aug_labels.append(label)

print(f"Original texts: {len(texts)}")
print(f"Augmented dataset size: {len(aug_texts)}")

# -------------------------
# 4. Save to File
# -------------------------
with open("dataset_generated.txt", "w", encoding="utf-8") as f:
    f.write("# 3. Dataset\ntexts = [\n")
    for t, l in zip(aug_texts, aug_labels):
        f.write(f'    "{t}",  # Label={l}\n')
    f.write("]\n\nlabels = [\n")
    for l in aug_labels:
        f.write(f"    {l},\n")
    f.write("]\n")

print("Dataset saved to 'dataset_generated.txt'.")
