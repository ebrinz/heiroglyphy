import os
import pickle
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm

# Configuration
DATA_DIR = "heiro_v3/data"
CACHE_FILE = os.path.join(DATA_DIR, "german_english_translations.pkl")
HIEROGLYPHIC_CORPUS_FILE = os.path.join(DATA_DIR, "hieroglyphic_corpus.txt")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors.pkl")
CLEAN_CORPUS_FILE = os.path.join(DATA_DIR, "clean_corpora.pkl")

print(f"Loading data from {CACHE_FILE}...")
with open(CACHE_FILE, 'rb') as f:
    raw_data = pickle.load(f)

print(f"Loaded {len(raw_data)} entries.")

def clean_hieroglyphic(text):
    """Normalize hieroglyphic transliteration."""
    if not isinstance(text, str): return ""
    # Remove brackets and uncertain markers often found in TLA
    text = re.sub(r'[\[\]\(\)\?\<\>]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_english(text):
    """Normalize English text."""
    if not isinstance(text, str): return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

hieroglyphic_sentences = []
english_sentences = []

print("Cleaning Data...")
for entry in tqdm(raw_data, desc="Cleaning Data"):
    h_clean = clean_hieroglyphic(entry.get('hieroglyphic', ''))
    e_clean = clean_english(entry.get('english', ''))
    
    if h_clean and e_clean:
        hieroglyphic_sentences.append(h_clean)
        english_sentences.append(e_clean)

print(f"\nPrepared {len(hieroglyphic_sentences)} parallel sentences.")

co_occurrence = Counter()
h_freq = Counter()
e_freq = Counter()

print("Building co-occurrence matrix...")
for h_sent, e_sent in zip(hieroglyphic_sentences, english_sentences):
    h_words = set(h_sent.split())
    e_words = set(e_sent.split())
    
    for h in h_words:
        h_freq[h] += 1
        for e in e_words:
            co_occurrence[(h, e)] += 1
            e_freq[e] += 1

print(f"Unique Hieroglyphic words: {len(h_freq)}")
print(f"Unique English words: {len(e_freq)}")

anchors = {}
MIN_COUNT = 5
CONFIDENCE_THRESHOLD = 0.3

print("Extracting anchors...")
for (h, e), count in co_occurrence.items():
    if count < MIN_COUNT:
        continue
        
    # Conditional probability P(e|h)
    prob = count / h_freq[h]
    
    if prob > CONFIDENCE_THRESHOLD:
        # We keep the best translation for each hieroglyphic word
        if h not in anchors or anchors[h]['prob'] < prob:
            anchors[h] = {'english': e, 'prob': prob, 'count': count}

print(f"Found {len(anchors)} potential anchors.")

# Save Corpora
with open(CLEAN_CORPUS_FILE, 'wb') as f:
    pickle.dump({
        'hieroglyphic': hieroglyphic_sentences,
        'english': english_sentences
    }, f)

# Save Anchors
anchor_list = [{'hieroglyphic': h, 'english': data['english']} for h, data in anchors.items()]
with open(ANCHOR_FILE, 'wb') as f:
    pickle.dump(anchor_list, f)

print("Data saved successfully.")
