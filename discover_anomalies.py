"""
Discover Anomalies - v10.2 Edition

Analyzes the latest SOTA model (v10.2 with Lexicon mappings) to find
interesting translations and potential anomalies.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from gensim.models import FastText, KeyedVectors
from sklearn.linear_model import Ridge

# Configuration - v10.2
REPO_ROOT = Path(__file__).parent
V10_ROOT = REPO_ROOT / "heiro_v10_refinement"
RESULTS_FILE = V10_ROOT / "results_v10.2.json"

# Model paths
HIEROGLYPHIC_MODEL = REPO_ROOT / "heiro_v7_FastTextVisual/models/fasttext_v7.model"
VISUAL_EMBEDDINGS = REPO_ROOT / "heiro_v9_use_visuals_again/data/processed/visual_embeddings_768d.pkl"
ENGLISH_EMBEDDINGS = REPO_ROOT / "heiro_v5_getdata/data/processed/glove.6B.300d.txt"
ANCHORS_FILE = REPO_ROOT / "heiro_v6_BERT/data/processed/anchors.json"
LEXICON_MAPPING = V10_ROOT / "data/lexicon_trans_to_codes.json"

print("=" * 60)
print("🔍 DISCOVER ANOMALIES - v10.2 Edition")
print("=" * 60)

# Load results
if RESULTS_FILE.exists():
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    print(f"\n✓ Latest Results (v10.2):")
    print(f"  Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"  Visual Match Rate: {results['visual_match_rate']*100:.2f}%")
    print(f"  Lexicon Size: {results['lexicon_size']}")
else:
    print("\n⚠ No results found. Run v10.2 notebook first.")
    exit(1)

print("\nLoading models...")
hier_model = FastText.load(str(HIEROGLYPHIC_MODEL))
text_embeddings = hier_model.wv

# Load visual embeddings
with open(VISUAL_EMBEDDINGS, 'rb') as f:
    visual_embeddings = pickle.load(f)

# Load Lexicon mapping
with open(LEXICON_MAPPING, 'r') as f:
    trans_to_codes = json.load(f)

# Load English embeddings
print("Loading GloVe (may take a moment)...")
english_embeddings = KeyedVectors.load_word2vec_format(
    str(ENGLISH_EMBEDDINGS), binary=False, no_header=True
)

# Load anchors
with open(ANCHORS_FILE, 'r') as f:
    anchors = json.load(f)

print("\nCreating fused embeddings...")
# Recreate fused embeddings (same logic as v10.2)
fused_embeddings = {}
for word in text_embeddings.index_to_key[:5000]:  # Limit for speed
    text_vec = text_embeddings[word]
    
    # Look up visual
    visual_vec = None
    if word in trans_to_codes:
        code_sequences = trans_to_codes[word]
        vectors = []
        for seq in code_sequences:
            seq_vectors = [visual_embeddings[code] for code in seq if code in visual_embeddings]
            if seq_vectors:
                vectors.append(np.mean(seq_vectors, axis=0))
        if vectors:
            visual_vec = np.mean(vectors, axis=0)
    
    if visual_vec is None:
        visual_vec = np.zeros(768)
    
    fused_embeddings[word] = np.concatenate([text_vec, visual_vec])

print("\nTraining alignment model...")
# Recreate the aligner
X, Y = [], []
for anchor in anchors:
    egy = anchor['hieroglyphic']
    eng = anchor['english'].lower()
    if egy in fused_embeddings and eng in english_embeddings:
        X.append(fused_embeddings[egy])
        Y.append(english_embeddings[eng])

X, Y = np.array(X), np.array(Y)
aligner = Ridge(alpha=1.0)
aligner.fit(X, Y)

print(f"✓ Trained on {len(X)} anchors")

# Interesting Words to Probe
probe_words = {
    "nfr": "Good/Beautiful",
    "ꜥnḫ": "Life",
    "Wsjr": "Osiris",
    "Ḥr,w": "Horus",
    "rʾ": "Sun/Re",
    "jnpw": "Anubis",
    "mꜣꜥ.t": "Maat/Truth",
    "nsw": "King",
    "nṯr": "God",
    "mw": "Water",
    "tꜣ": "Land/Bread",
    "ḥnq.t": "Beer",
    "pr": "House",
    "mꜣꜥ-ḫrw": "True of Voice (justified)",
    "ḏd": "Djed/Stability"
}

def get_translations(egy_word, k=10):
    """Get top-k English translations for an Egyptian word."""
    if egy_word not in fused_embeddings:
        return ["(Not in Vocab)"]
    
    # Project to English space
    egy_vec = fused_embeddings[egy_word]
    eng_pred = aligner.predict([egy_vec])[0]
    
    # Find nearest neighbors
    neighbors = english_embeddings.similar_by_vector(eng_pred, topn=k)
    return [f"{word} ({sim:.3f})" for word, sim in neighbors]

print("\n" + "=" * 60)
print("🔎 DISCOVERIES - Top Translations")
print("=" * 60)

for egy_word, meaning in probe_words.items():
    print(f"\n📜 {egy_word} ({meaning})")
    translations = get_translations(egy_word)
    for i, trans in enumerate(translations, 1):
        print(f"  {i:2d}. {trans}")

# Show words with visual features
print("\n" + "=" * 60)
print("🖼️  WORDS WITH VISUAL FEATURES")
print("=" * 60)

words_with_visuals = [w for w in probe_words.keys() if w in trans_to_codes]
print(f"\nFound {len(words_with_visuals)}/{len(probe_words)} probe words with visual mappings:")
for word in words_with_visuals:
    codes = trans_to_codes[word][0] if trans_to_codes[word] else []
    print(f"  {word}: {' + '.join(codes)}")

print("\n" + "=" * 60)
