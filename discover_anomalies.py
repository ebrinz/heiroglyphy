import os
import pickle
import numpy as np
from gensim.models import FastText, Word2Vec

# Configuration
DATA_DIR = "heiro_v4/data"
MODELS_DIR = "heiro_v4/models"
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors.pkl")
HIEROGLYPHIC_MODEL_FILE = os.path.join(MODELS_DIR, "hieroglyphic_fasttext.model")
ENGLISH_MODEL_FILE = os.path.join(MODELS_DIR, "english_word2vec.model")

print("Loading models...")
hier_model = FastText.load(HIEROGLYPHIC_MODEL_FILE)
eng_model = Word2Vec.load(ENGLISH_MODEL_FILE)

print("Loading anchors...")
with open(ANCHOR_FILE, 'rb') as f:
    anchors = pickle.load(f)

# Re-calculate Rotation (V3 Style)
print("Calculating Rotation...")
valid_anchors = []
X_list = []
Y_list = []
for anchor in anchors:
    h_word = anchor['hieroglyphic']
    e_word = anchor['english']
    if e_word in eng_model.wv:
        valid_anchors.append((h_word, e_word))
        X_list.append(hier_model.wv[h_word])
        Y_list.append(eng_model.wv[e_word])

X = np.array(X_list)
Y = np.array(Y_list)
U, S, Vt = np.linalg.svd(Y.T @ X)
R = U @ Vt

# Interesting Words to Probe
probe_words = {
    "nfr": "Good/Beautiful",
    "pr-aa": "Pharaoh (Great House)",
    "ra": "Sun God",
    "anubis": "Anubis", # Might not exist, let's check 'inpw'
    "inpw": "Anubis",
    "maat": "Truth/Order",
    "suten": "King",
    "hm-ntr": "Priest",
    "mw": "Water",
    "t": "Bread",
    "hqt": "Beer"
}

target_words = eng_model.wv.index_to_key[:20000]
target_vecs = np.array([eng_model.wv[w] for w in target_words])

def get_nn(h_word, k=5):
    if h_word not in hier_model.wv:
        return ["(Not in Vocab)"]
    vec = hier_model.wv[h_word]
    proj = vec @ R.T
    sims = np.dot(target_vecs, proj)
    indices = np.argsort(sims)[-k:][::-1]
    return [f"{target_words[i]} ({sims[i]:.2f})" for i in indices]

print("\n=== DISCOVERIES ===")
for h_word, meaning in probe_words.items():
    print(f"\nWord: {h_word} ({meaning})")
    matches = get_nn(h_word)
    for m in matches:
        print(f"  -> {m}")
