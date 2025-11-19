import os
import pickle
import logging
from gensim.models import FastText, Word2Vec

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Configuration
DATA_DIR = "heiro_v3/data"
MODELS_DIR = "heiro_v3/models"
CLEAN_CORPUS_FILE = os.path.join(DATA_DIR, "clean_corpora.pkl")
HIEROGLYPHIC_MODEL_FILE = os.path.join(MODELS_DIR, "hieroglyphic_fasttext.model")
ENGLISH_MODEL_FILE = os.path.join(MODELS_DIR, "english_word2vec.model")

# Hyperparameters
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 50

print(f"Loading corpora from {CLEAN_CORPUS_FILE}...")
with open(CLEAN_CORPUS_FILE, 'rb') as f:
    corpora = pickle.load(f)

hier_sentences = [s.split() for s in corpora['hieroglyphic']]
eng_sentences = [s.split() for s in corpora['english']]

print(f"Loaded {len(hier_sentences)} hieroglyphic sentences and {len(eng_sentences)} English sentences.")

print("Training Hieroglyphic FastText model...")
hier_model = FastText(
    sentences=hier_sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=1,  # Skip-gram
    epochs=EPOCHS,
    seed=42
)

print("Saving Hieroglyphic model...")
hier_model.save(HIEROGLYPHIC_MODEL_FILE)

print("Training English Word2Vec model...")
eng_model = Word2Vec(
    sentences=eng_sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=1,  # Skip-gram
    epochs=EPOCHS,
    seed=42
)

print("Saving English model...")
eng_model.save(ENGLISH_MODEL_FILE)

def check_similarity(model, word, title):
    print(f"\n--- {title}: '{word}' ---")
    try:
        similar = model.wv.most_similar(word, topn=5)
        for w, score in similar:
            print(f"{w}: {score:.3f}")
    except KeyError:
        print(f"Word '{word}' not in vocabulary.")

# Check Hieroglyphic
check_similarity(hier_model, 'nfr', "Hieroglyphic")
check_similarity(hier_model, 'ra', "Hieroglyphic")

# Check English
check_similarity(eng_model, 'god', "English")
check_similarity(eng_model, 'king', "English")
