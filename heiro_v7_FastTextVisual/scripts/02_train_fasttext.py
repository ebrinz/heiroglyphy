import logging
from pathlib import Path
from gensim.models import FastText
from gensim.utils import simple_preprocess

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = BASE_DIR / "data/processed/cleaned_corpus.txt"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fasttext_v7.model"

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not CLEAN_DATA_PATH.exists():
        print(f"Error: {CLEAN_DATA_PATH} not found.")
        return

    print(f"Training FastText model on {CLEAN_DATA_PATH}...")
    
    # Load corpus
    class MyCorpus:
        def __iter__(self):
            with open(CLEAN_DATA_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    # Simple split is enough as we already tokenized in cleaning
                    yield line.split()

    sentences = MyCorpus()
    
    # Train FastText
    # Parameters:
    # vector_size=300: Standard size, matches GloVe
    # window=5: Context window
    # min_count=1: Keep all glyphs for now
    # sg=1: Skip-gram (usually better for smaller datasets)
    # epochs=10: Train for a bit longer
    model = FastText(vector_size=300, window=5, min_count=1, sentences=sentences, epochs=10, sg=1)
    
    # Save model
    model.save(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")
    
    # Save vectors in word2vec format for easy inspection
    model.wv.save_word2vec_format(str(MODEL_DIR / "fasttext_v7.vec"))
    print(f"Vectors saved to {MODEL_DIR / 'fasttext_v7.vec'}")
    
    # Test a few similarities
    print("\nVocabulary size:", len(model.wv))
    
    # Try to find similar glyphs to common ones if they exist in vocab
    # Common glyphs: G43 (w), M17 (i), D21 (r)
    test_glyphs = ["G43", "M17", "D21", "A1"]
    for glyph in test_glyphs:
        if glyph in model.wv:
            print(f"\nMost similar to {glyph}:")
            print(model.wv.most_similar(glyph, topn=5))
        else:
            print(f"\nGlyph {glyph} not in vocabulary.")

if __name__ == "__main__":
    main()
