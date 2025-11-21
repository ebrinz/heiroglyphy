import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import FastText, KeyedVectors

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
FASTTEXT_MODEL_PATH = BASE_DIR / "models/fasttext_v7.model"
VISUAL_EMBED_PATH = BASE_DIR.parent / "heiro_v6_BERT/data/processed/visual_embeddings_768d.pkl"
LEXICON_PATH = BASE_DIR.parent / "heiro_v6_BERT/data/processed/hieroglyph_lexicon.csv"
FUSED_MODEL_PATH = BASE_DIR / "models/fused_embeddings_1068d.kv"

def main():
    # 1. Load FastText Model
    print(f"Loading FastText model from {FASTTEXT_MODEL_PATH}...")
    ft_model = FastText.load(str(FASTTEXT_MODEL_PATH))
    ft_wv = ft_model.wv
    print(f"FastText Vocab Size: {len(ft_wv)}")

    # 2. Load Visual Embeddings
    print(f"Loading Visual embeddings from {VISUAL_EMBED_PATH}...")
    with open(VISUAL_EMBED_PATH, 'rb') as f:
        visual_embeds = pickle.load(f)
    print(f"Visual Embeddings Size: {len(visual_embeds)}")

    # 3. Load Lexicon for Mapping
    print(f"Loading Lexicon from {LEXICON_PATH}...")
    lexicon_df = pd.read_csv(LEXICON_PATH)
    # Create map: lowercase_gardiner -> unicode
    # glyph_name is like 'a1', 'g43'
    gardiner_to_unicode = dict(zip(lexicon_df['glyph_name'], lexicon_df['unicode']))
    
    # 4. Fuse Embeddings
    print("Fusing embeddings...")
    fused_vectors = []
    words = []
    
    visual_dim = 768
    text_dim = 300
    
    matches = 0
    misses = 0
    
    for word in ft_wv.index_to_key:
        # Get text vector
        text_vec = ft_wv[word]
        
        # Get visual vector
        # Normalize word to match lexicon (lowercase)
        # FastText tokens might be 'G43', 'G43\', 'G43_'
        # We try to clean it up to find a match
        clean_word = word.lower().strip().replace('_', '').replace('\\', '')
        
        visual_vec = np.zeros(visual_dim, dtype=np.float32)
        
        if clean_word in gardiner_to_unicode:
            unicode_key = gardiner_to_unicode[clean_word]
            if unicode_key in visual_embeds:
                visual_vec = visual_embeds[unicode_key]
                matches += 1
            else:
                misses += 1
        else:
            misses += 1
            
        # Normalize vectors (L2)
        text_norm = np.linalg.norm(text_vec)
        if text_norm > 0:
            text_vec = text_vec / text_norm
            
        visual_norm = np.linalg.norm(visual_vec)
        if visual_norm > 0:
            visual_vec = visual_vec / visual_norm
            
        # Concatenate
        fused_vec = np.concatenate([text_vec, visual_vec])
        fused_vectors.append(fused_vec)
        words.append(word)
        
    print(f"Fusion Complete. Matches: {matches}, Misses: {misses}")
    print(f"Match Rate: {matches / len(words):.2%}")
    
    # 5. Save Fused Model
    fused_vectors = np.array(fused_vectors)
    print(f"Fused Vectors Shape: {fused_vectors.shape}")
    
    kv = KeyedVectors(vector_size=text_dim + visual_dim)
    kv.add_vectors(words, fused_vectors)
    
    kv.save(str(FUSED_MODEL_PATH))
    print(f"Saved fused model to {FUSED_MODEL_PATH}")

if __name__ == "__main__":
    main()
