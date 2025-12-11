import logging
import json
import pickle
from pathlib import Path
from gensim.models import FastText

# Setup paths
PROJECT_ROOT = Path("heiro_v10_refinement")
REPO_ROOT = Path(".")

def analyze_vocab():
    # Load V7 FastText model
    v7_model_path = REPO_ROOT / 'heiro_v7_FastTextVisual/models/fasttext_v7.model'
    print(f'Loading V7 FastText model from {v7_model_path}...')
    fasttext_model = FastText.load(str(v7_model_path))
    vocab = fasttext_model.wv.index_to_key
    print(f'Vocab size: {len(vocab)}')
    
    # Sample vocab
    print("\nSample Vocabulary (First 50):")
    print(vocab[:50])
    
    # Load our current mapping
    mapping_path = PROJECT_ROOT / 'data/gardiner_mapping.json'
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        print(f"\nCurrent Mapping Size: {len(mapping)}")
        
        # Create reverse mapping (Trans -> Code)
        trans_to_code = {}
        for code, trans in mapping.items():
            for t in trans.split(','):
                t = t.strip()
                if t:
                    trans_to_code[t] = code
                    
        # Check overlap
        matches = 0
        for word in vocab:
            if word in trans_to_code:
                matches += 1
        
        print(f"\nDirect Matches: {matches} / {len(vocab)} ({matches/len(vocab)*100:.2f}%)")
        
        # Check for potential matches (e.g. substring)
        print("\nChecking for potential matches (first 20 vocab words):")
        for word in vocab[:20]:
            match = "NO"
            if word in trans_to_code:
                match = f"YES -> {trans_to_code[word]}"
            print(f"  '{word}': {match}")

if __name__ == "__main__":
    analyze_vocab()
