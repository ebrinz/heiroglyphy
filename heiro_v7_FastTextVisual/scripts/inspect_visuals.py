import pickle
from pathlib import Path
import numpy as np

# Define path
BASE_DIR = Path(__file__).resolve().parent.parent
VISUAL_EMBED_PATH = BASE_DIR.parent / "heiro_v6_BERT/data/processed/visual_embeddings_768d.pkl"

def inspect_pickle():
    if not VISUAL_EMBED_PATH.exists():
        print(f"Error: {VISUAL_EMBED_PATH} not found.")
        return

    print(f"Loading {VISUAL_EMBED_PATH}...")
    try:
        with open(VISUAL_EMBED_PATH, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {len(data)}")
            first_key = next(iter(data))
            print(f"Sample Key: {first_key}")
            print(f"Sample Value Type: {type(data[first_key])}")
            if hasattr(data[first_key], 'shape'):
                print(f"Sample Value Shape: {data[first_key].shape}")
        else:
            print("Data is not a dictionary.")
            
    except Exception as e:
        print(f"Failed to load pickle: {e}")

if __name__ == "__main__":
    inspect_pickle()
