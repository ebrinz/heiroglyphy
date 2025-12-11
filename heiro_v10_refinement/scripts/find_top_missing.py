import logging
import json
from pathlib import Path
from gensim.models import FastText
from collections import Counter

# Setup paths
PROJECT_ROOT = Path("heiro_v10_refinement")
REPO_ROOT = Path(".")

def find_top_missing():
    # Load V7 FastText model
    v7_model_path = REPO_ROOT / 'heiro_v7_FastTextVisual/models/fasttext_v7.model'
    fasttext_model = FastText.load(str(v7_model_path))
    
    # Get vocab with counts (if available, otherwise just order)
    # FastText.wv.key_to_index preserves frequency order (most frequent first)
    vocab = fasttext_model.wv.index_to_key
    
    # Load mapping
    mapping_path = PROJECT_ROOT / 'data/gardiner_mapping.json'
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
        
    # Create reverse mapping
    trans_to_code = {}
    for code, trans in mapping.items():
        for t in trans.split(','):
            t = t.strip()
            if t:
                trans_to_code[t] = code
                
    # Check top 100 words
    print("Top 100 Most Frequent Words Analysis:")
    print(f"{'Rank':<5} {'Word':<15} {'Status':<10} {'Mapping'}")
    print("-" * 50)
    
    missing_count = 0
    for i, word in enumerate(vocab[:100]):
        status = "MISSING"
        map_val = ""
        
        # Simple normalization check
        norm_word = word.replace('=', '').replace('[', '').replace(']', '').split('(')[0]
        
        if word in trans_to_code:
            status = "FOUND"
            map_val = trans_to_code[word]
        elif norm_word in trans_to_code:
            status = "NORM_FOUND"
            map_val = trans_to_code[norm_word]
        else:
            missing_count += 1
            
        print(f"{i+1:<5} {word:<15} {status:<10} {map_val}")
        
    print(f"\nMissing in Top 100: {missing_count}")

if __name__ == "__main__":
    find_top_missing()
