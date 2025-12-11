"""
Script to execute v11 Phase 1: Data Aggregation & Cleaning
"""
import json
import gc
from pathlib import Path

def run_aggregation():
    # Paths
    PROJECT_ROOT = Path("heiro_v11")
    REPO_ROOT = Path(".")
    
    V5_DATA = REPO_ROOT / 'heiro_v5_getdata/data/processed'
    V10_DATA = REPO_ROOT / 'heiro_v10_refinement/data'
    V11_DATA = PROJECT_ROOT / 'data'
    
    print(f'Project Root: {PROJECT_ROOT}')
    
    # 1. Load TLA / BBAW / Ramses Corpus
    corpus_path = V5_DATA / 'hieroglyphic_corpus.json'
    print(f'Loading corpus from {corpus_path}...')
    
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    print(f'✓ Loaded {len(corpus)} entries')
    
    # 2. Load HamdiJr Data (Lexicon + nGrams)
    lexicon_path = V10_DATA / 'Lexicon.txt'
    lexicon_entries = []
    
    if lexicon_path.exists():
        with open(lexicon_path, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 4:
                    lexicon_entries.append({
                        'codes': parts[0].rstrip(','),
                        'transliteration': parts[1],
                        'english': parts[2],
                        'frequency': parts[3]
                    })
        print(f'✓ Loaded {len(lexicon_entries)} Lexicon entries')
    
    ngrams_path = V11_DATA / 'raw/nGrams.txt'
    ngrams = []
    
    if ngrams_path.exists():
        with open(ngrams_path, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    ngrams.append({
                        'codes': parts[0].rstrip(','),
                        'frequency': parts[1]
                    })
        print(f'✓ Loaded {len(ngrams)} n-grams')
    else:
        print('⚠ nGrams.txt not found')
        
    # 3. Clean Anchors
    anchors_path = V5_DATA / 'english_anchors.json'
    with open(anchors_path, 'r') as f:
        anchors = json.load(f)
    
    print(f'Loaded {len(anchors)} existing anchors')
    
    german_path = V5_DATA / 'german_anchors.json'
    german_words = set()
    if german_path.exists():
        with open(german_path, 'r') as f:
            german_data = json.load(f)
            for item in german_data:
                if 'german' in item:
                    german_words.add(item['german'].lower())
                elif 'english' in item: # Fallback if mixed
                    german_words.add(item['english'].lower())
        print(f'Loaded {len(german_words)} German words to exclude')
    
    clean_anchors = []
    rejected = []
    
    for anchor in anchors:
        eng = anchor['english'].lower()
        if eng in german_words:
            rejected.append((eng, 'German'))
            continue
        clean_anchors.append(anchor)
    
    print(f'✓ Retained {len(clean_anchors)} clean anchors')
    print(f'✗ Rejected {len(rejected)} anchors')
    
    # 4. Save v11 Dataset
    output_corpus = V11_DATA / 'processed/v11_corpus.json'
    output_anchors = V11_DATA / 'processed/v11_anchors.json'
    output_corpus.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_corpus, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    with open(output_anchors, 'w') as f:
        json.dump(clean_anchors, f, indent=2)
    
    print(f'✓ Saved v11 corpus to {output_corpus}')
    print(f'✓ Saved v11 anchors to {output_anchors}')

if __name__ == "__main__":
    run_aggregation()
