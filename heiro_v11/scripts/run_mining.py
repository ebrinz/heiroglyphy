"""
Script to execute v11 Phase 1.1: Anchor Mining
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def run_mining():
    # Paths
    PROJECT_ROOT = Path("heiro_v11")
    REPO_ROOT = Path(".")
    
    V5_DATA = REPO_ROOT / 'heiro_v5_getdata/data/processed'
    V10_DATA = REPO_ROOT / 'heiro_v10_refinement/data'
    V11_DATA = PROJECT_ROOT / 'data'
    
    print(f'Project Root: {PROJECT_ROOT}')
    
    candidates = {}
    
    # 1. TLA/BBAW Anchors
    anchors_path = V5_DATA / 'english_anchors.json'
    with open(anchors_path, 'r') as f:
        existing = json.load(f)
    
    for item in existing:
        egy = item['hieroglyphic']
        eng = item['english']
        candidates[egy] = {
            'hieroglyphic': egy,
            'current_translation': eng,
            'source': 'TLA_Existing'
        }
    
    print(f'Loaded {len(candidates)} existing English anchors')
    
    # 1.1 TLA German Anchors (to be translated)
    german_path = V5_DATA / 'german_anchors.json'
    if german_path.exists():
        with open(german_path, 'r') as f:
            german_anchors = json.load(f)
            
        german_count = 0
        for item in german_anchors:
            egy = item['hieroglyphic']
            # Some might have 'german' key, some 'english' (legacy issue)
            ger = item.get('german', item.get('english', ''))
            
            # If we already have this word from English anchors, skip?
            # No, let's keep it as a candidate source, maybe the German one is better or different.
            # Actually, if we have it in English, we prefer that, but we want to verify it's not German.
            # Let's just add it if it's not there, or update source.
            if egy not in candidates:
                candidates[egy] = {
                    'hieroglyphic': egy,
                    'current_translation': ger,
                    'source': 'TLA_German'
                }
                german_count += 1
        print(f'Added {german_count} anchors from TLA German')
    
    # 2. Mine HamdiJr Lexicon
    lexicon_path = V10_DATA / 'Lexicon.txt'
    lex_count = 0
    
    # Build Code -> Transliteration Map simultaneously
    code_to_trans = defaultdict(list)
    
    with open(lexicon_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 4:
                codes = parts[0].rstrip(',')
                trans = parts[1]
                eng = parts[2]
                
                code_to_trans[codes].append(trans)
                
                if trans not in candidates:
                    candidates[trans] = {
                        'hieroglyphic': trans,
                        'current_translation': eng,
                        'source': 'HamdiJr_Lexicon'
                    }
                    lex_count += 1
    
    print(f'Added {lex_count} new anchors from Lexicon')
    
    # 3. Mine N-Grams (Reverse Mapping)
    ngrams_path = V11_DATA / 'raw/nGrams.txt'
    ngram_count = 0
    
    if ngrams_path.exists():
        with open(ngrams_path, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    seq = parts[0].rstrip(',')
                    
                    if seq in code_to_trans:
                        for trans in code_to_trans[seq]:
                            if trans not in candidates:
                                candidates[trans] = {
                                    'hieroglyphic': trans,
                                    'current_translation': '', # Needs translation
                                    'source': 'HamdiJr_nGram'
                                }
                                ngram_count += 1
    
    print(f'Added {ngram_count} new anchors from n-grams')
    
    # 4. Export
    df = pd.DataFrame(list(candidates.values()))
    print(f'Total Candidates: {len(df)}')
    
    output_csv = V11_DATA / 'processed/anchors_to_translate.csv'
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    print(f'✓ Exported to {output_csv}')

if __name__ == "__main__":
    run_mining()
