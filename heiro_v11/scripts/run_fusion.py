"""
Script to execute v11 Phase 2: Visual Fusion with N-Grams
"""
import pickle
import numpy as np
from pathlib import Path
from gensim.models import FastText
from tqdm.auto import tqdm
from collections import defaultdict
import gc

import re

def normalize_word(word):
    """Normalize transliteration by removing markers."""
    variants = []
    original = word
    
    # Remove suffix markers (=f -> f)
    if word.startswith('='):
        variants.append(word[1:])
    
    # Remove parentheses
    if '(' in word:
        cleaned = re.sub(r'\([^)]*\)', '', word)
        if cleaned and cleaned != word:
            variants.append(cleaned)
            
    # Remove brackets
    if '[' in word or ']' in word:
        cleaned = word.replace('[', '').replace(']', '')
        if cleaned and cleaned != word:
            variants.append(cleaned)
            
    # Remove inflection markers (.n -> n)
    if '.' in word:
        parts = word.split('.')
        if parts[0] and parts[0] != word:
            variants.append(parts[0])
            
    if original not in variants:
        variants.append(original)
        
    return variants

def run_fusion():
    # Paths
    PROJECT_ROOT = Path("heiro_v11")
    REPO_ROOT = Path(".")
    
    V11_DATA = PROJECT_ROOT / 'data'
    V7_MODEL = REPO_ROOT / 'heiro_v7_FastTextVisual/models/fasttext_v7.model'
    V9_VISUAL = REPO_ROOT / 'heiro_v9_use_visuals_again/data/processed/visual_embeddings_768d.pkl'
    
    print(f'Project Root: {PROJECT_ROOT}')
    
    # 1. Load Data & Models
    print('Loading FastText model...')
    fasttext_model = FastText.load(str(V7_MODEL))
    text_embeddings = fasttext_model.wv
    print(f'✓ Loaded {len(text_embeddings)} text embeddings')
    
    print('Loading visual embeddings...')
    with open(V9_VISUAL, 'rb') as f:
        visual_embeddings = pickle.load(f)
    print(f'✓ Loaded {len(visual_embeddings)} visual embeddings')
    
    # Load Lexicon
    lexicon_path = REPO_ROOT / 'heiro_v10_refinement/data/Lexicon.txt'
    trans_to_codes = defaultdict(list)
    
    with open(lexicon_path, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 4:
                codes = parts[0].rstrip(',').split(',')
                trans = parts[1]
                freq = float(parts[3]) if parts[3] else 0.0
                if trans and codes:
                    trans_to_codes[trans].append({'codes': codes, 'freq': freq})
    
    print(f'✓ Loaded mappings for {len(trans_to_codes)} words')
    
    # Load nGrams
    ngrams_path = V11_DATA / 'raw/nGrams.txt'
    ngram_freqs = {}
    if ngrams_path.exists():
        with open(ngrams_path, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    seq = parts[0].rstrip(',')
                    freq = float(parts[1]) if parts[1] else 0.0
                    ngram_freqs[seq] = freq
        print(f'✓ Loaded {len(ngram_freqs)} n-gram frequencies')
    
    # 2. Generate Fused Embeddings
    fused_embeddings = {}
    visual_match_count = 0
    
    print('Generating fused embeddings...')
    
    for word in tqdm(text_embeddings.index_to_key, desc='Fusing'):
        text_vec = text_embeddings[word]
        visual_vec = None
        
        # Try normalized variants
        variants = normalize_word(word)
        candidates = []
        
        for variant in variants:
            if variant in trans_to_codes:
                candidates.extend(trans_to_codes[variant])
        
        if candidates:
            best_codes = None
            best_score = -1
            
            if len(candidates) == 1:
                best_codes = candidates[0]['codes']
            else:
                for cand in candidates:
                    seq_str = ','.join(cand['codes'])
                    score = ngram_freqs.get(seq_str, 0) * 1000 + cand['freq']
                    if score > best_score:
                        best_score = score
                        best_codes = cand['codes']
            
            if best_codes:
                vectors = []
                for code in best_codes:
                    if code in visual_embeddings:
                        vectors.append(visual_embeddings[code])
                
                if vectors:
                    visual_vec = np.mean(vectors, axis=0)
                    visual_match_count += 1
        
        if visual_vec is None:
            visual_vec = np.zeros(768)
        
        fused_embeddings[word] = np.concatenate([text_vec, visual_vec])
        
        # Store just the visual part for advanced training
        if visual_match_count > 0 and np.any(visual_vec):
             # Only store if we actually found a visual match to save space/time? 
             # Actually, for the model we need a vector for every word. 
             # But we can just store the map for words that have visuals.
             pass

    # Save Fused
    output_path = V11_DATA / 'processed/fused_embeddings_v11.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(fused_embeddings, f)
    
    print(f'✓ Saved fused embeddings to {output_path}')

    # Save Word->Visual Map (for Weighted Fusion)
    # We need to reconstruct it or save it during the loop. 
    # Let's just save the whole fused dictionary, we can split it later.
    # Fused is [Text, Visual]. Text is 768, Visual is 768.
    # We can split in the training script: fused[:768], fused[768:]
    # So we don't need to save separately.


if __name__ == "__main__":
    run_fusion()
