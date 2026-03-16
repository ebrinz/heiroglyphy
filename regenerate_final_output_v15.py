#!/usr/bin/env python3
"""
Regenerate final_output vectors using V15 SOTA methodology.

V15 key changes vs V10:
- FastText trained with min_count=5, window=10 (sharper embeddings, 10,833 words)
- Ridge Regression alpha=0.001 (sharper, more discriminative vectors)
- Same visual fusion pipeline (1536d → 300d GloVe)

Usage:
    python regenerate_final_output_v15.py
"""

import json
import pickle  # noqa: S403 — project's own serialized vocab data
import re
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    REPO_ROOT = Path(__file__).parent

    print("=" * 60)
    print("REGENERATING FINAL OUTPUT WITH V15 SOTA METHODOLOGY")
    print("  FastText: min_count=5, window=10")
    print("  Ridge: alpha=0.001")
    print("=" * 60)

    # 1. Load V15 FastText vectors (768d)
    print("\n[1/8] Loading V15 FastText vectors...")
    v15_path = REPO_ROOT / 'heiro_v15/models/fasttext_mc5_w10.vec'
    text_embeddings = KeyedVectors.load_word2vec_format(str(v15_path), binary=False)
    print(f"  Loaded {len(text_embeddings)} text embeddings ({text_embeddings.vector_size}d)")

    # 2. Load V9 visual embeddings (768d)
    print("\n[2/8] Loading V9 visual embeddings...")
    visual_path = REPO_ROOT / 'heiro_v9_use_visuals_again/data/processed/visual_embeddings_768d.pkl'
    # Project's own serialized visual embeddings
    with open(visual_path, 'rb') as f:  # noqa: S301
        visual_embeddings = pickle.load(f)  # noqa: S301
    print(f"  Loaded {len(visual_embeddings)} visual embeddings (768d)")

    # 3. Load V10 Gardiner mapping
    print("\n[3/8] Loading V10 Gardiner mapping...")
    mapping_path = REPO_ROOT / 'heiro_v10_refinement/data/gardiner_mapping.json'
    with open(mapping_path, 'r', encoding='utf-8') as f:
        gardiner_mapping = json.load(f)

    trans_to_gardiner = {}
    for code, trans_str in gardiner_mapping.items():
        parts = re.split(r'[,;]', trans_str)
        for part in parts:
            clean_part = re.sub(r'\(.*?\)', '', part).strip()
            if clean_part:
                if clean_part not in trans_to_gardiner:
                    trans_to_gardiner[clean_part] = []
                trans_to_gardiner[clean_part].append(code)

    print(f"  Loaded {len(gardiner_mapping)} Gardiner codes")

    # 4. Create fused embeddings (1536d)
    print("\n[4/8] Creating fused embeddings (1536d)...")
    fused_embeddings = {}
    visual_match_count = 0

    for word in tqdm(text_embeddings.index_to_key, desc="  Fusing"):
        text_vec = text_embeddings[word]
        visual_vec = None

        codes = trans_to_gardiner.get(word)
        if codes:
            vectors = []
            for code in codes:
                if code in visual_embeddings:
                    vectors.append(visual_embeddings[code])
            if vectors:
                visual_vec = np.mean(vectors, axis=0)
                visual_match_count += 1

        if visual_vec is None:
            visual_vec = np.zeros(768)

        fused_vec = np.concatenate([text_vec, visual_vec])
        fused_embeddings[word] = fused_vec

    print(f"  Created {len(fused_embeddings)} fused embeddings")
    print(f"  Visual match rate: {visual_match_count/len(fused_embeddings)*100:.2f}%")

    # 5. Load anchors and GloVe
    print("\n[5/8] Loading anchors and GloVe...")
    anchors_path = REPO_ROOT / 'heiro_v5_getdata/data/processed/english_anchors.json'
    with open(anchors_path, 'r', encoding='utf-8') as f:
        anchors = json.load(f)
    print(f"  Loaded {len(anchors)} anchor pairs")

    glove_path = REPO_ROOT / 'heiro_v5_getdata/data/processed/glove.6B.300d.txt'
    print(f"  Loading GloVe (this takes ~1 minute)...")
    english_embeddings = KeyedVectors.load_word2vec_format(str(glove_path), binary=False, no_header=True)
    print(f"  Loaded {len(english_embeddings)} English embeddings (300d)")

    # 6. Prepare training data
    print("\n[6/8] Preparing training data...")
    X = []
    Y = []
    valid_anchors = []

    for anchor in anchors:
        egy_word = anchor['hieroglyphic']
        eng_word = anchor['english'].lower()

        if egy_word in fused_embeddings and eng_word in english_embeddings:
            X.append(fused_embeddings[egy_word])
            Y.append(english_embeddings[eng_word])
            valid_anchors.append((egy_word, eng_word))

    X = np.array(X)
    Y = np.array(Y)
    print(f"  Valid anchors: {len(X)} / {len(anchors)} ({len(X)/len(anchors)*100:.1f}%)")

    # 7. Train Ridge Regression (alpha=0.001) and evaluate
    print("\n[7/8] Training alignment model (alpha=0.001)...")
    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=0.2, random_state=42
    )

    aligner = Ridge(alpha=0.001)
    aligner.fit(X_train, Y_train)

    # Evaluate
    correct_top1 = correct_top5 = correct_top10 = 0
    Y_pred = aligner.predict(X_test)

    for i in range(len(X_test)):
        pred_vec = Y_pred[i]
        true_word = anchors_test[i][1]
        neighbors = english_embeddings.similar_by_vector(pred_vec, topn=10)
        neighbor_words = [w for w, s in neighbors]

        if true_word == neighbor_words[0]:
            correct_top1 += 1
        if true_word in neighbor_words[:5]:
            correct_top5 += 1
        if true_word in neighbor_words[:10]:
            correct_top10 += 1

    acc_top1 = correct_top1 / len(X_test) * 100
    acc_top5 = correct_top5 / len(X_test) * 100
    acc_top10 = correct_top10 / len(X_test) * 100

    print(f"  Top-1 Accuracy: {acc_top1:.2f}%")
    print(f"  Top-5 Accuracy: {acc_top5:.2f}%")
    print(f"  Top-10 Accuracy: {acc_top10:.2f}%")

    # 8. Project all Egyptian vectors and save
    print("\n[8/8] Projecting all vectors and saving...")

    vocab = list(fused_embeddings.keys())
    all_fused = np.array([fused_embeddings[w] for w in vocab])

    aligned_vectors = aligner.predict(all_fused)

    # L2 normalize
    norms = np.linalg.norm(aligned_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    aligned_vectors = aligned_vectors / norms

    # Convert to float16 for compression
    aligned_vectors_f16 = aligned_vectors.astype(np.float16)

    # Save
    output_dir = REPO_ROOT / 'final_output'
    output_dir.mkdir(exist_ok=True)

    np.savez_compressed(
        output_dir / 'egyptian_aligned_vectors.npz',
        vectors=aligned_vectors_f16
    )

    vocab_dict = {word: i for i, word in enumerate(vocab)}
    with open(output_dir / 'egyptian_aligned_vocab.pkl', 'wb') as f:
        pickle.dump(vocab_dict, f)

    print(f"  Saved {len(vocab)} vectors to final_output/")
    print(f"  Vector shape: {aligned_vectors_f16.shape}")

    # Also update concept vectors to include 'silence' and other needed words
    print("\n  Updating concept vectors...")
    cat_path = output_dir / 'concept_categories.json'
    with open(cat_path, 'r') as f:
        categories = json.load(f)

    concept_words = []
    concept_vecs = []
    for cat, words in categories.items():
        for w in words:
            if w in english_embeddings and w not in concept_words:
                concept_words.append(w)
                concept_vecs.append(english_embeddings[w])

    for w in ["silence", "quiet", "voice", "sound", "noise"]:
        if w in english_embeddings and w not in concept_words:
            concept_words.append(w)
            concept_vecs.append(english_embeddings[w])

    concept_array = np.array(concept_vecs, dtype=np.float32)
    np.savez_compressed(
        output_dir / 'concept_vectors.npz',
        vectors=concept_array,
        words=np.array(concept_words)
    )
    print(f"  Saved {len(concept_words)} concept vectors (including 'silence')")

    # Save metadata
    metadata = {
        'methodology': 'V15 SOTA (1536d fused -> 300d GloVe)',
        'text_embeddings': '768d FastText V15 (min_count=5, window=10)',
        'visual_embeddings': '768d ResNet-50 (V9)',
        'alignment': 'Ridge Regression (alpha=0.001)',
        'accuracy': {
            'top1': acc_top1,
            'top5': acc_top5,
            'top10': acc_top10
        },
        'vocab_size': len(vocab),
        'vector_dim': 300,
        'anchor_coverage': f"{len(X)}/{len(anchors)} ({len(X)/len(anchors)*100:.1f}%)",
        'visual_match_rate': f"{visual_match_count/len(fused_embeddings)*100:.2f}%"
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE! Final output regenerated with V15 SOTA methodology")
    print(f"  Accuracy: {acc_top1:.2f}% Top-1")
    print(f"  Vocabulary: {len(vocab)} words")
    print("=" * 60)

if __name__ == '__main__':
    main()
