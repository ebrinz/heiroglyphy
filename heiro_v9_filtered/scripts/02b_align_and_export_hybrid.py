"""
V9 Step 2b: Align with Hybrid-Filtered Anchors

Uses hybrid-filtered anchors (more balanced) to learn alignment.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEV_DIR = Path("/Users/crashy/Development/heiroglyphy")

# Input paths
FUSED_MODEL_PATH = DEV_DIR / "heiro_v7_FastTextVisual/models/fused_embeddings_1536d.kv"
GLOVE_PATH = DEV_DIR / "heiro_v5_getdata/data/processed/glove.6B.300d.txt"
FILTERED_ANCHORS_PATH = BASE_DIR / "data/processed/filtered_anchors_hybrid.json"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
ALIGNED_VECTORS_PATH = OUTPUT_DIR / "egyptian_aligned_vectors.npy"
VOCAB_PATH = OUTPUT_DIR / "egyptian_aligned_vocab.pkl"
TRANSFORM_PATH = OUTPUT_DIR / "procrustes_transform.npy"
RESULTS_PATH = BASE_DIR / "data/processed/alignment_results_v9.json"


def main():
    print("=" * 70)
    print("V9: Hybrid-Filtered Anchor Alignment")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Egyptian embeddings
    print(f"\n[1/5] Loading Egyptian embeddings...")
    hiero_kv = KeyedVectors.load(str(FUSED_MODEL_PATH))
    print(f"  Loaded {len(hiero_kv)} Egyptian vectors (dim={hiero_kv.vector_size})")

    # 2. Load GloVe
    print(f"\n[2/5] Loading GloVe...")
    glove_kv = KeyedVectors.load_word2vec_format(str(GLOVE_PATH), binary=False, no_header=True)
    print(f"  Loaded {len(glove_kv)} English vectors (dim={glove_kv.vector_size})")

    # 3. Load filtered anchors
    print(f"\n[3/5] Loading hybrid-filtered anchors...")
    with open(FILTERED_ANCHORS_PATH, 'r') as f:
        anchors = json.load(f)
    print(f"  Loaded {len(anchors)} filtered anchor pairs")

    # 4. Prepare alignment data
    print(f"\n[4/5] Preparing alignment data...")
    X_anchors = []
    Y_anchors = []
    valid_anchors = []

    for anchor in anchors:
        h_word = anchor['hieroglyphic']
        e_word = anchor['english'].lower()

        if h_word in hiero_kv and e_word in glove_kv:
            X_anchors.append(hiero_kv[h_word])
            Y_anchors.append(glove_kv[e_word])
            valid_anchors.append((h_word, e_word, anchor['confidence']))

    X_anchors = np.array(X_anchors)
    Y_anchors = np.array(Y_anchors)

    print(f"  Valid anchors: {len(X_anchors)} / {len(anchors)} ({100*len(X_anchors)/len(anchors):.1f}%)")

    # Split for evaluation
    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X_anchors, Y_anchors, valid_anchors, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 5. Learn alignment
    print(f"\n[5/5] Learning alignment...")
    W, residuals, rank, s = np.linalg.lstsq(X_train, Y_train, rcond=None)
    print(f"  Transformation matrix: {W.shape}")

    # Evaluate
    Y_pred_test = X_test @ W

    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0

    print("  Evaluating...")
    for i in range(len(X_test)):
        pred_vec = Y_pred_test[i]
        true_word = anchors_test[i][1]

        neighbors = glove_kv.similar_by_vector(pred_vec, topn=10)
        neighbor_words = [w for w, _ in neighbors]

        if true_word == neighbor_words[0]:
            correct_top1 += 1
        if true_word in neighbor_words[:5]:
            correct_top5 += 1
        if true_word in neighbor_words[:10]:
            correct_top10 += 1

    acc_top1 = 100 * correct_top1 / len(X_test)
    acc_top5 = 100 * correct_top5 / len(X_test)
    acc_top10 = 100 * correct_top10 / len(X_test)

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Test samples: {len(X_test)}")
    print(f"  Top-1 Accuracy:  {acc_top1:.2f}%")
    print(f"  Top-5 Accuracy:  {acc_top5:.2f}%")
    print(f"  Top-10 Accuracy: {acc_top10:.2f}%")
    print()
    print("  Comparison:")
    print(f"    V7 (all anchors):         29.10%")
    print(f"    V9 strict (content only): 23.16%")
    print(f"    V9 hybrid (this):         {acc_top1:.2f}%")

    # Save results
    results = {
        "model": "V9 Hybrid-Filtered Anchors",
        "test_samples": len(X_test),
        "valid_anchors": len(X_anchors),
        "total_filtered_anchors": len(anchors),
        "anchor_coverage": len(X_anchors) / len(anchors) * 100,
        "top1_accuracy": acc_top1,
        "top5_accuracy": acc_top5,
        "top10_accuracy": acc_top10,
        "v7_comparison": 29.10,
        "improvement_over_v7": acc_top1 - 29.10,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    # Export all Egyptian vectors
    print("\n" + "=" * 70)
    print("EXPORTING OUTPUT FILES")
    print("=" * 70)

    egyptian_words = list(hiero_kv.index_to_key)
    egyptian_vectors = np.array([hiero_kv[w] for w in egyptian_words])

    print(f"\n  Transforming {len(egyptian_words)} Egyptian vectors...")
    aligned_vectors = egyptian_vectors @ W

    # L2 normalize
    norms = np.linalg.norm(aligned_vectors, axis=1, keepdims=True)
    aligned_vectors = aligned_vectors / np.where(norms > 0, norms, 1)

    vocab = {word: idx for idx, word in enumerate(egyptian_words)}

    # Save
    print(f"\n  Saving egyptian_aligned_vectors.npy...")
    np.save(ALIGNED_VECTORS_PATH, aligned_vectors.astype(np.float32))
    print(f"    -> Size: {ALIGNED_VECTORS_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n  Saving egyptian_aligned_vocab.pkl...")
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"    -> Size: {VOCAB_PATH.stat().st_size / 1024:.1f} KB")

    print(f"\n  Saving procrustes_transform.npy...")
    np.save(TRANSFORM_PATH, W.astype(np.float32))
    print(f"    -> Size: {TRANSFORM_PATH.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
