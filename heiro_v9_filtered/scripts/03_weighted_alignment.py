"""
V9 Step 3: Weighted Alignment

Instead of filtering anchors, weight them by confidence during alignment.
High-confidence anchors contribute more to the loss function.

This preserves quantity (needed for good alignment) while prioritizing quality.
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

FUSED_MODEL_PATH = DEV_DIR / "heiro_v7_FastTextVisual/models/fused_embeddings_1536d.kv"
GLOVE_PATH = DEV_DIR / "heiro_v5_getdata/data/processed/glove.6B.300d.txt"
ANCHORS_PATH = BASE_DIR.parent / "heiro_v6_BERT/data/processed/anchors.json"

OUTPUT_DIR = BASE_DIR / "outputs"
ALIGNED_VECTORS_PATH = OUTPUT_DIR / "egyptian_aligned_vectors.npy"
VOCAB_PATH = OUTPUT_DIR / "egyptian_aligned_vocab.pkl"
TRANSFORM_PATH = OUTPUT_DIR / "procrustes_transform.npy"
RESULTS_PATH = BASE_DIR / "data/processed/alignment_results_v9.json"


def weighted_lstsq(X, Y, weights):
    """
    Weighted least squares: minimize sum_i w_i * ||X_i @ W - Y_i||^2

    Solution: W = (X.T @ diag(w) @ X)^-1 @ X.T @ diag(w) @ Y
    """
    # sqrt(weights) because we square them in the objective
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    X_weighted = X * sqrt_w
    Y_weighted = Y * sqrt_w

    W, _, _, _ = np.linalg.lstsq(X_weighted, Y_weighted, rcond=None)
    return W


def main():
    print("=" * 70)
    print("V9: Confidence-Weighted Alignment")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n[1/5] Loading Egyptian embeddings...")
    hiero_kv = KeyedVectors.load(str(FUSED_MODEL_PATH))
    print(f"  Loaded {len(hiero_kv)} Egyptian vectors")

    print(f"\n[2/5] Loading GloVe...")
    glove_kv = KeyedVectors.load_word2vec_format(str(GLOVE_PATH), binary=False, no_header=True)
    print(f"  Loaded {len(glove_kv)} English vectors")

    print(f"\n[3/5] Loading ALL anchors (no filtering)...")
    with open(ANCHORS_PATH, 'r') as f:
        anchors = json.load(f)
    print(f"  Loaded {len(anchors)} anchor pairs")

    # Prepare data with confidence weights
    print(f"\n[4/5] Preparing weighted alignment data...")
    X_list = []
    Y_list = []
    weights = []
    valid_anchors = []

    for anchor in anchors:
        h_word = anchor['hieroglyphic']
        e_word = anchor['english'].lower()
        conf = anchor['confidence']

        if h_word in hiero_kv and e_word in glove_kv:
            X_list.append(hiero_kv[h_word])
            Y_list.append(glove_kv[e_word])
            # Use confidence^2 as weight (emphasize high-confidence more)
            weights.append(conf ** 2)
            valid_anchors.append((h_word, e_word, conf))

    X = np.array(X_list)
    Y = np.array(Y_list)
    weights = np.array(weights)

    print(f"  Valid anchors: {len(X)} / {len(anchors)}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  Avg weight: {weights.mean():.3f}")

    # Split (maintain weights)
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    w_train = weights[train_idx]
    anchors_test = [valid_anchors[i] for i in test_idx]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Learn weighted alignment
    print(f"\n[5/5] Learning weighted alignment...")
    W = weighted_lstsq(X_train, Y_train, w_train)
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
    print(f"    V7 (unweighted):        29.10%")
    print(f"    V9 strict filtering:    23.16%")
    print(f"    V9 hybrid filtering:    24.43%")
    print(f"    V9 weighted (this):     {acc_top1:.2f}%")

    # Save results
    results = {
        "model": "V9 Confidence-Weighted Alignment",
        "test_samples": len(X_test),
        "valid_anchors": len(X),
        "total_anchors": len(anchors),
        "top1_accuracy": acc_top1,
        "top5_accuracy": acc_top5,
        "top10_accuracy": acc_top10,
        "v7_comparison": 29.10,
        "improvement": acc_top1 - 29.10,
        "weighting": "confidence^2",
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    # Export
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

    np.save(ALIGNED_VECTORS_PATH, aligned_vectors.astype(np.float32))
    print(f"  egyptian_aligned_vectors.npy: {ALIGNED_VECTORS_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"  egyptian_aligned_vocab.pkl: {VOCAB_PATH.stat().st_size / 1024:.1f} KB")

    np.save(TRANSFORM_PATH, W.astype(np.float32))
    print(f"  procrustes_transform.npy: {TRANSFORM_PATH.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
