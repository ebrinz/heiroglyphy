"""
V9 Step 4: Ridge Alignment (matching V7 methodology)

Uses the same Ridge regression approach as V7, but properly exports
all the required output files.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.linear_model import Ridge
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


def main():
    print("=" * 70)
    print("V9: Ridge Regression Alignment (V7 methodology)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n[1/5] Loading Egyptian embeddings...")
    hiero_kv = KeyedVectors.load(str(FUSED_MODEL_PATH))
    print(f"  Loaded {len(hiero_kv)} Egyptian vectors (dim={hiero_kv.vector_size})")

    print(f"\n[2/5] Loading GloVe...")
    glove_kv = KeyedVectors.load_word2vec_format(str(GLOVE_PATH), binary=False, no_header=True)
    print(f"  Loaded {len(glove_kv)} English vectors (dim={glove_kv.vector_size})")

    print(f"\n[3/5] Loading anchors...")
    with open(ANCHORS_PATH, 'r') as f:
        anchors = json.load(f)
    print(f"  Loaded {len(anchors)} anchor pairs")

    # Prepare data
    print(f"\n[4/5] Preparing alignment data...")
    X_list = []
    Y_list = []
    valid_anchors = []

    for anchor in anchors:
        h_word = anchor['hieroglyphic']
        e_word = anchor['english'].lower()

        if h_word in hiero_kv and e_word in glove_kv:
            X_list.append(hiero_kv[h_word])
            Y_list.append(glove_kv[e_word])
            valid_anchors.append((h_word, e_word, anchor['confidence']))

    X = np.array(X_list)
    Y = np.array(Y_list)

    print(f"  Valid anchors: {len(X)} / {len(anchors)} ({100*len(X)/len(anchors):.1f}%)")

    # Split
    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Ridge alignment (same as V7)
    print(f"\n[5/5] Training Ridge alignment (alpha=1.0)...")
    aligner = Ridge(alpha=1.0)
    aligner.fit(X_train, Y_train)

    r2_train = aligner.score(X_train, Y_train)
    r2_test = aligner.score(X_test, Y_test)
    print(f"  R² Train: {r2_train:.4f}")
    print(f"  R² Test: {r2_test:.4f}")

    # Extract the transformation matrix
    W = aligner.coef_.T  # Ridge stores as (output_dim, input_dim), transpose to (input, output)
    bias = aligner.intercept_
    print(f"  Transformation: {W.shape}, bias: {bias.shape}")

    # Evaluate
    Y_pred_test = aligner.predict(X_test)

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
    print("  This should match V7's 29.10%")

    # Save results
    results = {
        "model": "V9 Ridge Alignment (V7 replication)",
        "test_samples": len(X_test),
        "valid_anchors": len(X),
        "total_anchors": len(anchors),
        "anchor_coverage": len(X) / len(anchors) * 100,
        "top1_accuracy": acc_top1,
        "top5_accuracy": acc_top5,
        "top10_accuracy": acc_top10,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    # Export
    print("\n" + "=" * 70)
    print("EXPORTING OUTPUT FILES")
    print("=" * 70)

    egyptian_words = list(hiero_kv.index_to_key)
    egyptian_vectors = np.array([hiero_kv[w] for w in egyptian_words])

    print(f"\n  Transforming {len(egyptian_words)} Egyptian vectors...")
    # Use the aligner's predict method to include bias
    aligned_vectors = aligner.predict(egyptian_vectors)

    # L2 normalize
    norms = np.linalg.norm(aligned_vectors, axis=1, keepdims=True)
    aligned_vectors = aligned_vectors / np.where(norms > 0, norms, 1)

    vocab = {word: idx for idx, word in enumerate(egyptian_words)}

    # Save vectors
    np.save(ALIGNED_VECTORS_PATH, aligned_vectors.astype(np.float32))
    print(f"  egyptian_aligned_vectors.npy: {ALIGNED_VECTORS_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    # Save vocab
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"  egyptian_aligned_vocab.pkl: {VOCAB_PATH.stat().st_size / 1024:.1f} KB")

    # Save transform (include bias as extra row)
    # Shape: (1537, 300) where last row is bias
    transform_with_bias = np.vstack([W, bias.reshape(1, -1)])
    np.save(TRANSFORM_PATH, transform_with_bias.astype(np.float32))
    print(f"  procrustes_transform.npy: {TRANSFORM_PATH.stat().st_size / 1024:.1f} KB")

    # Also save just the coefficient matrix for those who want it
    COEF_PATH = OUTPUT_DIR / "ridge_coefficients.npy"
    np.save(COEF_PATH, W.astype(np.float32))
    print(f"  ridge_coefficients.npy: {COEF_PATH.stat().st_size / 1024:.1f} KB")

    BIAS_PATH = OUTPUT_DIR / "ridge_bias.npy"
    np.save(BIAS_PATH, bias.astype(np.float32))
    print(f"  ridge_bias.npy: {BIAS_PATH.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nFiles in {OUTPUT_DIR}:")
    print(f"  1. egyptian_aligned_vectors.npy - All Egyptian words in English GloVe space")
    print(f"  2. egyptian_aligned_vocab.pkl   - word -> index mapping")
    print(f"  3. procrustes_transform.npy     - transformation matrix (with bias)")
    print(f"  4. ridge_coefficients.npy       - W matrix only (1536x300)")
    print(f"  5. ridge_bias.npy               - bias vector (300,)")


if __name__ == "__main__":
    main()
