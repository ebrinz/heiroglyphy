#!/usr/bin/env python3
"""
V15: Retrain FastText with better parameters.

V7 used: vector_size=300, window=5, min_count=1, sg=1, epochs=10
  -> 80,662 vocab, 65.5% hapax, sparse context from 7.8 avg words/line

This script tests multiple FastText configurations and evaluates each
through the full alignment pipeline (fuse -> Ridge(alpha=0.1) -> Top-1/5/10).

Key changes to test:
  - min_count=3/5: filter the 52K hapax noise words
  - window=10/15: capture more context from short texts (median 6 words)
  - vector_size=768: match V7 dimensionality
  - epochs=20/50: more passes over sparse data

Run:
    python heiro_v15/scripts/train_and_evaluate.py

Estimated time: ~20-30 minutes (trains multiple models)

Note: Pickle is used to load visual_embeddings_768d.pkl -- project's own data.
"""

import json
import pickle  # required for project's .pkl visual embeddings
import re
import time
import numpy as np
from pathlib import Path
from gensim.models import FastText, KeyedVectors
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO / "heiro_v7_FastTextVisual/data/processed/cleaned_corpus.txt"
MODEL_DIR = REPO / "heiro_v15/models"
RESULTS_PATH = REPO / "heiro_v15/results/retrain_results.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


class CorpusIterator:
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.split()


def load_alignment_data():
    """Load everything needed for alignment evaluation."""
    print("Loading alignment data...")

    with open(REPO / "heiro_v9_use_visuals_again/data/processed/visual_embeddings_768d.pkl", "rb") as f:
        vis_emb = pickle.load(f)

    with open(REPO / "heiro_v10_refinement/data/gardiner_mapping.json") as f:
        gmap = json.load(f)
    trans_to_g = {}
    for code, ts in gmap.items():
        for p in re.split(r"[,;]", ts):
            c = re.sub(r"\(.*?\)", "", p).strip()
            if c:
                trans_to_g.setdefault(c, []).append(code)

    with open(REPO / "heiro_v5_getdata/data/processed/english_anchors.json") as f:
        anchors = json.load(f)

    glove = KeyedVectors.load_word2vec_format(
        str(REPO / "heiro_v5_getdata/data/processed/glove.6B.300d.txt"),
        binary=False, no_header=True,
    )

    return vis_emb, trans_to_g, anchors, glove


def evaluate_model(text_kv, vis_emb, trans_to_g, anchors, glove, alpha=0.1):
    """Full pipeline: fuse -> Ridge -> evaluate Top-1/5/10."""
    vec_size = text_kv.vector_size

    # Build fused embeddings
    fused = {}
    vis_match = 0
    for word in text_kv.index_to_key:
        tv = text_kv[word]
        vv = None
        codes = trans_to_g.get(word)
        if codes:
            vecs = [vis_emb[c] for c in codes if c in vis_emb]
            if vecs:
                vv = np.mean(vecs, axis=0)
                vis_match += 1
        if vv is None:
            vv = np.zeros(768)
        fused[word] = np.concatenate([tv, vv])

    fused_dim = vec_size + 768

    # Build anchor arrays
    X, Y, pairs = [], [], []
    for a in anchors:
        egy, eng = a["hieroglyphic"], a["english"].lower()
        if egy in fused and eng in glove:
            X.append(fused[egy])
            Y.append(glove[eng])
            pairs.append((egy, eng))

    X, Y = np.array(X), np.array(Y)
    X_tr, X_te, Y_tr, Y_te, p_tr, p_te = train_test_split(
        X, Y, pairs, test_size=0.2, random_state=42
    )

    # Train Ridge
    model = Ridge(alpha=alpha)
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_te)

    # Evaluate
    correct = {1: 0, 5: 0, 10: 0}
    for i in range(len(Y_pred)):
        true_word = p_te[i][1]
        neighbors = glove.similar_by_vector(Y_pred[i], topn=10)
        words = [w for w, _ in neighbors]
        if true_word == words[0]:
            correct[1] += 1
        if true_word in words[:5]:
            correct[5] += 1
        if true_word in words[:10]:
            correct[10] += 1

    n = len(Y_pred)
    return {
        "top1": round(correct[1] / n * 100, 2),
        "top5": round(correct[5] / n * 100, 2),
        "top10": round(correct[10] / n * 100, 2),
        "vocab_size": len(text_kv),
        "valid_anchors": len(X),
        "train_anchors": len(X_tr),
        "fused_dim": fused_dim,
        "visual_matches": vis_match,
    }


def main():
    corpus = CorpusIterator(CORPUS)
    vis_emb, trans_to_g, anchors, glove = load_alignment_data()

    # Configurations to test
    configs = [
        # V7 baseline reproduction
        {"name": "V7_baseline",     "vector_size": 768, "window": 5,  "min_count": 1, "sg": 1, "epochs": 10},
        # Better min_count (filter noise)
        {"name": "mc3_w5",          "vector_size": 768, "window": 5,  "min_count": 3, "sg": 1, "epochs": 10},
        {"name": "mc5_w5",          "vector_size": 768, "window": 5,  "min_count": 5, "sg": 1, "epochs": 10},
        # Larger window (capture more context from short texts)
        {"name": "mc3_w10",         "vector_size": 768, "window": 10, "min_count": 3, "sg": 1, "epochs": 10},
        {"name": "mc3_w15",         "vector_size": 768, "window": 15, "min_count": 3, "sg": 1, "epochs": 10},
        {"name": "mc5_w10",         "vector_size": 768, "window": 10, "min_count": 5, "sg": 1, "epochs": 10},
        # More epochs (more passes over sparse data)
        {"name": "mc3_w10_e20",     "vector_size": 768, "window": 10, "min_count": 3, "sg": 1, "epochs": 20},
        {"name": "mc3_w10_e50",     "vector_size": 768, "window": 10, "min_count": 3, "sg": 1, "epochs": 50},
        # Best combo with CBOW (sometimes better for small corpora)
        {"name": "mc3_w10_cbow",    "vector_size": 768, "window": 10, "min_count": 3, "sg": 0, "epochs": 20},
    ]

    all_results = []

    print("=" * 70)
    print("V15: FastText Parameter Sweep")
    print(f"Corpus: {CORPUS} (100,729 lines, 789K tokens)")
    print("=" * 70)

    for i, cfg in enumerate(configs):
        name = cfg.pop("name")
        print(f"\n{'_' * 70}")
        print(f"[{i+1}/{len(configs)}] {name}: {cfg}")
        print(f"{'_' * 70}")

        t0 = time.time()

        # Train FastText
        print("  Training FastText...")
        model = FastText(sentences=corpus, **cfg)
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.1f}s -- vocab: {len(model.wv)}")

        # Save vectors
        vec_path = MODEL_DIR / f"fasttext_{name}.vec"
        model.wv.save_word2vec_format(str(vec_path))

        # Evaluate through alignment pipeline
        print("  Evaluating alignment...")
        t1 = time.time()
        metrics = evaluate_model(model.wv, vis_emb, trans_to_g, anchors, glove, alpha=0.1)
        eval_time = time.time() - t1

        delta = metrics["top1"] - 31.57
        print(f"\n  Top-1: {metrics['top1']}%  (delta vs V13: {delta:+.2f}%)")
        print(f"  Top-5: {metrics['top5']}%  Top-10: {metrics['top10']}%")
        print(f"  Vocab: {metrics['vocab_size']}, Anchors: {metrics['valid_anchors']}")
        print(f"  Time: train={train_time:.1f}s, eval={eval_time:.1f}s")

        result = {
            "name": name,
            "config": cfg,
            "train_time_s": round(train_time, 1),
            **metrics,
            "delta_v13": round(delta, 2),
        }
        all_results.append(result)
        cfg["name"] = name  # restore

        # Save incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump({"experiments": all_results}, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("V15 SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Name':<20s} {'Vocab':>7s} {'Top-1':>7s} {'Top-5':>7s} {'Top-10':>7s} {'d V13':>7s}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['name']:<20s} {r['vocab_size']:>7d} {r['top1']:>7.2f} {r['top5']:>7.2f} {r['top10']:>7.2f} {r['delta_v13']:>+7.2f}")

    best = max(all_results, key=lambda x: x["top1"])
    print(f"\nBest: {best['name']} -- {best['top1']}% Top-1 (delta {best['delta_v13']:+.2f}% vs V13)")
    print(f"   Config: {best['config']}")

    # Save final
    with open(RESULTS_PATH, "w") as f:
        json.dump({"experiments": all_results, "best": best}, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
