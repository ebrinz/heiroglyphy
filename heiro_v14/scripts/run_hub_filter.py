#!/usr/bin/env python3
"""
V14 Hub-Filtering Experiments (B1/B2/B3)

Tests whether removing English stopwords from the alignment target
improves retrieval accuracy by eliminating the 82% hubness problem.

Run:
    python heiro_v14/scripts/run_hub_filter.py

Estimated time: ~10-20 minutes

Note: Uses pickle to load visual_embeddings_768d.pkl (project's own data).
"""

import json
import pickle  # required for project's .pkl visual embeddings
import re
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent.parent

print("Loading data...")
text_emb = KeyedVectors.load_word2vec_format(
    str(REPO / "heiro_v7_FastTextVisual/models/fasttext_v7.vec"), binary=False
)
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

print("Building fused embeddings...")
fused = {}
for w in text_emb.index_to_key:
    tv = text_emb[w]
    vv = None
    codes = trans_to_g.get(w)
    if codes:
        vecs = [vis_emb[c] for c in codes if c in vis_emb]
        if vecs:
            vv = np.mean(vecs, axis=0)
    if vv is None:
        vv = np.zeros(768)
    fused[w] = np.concatenate([tv, vv])

seed_pairs = [
    (a["hieroglyphic"], a["english"].lower())
    for a in anchors
    if a["hieroglyphic"] in fused and a["english"].lower() in glove
]
train_pairs, test_pairs = train_test_split(seed_pairs, test_size=0.2, random_state=42)

STOPWORDS = {
    "the", "of", "and", "to", "a", "in", "is", "it", "for", "on", "that", "with",
    "as", "at", "by", "from", "or", "an", "be", "this", "which", "not", "are", "was",
    "but", "have", "had", "has", "been", "were", "their", "its", "his", "her", "he",
    "she", "they", "we", "you", "i", "my", "your", "our", "me", "him", "them", "us",
    "so", "if", "do", "did", "does", "will", "would", "could", "should", "may", "can",
    "no", "all", "each", "every", "any", "some", "what", "who", "how", "when", "where",
    "than", "then", "also", "just", "more", "very", "most", "only", "own", "same",
    "other", "such", "into", "about", "up", "out", "over", "after", "between", "through",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    ".", ",", "!", "?", "-", "--", "...", "'s", "n't",
}

# Build filtered GloVe
filtered_words = [w for w in glove.index_to_key if w not in STOPWORDS]
filtered_indices = [glove.key_to_index[w] for w in filtered_words]
filtered_vectors = glove.vectors[filtered_indices].astype(np.float32)
filtered_norms = np.linalg.norm(filtered_vectors, axis=1, keepdims=True)
filtered_normed = filtered_vectors / np.maximum(filtered_norms, 1e-10)
print(f"Filtered GloVe: {len(filtered_words)} (removed {len(glove) - len(filtered_words)})")

test_content = [p for p in test_pairs if p[1] not in STOPWORDS]
n_valid = len(test_content)
n_skipped = len(test_pairs) - n_valid
print(f"Test: {n_valid} content pairs, {n_skipped} stopword pairs skipped")


def eval_filtered(model, test_content):
    X = np.array([fused[p[0]] for p in test_content])
    Y_pred = model.predict(X)
    correct = {1: 0, 5: 0, 10: 0}
    for i in range(len(Y_pred)):
        pv = Y_pred[i] / max(np.linalg.norm(Y_pred[i]), 1e-10)
        sims = filtered_normed @ pv.astype(np.float32)
        top = np.argpartition(sims, -10)[-10:]
        top = top[np.argsort(sims[top])[::-1]]
        words = [filtered_words[j] for j in top]
        tw = test_content[i][1]
        if tw == words[0]:
            correct[1] += 1
        if tw in words[:5]:
            correct[5] += 1
        if tw in words[:10]:
            correct[10] += 1
    n = len(test_content)
    return {
        "top1": round(correct[1] / n * 100, 2),
        "top5": round(correct[5] / n * 100, 2),
        "top10": round(correct[10] / n * 100, 2),
    }


# ── B1: Filtered retrieval only ──────────────────────────────────────────
print("\n" + "=" * 70)
print("B1: Filtered RETRIEVAL (training unchanged)")
print("=" * 70)
X_tr = np.array([fused[p[0]] for p in train_pairs])
Y_tr = np.array([glove[p[1]] for p in train_pairs])
m1 = Ridge(alpha=0.1)
m1.fit(X_tr, Y_tr)
b1 = eval_filtered(m1, test_content)
print(f"  Top-1: {b1['top1']}%  Top-5: {b1['top5']}%  Top-10: {b1['top10']}%")

# ── B2: Filtered training + retrieval ────────────────────────────────────
print("\n" + "=" * 70)
print("B2: Filtered TRAINING + RETRIEVAL")
print("=" * 70)
filtered_train = [(e, n) for e, n in train_pairs if n not in STOPWORDS]
print(f"  Training anchors: {len(train_pairs)} -> {len(filtered_train)}")
X_tr2 = np.array([fused[p[0]] for p in filtered_train])
Y_tr2 = np.array([glove[p[1]] for p in filtered_train])
m2 = Ridge(alpha=0.1)
m2.fit(X_tr2, Y_tr2)
b2 = eval_filtered(m2, test_content)
print(f"  Top-1: {b2['top1']}%  Top-5: {b2['top5']}%  Top-10: {b2['top10']}%")

# ── B3: Filtered + iterative Procrustes ──────────────────────────────────
print("\n" + "=" * 70)
print("B3: Filtered + ITERATIVE PROCRUSTES")
print("=" * 70)

egy_words = list(fused.keys())
egy_matrix = np.array([fused[w] for w in egy_words])

cur_train = list(filtered_train)
existing = set(cur_train)
b3_log = []

for it in range(6):
    print(f"\n  Iter {it} | Anchors: {len(cur_train)}")
    Xt = np.array([fused[p[0]] for p in cur_train])
    Yt = np.array([glove[p[1]] for p in cur_train])
    m3 = Ridge(alpha=0.1)
    m3.fit(Xt, Yt)

    metrics = eval_filtered(m3, test_content)
    print(f"  Top-1: {metrics['top1']}%  Top-5: {metrics['top5']}%  Top-10: {metrics['top10']}%")

    if it < 5:
        proj = m3.predict(egy_matrix)
        pn = np.linalg.norm(proj, axis=1, keepdims=True)
        proj_normed = (proj / np.maximum(pn, 1e-10)).astype(np.float32)

        bs = 5000
        egy_nn = np.zeros(len(egy_words), dtype=np.int32)
        egy_sim = np.zeros(len(egy_words), dtype=np.float32)
        for s in range(0, len(egy_words), bs):
            e = min(s + bs, len(egy_words))
            sm = proj_normed[s:e] @ filtered_normed.T
            egy_nn[s:e] = np.argmax(sm, axis=1)
            egy_sim[s:e] = np.max(sm, axis=1)

        cands = sorted(set(egy_nn.tolist()))
        en_nn = {}
        for s in range(0, len(cands), bs):
            e = min(s + bs, len(cands))
            bi = cands[s:e]
            sm = filtered_normed[bi] @ proj_normed.T
            best = np.argmax(sm, axis=1)
            for j, ei in enumerate(bi):
                en_nn[ei] = int(best[j])

        new = 0
        for ei in range(len(egy_words)):
            eni = int(egy_nn[ei])
            if egy_sim[ei] < 0.45:
                continue
            if en_nn.get(eni) == ei:
                pair = (egy_words[ei], filtered_words[eni])
                if pair not in existing and pair[1] not in STOPWORDS:
                    cur_train.append(pair)
                    existing.add(pair)
                    new += 1

        print(f"  New MNN: {new}")
        b3_log.append({**metrics, "anchors": len(cur_train), "new": new})
        if new == 0:
            print("  Converged.")
            break
    else:
        b3_log.append({**metrics, "anchors": len(cur_train)})

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"V13 baseline (all words):     Top-1: 31.57%  Top-5: 38.81%  Top-10: 42.61%")
print(f"B1 (filtered retrieval):      Top-1: {b1['top1']}%  Top-5: {b1['top5']}%  Top-10: {b1['top10']}%")
print(f"B2 (filtered train+retr):     Top-1: {b2['top1']}%  Top-5: {b2['top5']}%  Top-10: {b2['top10']}%")
if b3_log:
    best_b3 = max(b3_log, key=lambda x: x["top1"])
    print(f"B3 (filtered+iterative):      Top-1: {best_b3['top1']}%  Top-5: {best_b3['top5']}%  Top-10: {best_b3['top10']}%")
    for i, r in enumerate(b3_log):
        print(f"  Iter {i}: {r['anchors']} anchors -> {r['top1']}%  (new: {r.get('new', '-')})")
print(f"\nTested on {n_valid} content-word pairs ({n_skipped} stopword pairs excluded)")

out_path = REPO / "heiro_v14/results/hub_filter_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(
        {"b1": b1, "b2": b2, "b3": b3_log, "n_valid": n_valid, "n_skipped": n_skipped},
        f, indent=2,
    )
print(f"Saved to {out_path}")
