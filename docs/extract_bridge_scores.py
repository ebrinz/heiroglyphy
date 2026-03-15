"""
Extract alignment scores and midpoint scores for each discovery term.

Alignment score: cosine similarity between aligned Egyptian word and its
                 English translation in the shared GloVe space.
Midpoint score:  cosine similarity between an English midpoint query vector
                 and the nearest Egyptian result.

The concept vocabulary (279 curated words) is augmented at runtime with
any GloVe words needed for midpoint queries (e.g. "silence") so that
every discovery can be computed faithfully.

Usage:
    python docs/extract_bridge_scores.py

Output: docs/bridge_scores.json

Note: Uses pickle to load the project's own serialized vocabulary mapping
(egyptian_aligned_vocab.pkl), not untrusted external data.
"""

import json
import pickle  # required for project's own .pkl vocab file  # noqa: S403
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINAL = REPO / "final_output"
OUT = REPO / "docs" / "bridge_scores.json"
GLOVE_PATH = REPO / "heiro_v5_getdata" / "data" / "processed" / "glove.6B.300d.txt"


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── Load embeddings ──────────────────────────────────────────────────────────
print("Loading Egyptian aligned vectors...")
eg_data = np.load(FINAL / "egyptian_aligned_vectors.npz")
eg_vectors = eg_data["vectors"].astype(np.float32)
# Ensure Egyptian vectors are unit-normalized for dot-product = cosine sim
eg_norms = np.linalg.norm(eg_vectors, axis=1, keepdims=True)
eg_norms[eg_norms == 0] = 1
eg_vectors = eg_vectors / eg_norms

# Load vocabulary mapping (project's own serialized data)
with open(FINAL / "egyptian_aligned_vocab.pkl", "rb") as f:  # noqa: S301
    eg_vocab = pickle.load(f)  # noqa: S301
eg_idx = eg_vocab if isinstance(eg_vocab, dict) else {w: i for i, w in enumerate(eg_vocab)}
eg_idx_to_word = {v: k for k, v in eg_idx.items()}

print("Loading concept vectors...")
concept_data = np.load(FINAL / "concept_vectors.npz", allow_pickle=True)
en_vectors_raw = concept_data["vectors"].astype(np.float32)
en_words = list(concept_data["words"])
en_idx = {w: i for i, w in enumerate(en_words)}


# ── Augment concept vocabulary from full GloVe ───────────────────────────────
EXTRA_WORDS = ["silence", "quiet", "voice", "sound", "noise"]

def load_glove_words(words_needed):
    """Load specific word vectors from full GloVe file."""
    found = {}
    words_set = set(words_needed)
    print(f"Loading {len(words_set)} extra words from GloVe...")
    with open(GLOVE_PATH, "r") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in words_set:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                found[word] = vec
                words_set.discard(word)
                if not words_set:
                    break
    if words_set:
        print(f"  Warning: not found in GloVe: {words_set}")
    return found

extra_needed = [w for w in EXTRA_WORDS if w not in en_idx]
if extra_needed and GLOVE_PATH.exists():
    extra_vecs = load_glove_words(extra_needed)
    for word, vec in extra_vecs.items():
        en_words.append(word)
        en_idx[word] = len(en_words) - 1
        en_vectors_raw = np.vstack([en_vectors_raw, vec.reshape(1, -1)])
    print(f"  Added {len(extra_vecs)} words: {list(extra_vecs.keys())}")
    print(f"  Concept vocabulary: {len(en_words)} words")

# Normalize all concept vectors to unit length
en_norms = np.linalg.norm(en_vectors_raw, axis=1, keepdims=True)
en_norms[en_norms == 0] = 1
en_vectors = en_vectors_raw / en_norms


def get_eg(word):
    idx = eg_idx.get(word)
    if idx is not None:
        return eg_vectors[idx]
    return None


def get_en(word):
    idx = en_idx.get(word)
    if idx is not None:
        return en_vectors[idx]
    return None


def midpoint(word_a, word_b):
    """Compute normalized midpoint of two English concept vectors."""
    va, vb = get_en(word_a), get_en(word_b)
    if va is None or vb is None:
        missing = []
        if va is None:
            missing.append(word_a)
        if vb is None:
            missing.append(word_b)
        return None, missing
    mid = normalize((va + vb) / 2)
    return mid, None


def nearest_egyptian(query_vec, top_k=20):
    """Find nearest Egyptian words to a query vector (cosine similarity)."""
    query_vec = normalize(query_vec)
    sims = eg_vectors @ query_vec
    top = np.argsort(sims)[-top_k:][::-1]
    return [(eg_idx_to_word[i], float(sims[i])) for i in top]


def alignment_score(eg_word, en_word):
    """Cosine similarity between aligned Egyptian word and English concept."""
    ev, nv = get_eg(eg_word), get_en(en_word)
    if ev is None or nv is None:
        return None
    return float(np.dot(ev, nv))  # both unit-normalized, dot = cosine


def find_term_rank(results, target_roots):
    """Find the rank (1-indexed) of a target term in results."""
    for i, (word, score) in enumerate(results):
        for root in target_roots:
            if root in word:
                return i + 1, word, score
    return None, None, None


# ── Compute discovery scores ────────────────────────────────────────────────
print("\nComputing discovery scores...")
discoveries = {}


# D1: Gold Is Divine Flesh
print("  D1: Gold Is Divine Flesh")
mid, missing = midpoint("gold", "divine")
if mid is not None:
    nearest = nearest_egyptian(mid, 20)
    discoveries["D1_Gold"] = {
        "title": "Gold Is Divine Flesh",
        "primary_term": "nṯr",
        "literal": "god, divine being",
        "alignment_score": alignment_score("nṯr", "god"),
        "midpoint_query": ["gold", "divine"],
        "midpoint_results": nearest[:10],
        "midpoint_score": nearest[0][1] if nearest else None,
        "secondary": {
            "term": "nbw",
            "literal": "gold",
            "alignment_score": alignment_score("nbw", "gold"),
        }
    }


# D2: Silence Is the Condition of the Dead
print("  D2: Silence Is the Condition of the Dead")
mid, missing = midpoint("silence", "death")
if mid is not None:
    nearest = nearest_egyptian(mid, 20)
    mwt_rank, mwt_word, mwt_score = find_term_rank(nearest, ["mwt", "m(w)t", "mw,t"])
    discoveries["D2_Silence"] = {
        "title": "Silence Is the Condition of the Dead",
        "primary_term": "mwt",
        "literal": "to die, dead",
        "alignment_score": alignment_score("mwt", "death"),
        "midpoint_query": ["silence", "death"],
        "midpoint_results": nearest[:10],
        "midpoint_score": nearest[0][1] if nearest else None,
        "mwt_rank": mwt_rank,
        "mwt_match": mwt_word,
        "mwt_score": mwt_score,
    }
else:
    discoveries["D2_Silence"] = {
        "title": "Silence Is the Condition of the Dead",
        "primary_term": "mwt",
        "literal": "to die, dead",
        "alignment_score": alignment_score("mwt", "death"),
        "midpoint_query": ["silence", "death"],
        "midpoint_results": [],
        "midpoint_score": None,
        "note": f"Missing concept words: {missing}"
    }


# D3: Seeing Was an Act of Magical Power
print("  D3: Seeing Was an Act of Magical Power")
mid, missing = midpoint("eye", "knowledge")
if mid is not None:
    nearest = nearest_egyptian(mid, 20)
    hka_rank, hka_word, hka_score = find_term_rank(nearest, ["ḥkꜣ", "ḥk"])
    discoveries["D3_Seeing"] = {
        "title": "Seeing Was an Act of Magical Power",
        "primary_term": "jr.t",
        "literal": "eye",
        "alignment_score": alignment_score("jr.t", "eye"),
        "midpoint_query": ["eye", "knowledge"],
        "midpoint_results": nearest[:10],
        "midpoint_score": nearest[0][1] if nearest else None,
        "hka_rank": hka_rank,
        "hka_match": hka_word,
        "hka_score": hka_score,
        "secondary": {
            "term": "ḥkꜣ",
            "literal": "magic, magical power",
            "alignment_score": alignment_score("ḥkꜣ", "magic"),
        }
    }


# D4: The Snake Is Divine, Not Wise
print("  D4: The Snake Is Divine, Not Wise")
mid, missing = midpoint("snake", "wisdom")
if mid is not None:
    nearest = nearest_egyptian(mid, 20)
    discoveries["D4_Snake"] = {
        "title": "The Snake Is Divine, Not Wise",
        "primary_term": nearest[0][0] if nearest else "nṯr",
        "literal": "serpent → finds god/divine",
        "alignment_score": None,
        "midpoint_query": ["snake", "wisdom"],
        "midpoint_results": nearest[:10],
        "midpoint_score": nearest[0][1] if nearest else None,
    }


# D5: Temple Is to House as God Is to Man
print("  D5: Temple : House :: God : Man")
h, t, m = get_en("house"), get_en("temple"), get_en("man")
if h is not None and t is not None and m is not None:
    analogy_vec = normalize(t - h + m)
    nearest = nearest_egyptian(analogy_vec, 20)
    ntr_rank, ntr_word, ntr_score = find_term_rank(nearest, ["nṯr"])
    discoveries["D5_Temple"] = {
        "title": "Temple : House :: God : Man",
        "primary_term": "nṯr",
        "literal": "god",
        "alignment_score": alignment_score("nṯr", "god"),
        "analogy_query": "temple - house + man",
        "analogy_results": nearest[:10],
        "analogy_score": nearest[0][1] if nearest else None,
        "ntr_rank": ntr_rank,
        "ntr_match": ntr_word,
        "ntr_score": ntr_score,
    }


# D6: Mother Is Royalty, Not Earth
print("  D6: Mother Is Royalty, Not Earth")
mid, missing = midpoint("mother", "earth")
if mid is not None:
    nearest = nearest_egyptian(mid, 20)
    discoveries["D6_Mother"] = {
        "title": "Mother Is Royalty, Not Earth",
        "primary_term": "mw.t",
        "literal": "mother",
        "alignment_score": alignment_score("mw.t", "mother"),
        "midpoint_query": ["mother", "earth"],
        "midpoint_results": nearest[:10],
        "midpoint_score": nearest[0][1] if nearest else None,
    }


# D7: Truth and Power Are the Same Force
print("  D7: Truth and Power Are the Same Force")
mid, missing = midpoint("truth", "power")
if mid is not None:
    nearest = nearest_egyptian(mid, 20)
    sxm_rank, sxm_word, sxm_score = find_term_rank(nearest, ["sḫm"])
    xft_rank, xft_word, xft_score = find_term_rank(nearest, ["ḫft", "ḫfti"])
    discoveries["D7_Truth"] = {
        "title": "Truth and Power Are the Same Force",
        "primary_term": "mꜣꜥ.t",
        "literal": "truth, justice, cosmic order",
        "alignment_score": alignment_score("mꜣꜥ.t", "truth"),
        "midpoint_query": ["truth", "power"],
        "midpoint_results": nearest[:10],
        "midpoint_score": nearest[0][1] if nearest else None,
        "sxm_rank": sxm_rank,
        "sxm_match": sxm_word,
        "sxm_score": sxm_score,
        "xft_rank": xft_rank,
        "xft_match": xft_word,
        "xft_score": xft_score,
        "secondary": {
            "term": "sḫm",
            "literal": "power, authority",
            "alignment_score": alignment_score("sḫm", "power"),
        }
    }


# D8: Love and Fear Meet at Eternity
print("  D8: Love and Fear Meet at Eternity")
mid, missing = midpoint("love", "fear")
if mid is not None:
    nearest = nearest_egyptian(mid, 20)
    nhh_rank, nhh_word, nhh_score = find_term_rank(nearest, ["nḥḥ", "r-nḥḥ", "r-(n)ḥḥ"])
    discoveries["D8_Eternity"] = {
        "title": "Love and Fear Meet at Eternity",
        "primary_term": "r-nḥḥ",
        "literal": "to eternity, forever",
        "alignment_score": alignment_score("r-nḥḥ", "eternal"),
        "midpoint_query": ["love", "fear"],
        "midpoint_results": nearest[:10],
        "midpoint_score": nearest[0][1] if nearest else None,
        "nhh_rank": nhh_rank,
        "nhh_match": nhh_word,
        "nhh_score": nhh_score,
    }


# ── Write output ─────────────────────────────────────────────────────────────
output = {
    "model_version": "V10 (1536d fused → 300d GloVe, Ridge alpha=1.0)",
    "accuracy": {
        "top1_pct": 30.67,
        "description": "Top-1 retrieval accuracy (V10 aligned vectors)"
    },
    "note": "alignment_score = cosine similarity between aligned Egyptian word and English concept. midpoint_score = cosine similarity of the top Egyptian result to the midpoint query vector.",
    "discoveries": discoveries,
}

with open(OUT, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nWrote {OUT}")
print(f"Model: V10 (alpha=1.0, accuracy=30.67%)")
print(f"Concept vocabulary: {len(en_words)} words")
print()
for name, d in discoveries.items():
    bs = d.get("alignment_score")
    ms = d.get("midpoint_score") or d.get("analogy_score")
    top = d.get("midpoint_results") or d.get("analogy_results") or []
    top_words = [f"{w}({s:.3f})" for w, s in top[:5]]
    print(f"  {name}: alignment={bs}, midpoint={ms}")
    print(f"    top: {', '.join(top_words)}")
    for key in ["mwt_rank", "hka_rank", "ntr_rank", "sxm_rank", "xft_rank", "nhh_rank"]:
        if key in d and d[key] is not None:
            match_key = key.replace("_rank", "_match")
            score_key = key.replace("_rank", "_score")
            print(f"    {key}: #{d[key]} = {d[match_key]} ({d[score_key]:.3f})")
    print()
