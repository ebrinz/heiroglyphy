"""
Extract bridge scores and midpoint scores for each discovery term.

Bridge score: cosine similarity between aligned Egyptian word and its English translation.
Midpoint score: cosine similarity between the English midpoint query and the Egyptian result.

Usage:
    python docs/extract_bridge_scores.py

Output: docs/bridge_scores.json

Note: Uses pickle to load the project's own serialized vocabulary mapping
(egyptian_aligned_vocab.pkl), not untrusted external data.
"""

import json
import pickle  # required for project's own .pkl vocab file
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINAL = REPO / "final_output"
OUT = REPO / "docs" / "bridge_scores.json"


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── Load embeddings ──────────────────────────────────────────────────────────
print("Loading embeddings...")
eg_data = np.load(FINAL / "egyptian_aligned_vectors.npz")
eg_vectors = eg_data["vectors"].astype(np.float32)

with open(FINAL / "egyptian_aligned_vocab.pkl", "rb") as f:
    eg_vocab = pickle.load(f)
eg_idx = eg_vocab if isinstance(eg_vocab, dict) else {w: i for i, w in enumerate(eg_vocab)}
eg_idx_to_word = {v: k for k, v in eg_idx.items()}

concept_data = np.load(FINAL / "concept_vectors.npz", allow_pickle=True)
en_vectors = concept_data["vectors"].astype(np.float32)
en_words = list(concept_data["words"])
en_idx = {w: i for i, w in enumerate(en_words)}


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
    va, vb = get_en(word_a), get_en(word_b)
    if va is None or vb is None:
        return None
    return normalize((normalize(va) + normalize(vb)) / 2)


def nearest_egyptian(query_vec, top_k=5):
    sims = eg_vectors @ query_vec
    top = np.argsort(sims)[-top_k:][::-1]
    return [(eg_idx_to_word[i], float(sims[i])) for i in top]


def bridge_score(eg_word, en_word):
    ev, nv = get_eg(eg_word), get_en(en_word)
    if ev is None or nv is None:
        return None
    return cosine_sim(ev, nv)


# ── Compute discovery scores ────────────────────────────────────────────────
print("Computing discovery scores...")

discoveries = {}

# D1: Gold Is Divine Flesh
mid = midpoint("gold", "divine")
if mid is not None:
    nearest = nearest_egyptian(mid, 5)
    discoveries["D1_Gold"] = {
        "title": "Gold Is Divine Flesh",
        "primary_term": "nṯr",
        "literal": "god, divine being",
        "bridge_score": bridge_score("nṯr", "god"),
        "midpoint_query": ["gold", "divine"],
        "midpoint_results": nearest,
        "midpoint_score": nearest[0][1] if nearest else None,
        "secondary": {
            "term": "nbw",
            "literal": "gold",
            "bridge_score": bridge_score("nbw", "gold"),
        }
    }

# D2: Silence Is the Condition of the Dead
mid = midpoint("silence", "death") if "silence" in en_idx else None
if mid is not None:
    nearest = nearest_egyptian(mid, 5)
    discoveries["D2_Silence"] = {
        "title": "Silence Is the Condition of the Dead",
        "primary_term": "mwt",
        "literal": "to die, dead",
        "bridge_score": bridge_score("mwt", "death"),
        "midpoint_query": ["silence", "death"],
        "midpoint_results": nearest,
        "midpoint_score": nearest[0][1] if nearest else None,
    }
else:
    nearest = nearest_egyptian(get_en("death"), 5)
    discoveries["D2_Silence"] = {
        "title": "Silence Is the Condition of the Dead",
        "primary_term": "mwt",
        "literal": "to die, dead",
        "bridge_score": bridge_score("mwt", "death"),
        "midpoint_query": ["silence", "death"],
        "midpoint_results": nearest,
        "midpoint_score": bridge_score("mwt", "death"),
        "note": "silence not in concept vectors; used death vector directly"
    }

# D3: Seeing Was an Act of Magical Power
mid = midpoint("eye", "knowledge")
if mid is not None:
    nearest = nearest_egyptian(mid, 5)
    hka_bridge = bridge_score("ḥkꜣ", "magic") if "magic" in en_idx else None
    discoveries["D3_Seeing"] = {
        "title": "Seeing Was an Act of Magical Power",
        "primary_term": "jr.t",
        "literal": "eye",
        "bridge_score": bridge_score("jr.t", "eye"),
        "midpoint_query": ["eye", "knowledge"],
        "midpoint_results": nearest,
        "midpoint_score": nearest[0][1] if nearest else None,
        "secondary": {
            "term": "ḥkꜣ",
            "literal": "magic, magical power",
            "bridge_score": hka_bridge,
        }
    }

# D4: The Snake Is Divine, Not Wise
mid = midpoint("snake", "wisdom")
if mid is not None:
    nearest = nearest_egyptian(mid, 5)
    discoveries["D4_Snake"] = {
        "title": "The Snake Is Divine, Not Wise",
        "primary_term": nearest[0][0] if nearest else "nṯr",
        "literal": "serpent → finds god/divine",
        "bridge_score": None,
        "midpoint_query": ["snake", "wisdom"],
        "midpoint_results": nearest,
        "midpoint_score": nearest[0][1] if nearest else None,
    }

# D5: Temple Is to House as God Is to Man
h, t, m = get_en("house"), get_en("temple"), get_en("man")
if h is not None and t is not None and m is not None:
    analogy_vec = normalize(t - h + m)
    nearest = nearest_egyptian(analogy_vec, 5)
    discoveries["D5_Temple"] = {
        "title": "Temple : House :: God : Man",
        "primary_term": "nṯr",
        "literal": "god",
        "bridge_score": bridge_score("nṯr", "god"),
        "analogy_query": "temple - house + man",
        "analogy_results": nearest,
        "analogy_score": nearest[0][1] if nearest else None,
    }

# D6: Mother Is Royalty, Not Earth
mid = midpoint("mother", "earth")
if mid is not None:
    nearest = nearest_egyptian(mid, 5)
    discoveries["D6_Mother"] = {
        "title": "Mother Is Royalty, Not Earth",
        "primary_term": "mw.t",
        "literal": "mother",
        "bridge_score": bridge_score("mw.t", "mother"),
        "midpoint_query": ["mother", "earth"],
        "midpoint_results": nearest,
        "midpoint_score": nearest[0][1] if nearest else None,
    }

# D7: Truth and Power Are the Same Force
mid = midpoint("truth", "power")
if mid is not None:
    nearest = nearest_egyptian(mid, 5)
    discoveries["D7_Truth"] = {
        "title": "Truth and Power Are the Same Force",
        "primary_term": "mꜣꜥ.t",
        "literal": "truth, justice, cosmic order",
        "bridge_score": bridge_score("mꜣꜥ.t", "truth"),
        "midpoint_query": ["truth", "power"],
        "midpoint_results": nearest,
        "midpoint_score": nearest[0][1] if nearest else None,
        "secondary": {
            "term": "sḫm",
            "literal": "power, authority",
            "bridge_score": bridge_score("sḫm", "power"),
        }
    }

# D8: Love and Fear Meet at Eternity
mid = midpoint("love", "fear")
if mid is not None:
    nearest = nearest_egyptian(mid, 5)
    eternity_bridge = bridge_score("r-nḥḥ", "eternal") if "eternal" in en_idx else None
    discoveries["D8_Eternity"] = {
        "title": "Love and Fear Meet at Eternity",
        "primary_term": "r-nḥḥ",
        "literal": "to eternity, forever",
        "bridge_score": eternity_bridge,
        "midpoint_query": ["love", "fear"],
        "midpoint_results": nearest,
        "midpoint_score": nearest[0][1] if nearest else None,
    }

# ── Write output ─────────────────────────────────────────────────────────────
output = {
    "accuracy": {
        "top1_pct": 32.35,
        "description": "Top-1 retrieval accuracy without bilingual supervision"
    },
    "discoveries": discoveries,
}

with open(OUT, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nWrote {OUT}")
for name, d in discoveries.items():
    bs = d.get("bridge_score")
    ms = d.get("midpoint_score") or d.get("analogy_score")
    top = d.get("midpoint_results") or d.get("analogy_results") or []
    top_words = [f"{w}({s:.3f})" for w, s in top[:3]]
    print(f"  {name}: bridge={bs}, midpoint={ms}")
    print(f"    top: {', '.join(top_words)}")
