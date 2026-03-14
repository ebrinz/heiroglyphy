"""
Generate semantic-axis projections from real Heiroglyphy embeddings for Manim.

Instead of t-SNE (which distorts global structure), this projects all vectors
onto two human-interpretable axes derived from the embedding space itself:

    X-axis:  mortal ← → divine     (direction: "god" minus "man")
    Y-axis:  death  ← → life       (direction: "life" minus "death")

Every dot's position is semantically meaningful and the audience can read the axes.

Produces: docs/viz_data.json

Run:
    python docs/generate_viz_data.py

Note: Pickle is used to load egyptian_aligned_vocab.pkl — this is the project's
own serialized vocabulary mapping, not untrusted external data.
"""

import json
import pickle  # required for project's .pkl vocab file
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FINAL = REPO / "final_output"
OUT = REPO / "docs" / "viz_data.json"


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── Load vectors ───────────────────────────────────────────────────────────────
print("Loading Egyptian vectors...")
eg_data = np.load(FINAL / "egyptian_aligned_vectors.npz")
eg_vectors = eg_data["vectors"].astype(np.float32)

with open(FINAL / "egyptian_aligned_vocab.pkl", "rb") as f:
    eg_vocab = pickle.load(f)
idx_to_word = {v: k for k, v in eg_vocab.items()}

print("Loading concept vectors...")
concept_data = np.load(FINAL / "concept_vectors.npz", allow_pickle=True)
en_vectors = concept_data["vectors"].astype(np.float32)
en_words = list(concept_data["words"])
en_lookup = {w: i for i, w in enumerate(en_words)}

with open(FINAL / "concept_categories.json") as f:
    categories = json.load(f)

word_to_cat = {}
for cat, words in categories.items():
    for w in words:
        word_to_cat[w] = cat


# ── Define semantic axes ───────────────────────────────────────────────────────
print("Computing semantic axes...")

def axis_from(pos_word, neg_word):
    """Create a unit direction vector from neg_word → pos_word."""
    p = en_vectors[en_lookup[pos_word]]
    n = en_vectors[en_lookup[neg_word]]
    return normalize(p - n)

x_axis = axis_from("god", "man")       # mortal → divine
y_axis = axis_from("life", "death")    # death → life

# Orthogonalize y relative to x (Gram-Schmidt) so axes are independent
y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
y_axis = normalize(y_axis)

print(f"  X-axis (mortal→divine): cos(god)={np.dot(en_vectors[en_lookup['god']], x_axis):.3f}")
print(f"  Y-axis (death→life):    cos(life)={np.dot(en_vectors[en_lookup['life']], y_axis):.3f}")


# ── Project English concepts ──────────────────────────────────────────────────
print("Projecting English concepts...")

english_points = []
for i, word in enumerate(en_words):
    vec = en_vectors[i]
    x = float(np.dot(vec, x_axis))
    y = float(np.dot(vec, y_axis))
    cat = word_to_cat.get(word, "other")
    english_points.append({
        "word": word,
        "x": round(x, 4),
        "y": round(y, 4),
        "category": cat,
    })


# ── Select & project Egyptian words ───────────────────────────────────────────
print("Selecting representative Egyptian words...")

# Probe concepts — pick Egyptian words nearest to these
probe_concepts = [
    "water", "sun", "god", "king", "death", "life", "gold", "temple",
    "heart", "eye", "snake", "lion", "bread", "son", "mother", "truth",
    "sky", "star", "fire", "earth", "river", "falcon", "priest", "soul",
    "tomb", "magic", "love", "fear", "peace", "beer", "house", "justice",
    "moon", "sacred", "offering", "spirit", "queen", "sword", "wisdom",
]

eg_selected = {}  # idx → { word, concept }

for concept in probe_concepts:
    if concept not in en_lookup:
        continue
    cvec = en_vectors[en_lookup[concept]]
    sims = eg_vectors @ cvec
    top_indices = np.argsort(sims)[-3:][::-1]
    for idx in top_indices:
        if idx not in eg_selected:
            eg_selected[idx] = {
                "word": idx_to_word[idx],
                "concept": concept,
                "sim": float(sims[idx]),
            }

# Add top-150 frequency words for density
for i in range(150):
    if i not in eg_selected:
        sims = en_vectors @ eg_vectors[i]
        best = int(np.argmax(sims))
        eg_selected[i] = {
            "word": idx_to_word[i],
            "concept": en_words[best],
            "sim": float(sims[best]),
        }

egyptian_points = []
for idx, info in sorted(eg_selected.items()):
    vec = eg_vectors[idx]
    x = float(np.dot(vec, x_axis))
    y = float(np.dot(vec, y_axis))
    egyptian_points.append({
        "word": info["word"],
        "x": round(x, 4),
        "y": round(y, 4),
        "nearest_concept": info["concept"],
        "similarity": round(info["sim"], 4),
    })


# ── Build anchor connections ──────────────────────────────────────────────────
# Connect each selected Egyptian word to its nearest English concept
anchor_lines = []
for pt in egyptian_points:
    concept = pt["nearest_concept"]
    if concept in en_lookup:
        en_pt = english_points[en_lookup[concept]]
        anchor_lines.append({
            "egyptian": pt["word"],
            "english": concept,
            "eg_x": pt["x"],
            "eg_y": pt["y"],
            "en_x": en_pt["x"],
            "en_y": en_pt["y"],
        })

# ── Highlighted pairs for storytelling ─────────────────────────────────────────
# These are the "golden hits" and interesting cases from DISCOVERIES.md
highlights = []

def find_eg_point(word_substr):
    for pt in egyptian_points:
        if word_substr in pt["word"]:
            return pt
    return None

highlight_pairs = [
    ("mw", "water", "Perfect hit — 'mw' maps to 'water' across 4,000 years"),
    ("wsjr", "god", "Osiris — strongest deity alignment (61.5%)"),
    ("nṯr", "god", "nṯr — 'god/divine' in Egyptian"),
    ("nsw", "king", "nsw — 'king' in Egyptian"),
]

for eg_substr, en_word, note in highlight_pairs:
    eg_pt = find_eg_point(eg_substr)
    if eg_pt and en_word in en_lookup:
        en_pt = english_points[en_lookup[en_word]]
        highlights.append({
            "egyptian": eg_pt["word"],
            "english": en_word,
            "eg_x": eg_pt["x"],
            "eg_y": eg_pt["y"],
            "en_x": en_pt["x"],
            "en_y": en_pt["y"],
            "note": note,
        })


# ── Category colors ───────────────────────────────────────────────────────────
category_colors = {
    "elements":   "#e74c3c",
    "celestial":  "#f39c12",
    "geography":  "#27ae60",
    "plants":     "#2ecc71",
    "animals":    "#e67e22",
    "deities":    "#9b59b6",
    "afterlife":  "#8e44ad",
    "magic":      "#c0392b",
    "royalty":    "#f1c40f",
    "titles":     "#d4ac0d",
    "society":    "#3498db",
    "virtues":    "#1abc9c",
    "emotions":   "#e91e63",
    "states":     "#607d8b",
    "actions":    "#795548",
    "body":       "#ff7043",
    "objects":    "#78909c",
    "time":       "#5c6bc0",
    "numbers":    "#90a4ae",
    "egyptian":   "#f5c518",
}


# ── Write output ──────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "x_axis": "mortal ← → divine (god − man)",
        "y_axis": "death ← → life (life − death, orthogonalized)",
        "n_egyptian": len(egyptian_points),
        "n_english": len(english_points),
        "n_anchors": len(anchor_lines),
        "n_highlights": len(highlights),
    },
    "axes": {
        "x_label_pos": "divine",
        "x_label_neg": "mortal",
        "y_label_pos": "life",
        "y_label_neg": "death",
    },
    "category_colors": category_colors,
    "egyptian": egyptian_points,
    "english": english_points,
    "anchors": anchor_lines,
    "highlights": highlights,
}

with open(OUT, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nWrote {OUT}")
print(f"  {len(egyptian_points)} Egyptian points")
print(f"  {len(english_points)} English points")
print(f"  {len(anchor_lines)} anchor connections")
print(f"  {len(highlights)} highlighted pairs")
