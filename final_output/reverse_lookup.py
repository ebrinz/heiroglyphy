"""
Reverse lookup: Egyptian transliterated word → nearest English words in GloVe space.

Since the Egyptian vectors are already aligned to GloVe 300d space,
we just find nearest English neighbors for each Egyptian vector.
"""

import numpy as np
import pickle
import sys
from pathlib import Path

BASE = Path(__file__).parent

def load():
    data = np.load(BASE / "egyptian_aligned_vectors.npz")
    eg_vecs = data['vectors'].astype(np.float32)

    with open(BASE / "egyptian_aligned_vocab.pkl", 'rb') as f:
        eg_vocab = pickle.load(f)

    idx_to_eg = {v: k for k, v in eg_vocab.items()}

    # Load GloVe
    print("Loading GloVe...", file=sys.stderr)
    glove_words = []
    glove_vecs = []
    with open(BASE / "../heiro_v5_getdata/data/processed/glove.6B.300d.txt") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if len(vec) == 300:
                glove_words.append(word)
                glove_vecs.append(vec)

    glove_matrix = np.stack(glove_vecs)
    # L2 normalize
    norms = np.linalg.norm(glove_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    glove_matrix = glove_matrix / norms

    print(f"Loaded {len(glove_words)} English words, {len(eg_vocab)} Egyptian words", file=sys.stderr)
    return eg_vecs, eg_vocab, idx_to_eg, glove_matrix, glove_words


def reverse_lookup(word, eg_vecs, eg_vocab, glove_matrix, glove_words, topn=10):
    """Given an Egyptian transliterated word, find nearest English words."""
    if word not in eg_vocab:
        return None

    vec = eg_vecs[eg_vocab[word]].astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-10)

    sims = glove_matrix @ vec
    indices = np.argsort(sims)[-topn:][::-1]
    return [(glove_words[i], float(sims[i])) for i in indices]


def egyptian_neighbors(word, eg_vecs, eg_vocab, idx_to_eg, topn=10):
    """Find nearest Egyptian neighbors for an Egyptian word."""
    if word not in eg_vocab:
        return None

    vec = eg_vecs[eg_vocab[word]].astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-10)

    # Normalize all Egyptian vectors
    norms = np.linalg.norm(eg_vecs, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0] = 1
    normed = eg_vecs / norms

    sims = normed @ vec
    indices = np.argsort(sims)[-(topn+1):][::-1]
    # Skip the word itself
    results = []
    for i in indices:
        w = idx_to_eg[i]
        if w != word:
            results.append((w, float(sims[i])))
    return results[:topn]


if __name__ == "__main__":
    eg_vecs, eg_vocab, idx_to_eg, glove_matrix, glove_words = load()

    # Key Spell 125 words to test
    test_words = [
        "mAat",    # truth/justice
        "ib",      # heart
        "bA",      # soul/ba
        "Ax",      # akh/spirit
        "wsjr",    # Osiris
        "nTr",     # god
        "nTrw",    # gods
        "Htp",     # offering/peace
        "anx",     # life
        "mwt",     # death
        "wab",     # priest/pure
        "jmnt",    # west/underworld
        "dwAt",    # underworld/Duat
        "xpr",     # become/transform
        "wDa",     # judge
        "jb",      # heart (alternate)
        "HqA",     # ruler
        "isft",    # chaos/wrongdoing
        "grH",     # night
        "Dd",      # stability/djed
        "tm",      # not/complete
        "jr",      # do/make
        "rn",      # name
        "smA",     # kill/unite
        "xft",     # enemy
    ]

    for word in test_words:
        print(f"\n{'='*60}")
        print(f"Egyptian: {word}")

        # English neighbors
        results = reverse_lookup(word, eg_vecs, eg_vocab, glove_matrix, glove_words)
        if results:
            print(f"  English neighbors:")
            for eng, sim in results[:7]:
                print(f"    {eng:20s} {sim:.4f}")
        else:
            print(f"  NOT IN VOCABULARY")

        # Egyptian neighbors
        eg_results = egyptian_neighbors(word, eg_vecs, eg_vocab, idx_to_eg)
        if eg_results:
            print(f"  Egyptian neighbors:")
            for eg, sim in eg_results[:5]:
                print(f"    {eg:20s} {sim:.4f}")
