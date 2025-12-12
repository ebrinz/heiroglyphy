# Egyptian Hieroglyphic Word Vectors (GloVe-Aligned)

Pre-trained word vectors for 80,662 ancient Egyptian hieroglyphic words, aligned to English GloVe 300-dimensional space. This enables direct semantic comparison and vector arithmetic between Egyptian and English vocabularies.

## Files

### Core Vectors
| File | Size | Description |
|------|------|-------------|
| `egyptian_aligned_vectors.npz` | 43 MB | 80,662 × 300 float16 matrix (compressed) |
| `egyptian_aligned_vocab.pkl` | 1.5 MB | Dictionary: Egyptian word → vector index |
| `egyptian_lookup.py` | 9 KB | Python utility for semantic lookup (requires gensim) |
| `egyptian_lookup_lite.py` | 6 KB | Lightweight version (numpy only, for edge/mobile) |

### Reference Data
| File | Size | Description |
|------|------|-------------|
| `concept_vectors.npz` | 157 KB | 279 pre-computed English concept vectors (organized by category) |
| `concept_categories.json` | 4 KB | Category metadata (elements, deities, royalty, etc.) |
| `hieroglyph_dictionary.tsv` | 720 KB | 11,727 entries: hieroglyph → transliteration → English |
| `hieroglyph_dictionary.json` | 2 MB | Same data in JSON format for programmatic access |
| `EgyptianHiero.ttf` | 2.7 MB | Hieroglyphic font (EgyptianHiero 4.03) |
| `esoteric_glove_vectors.npz` | 62 KB | Legacy 113-concept vectors (superseded by concept_vectors.npz) |

## Quick Start

```python
import numpy as np
import pickle
from gensim.models import KeyedVectors

# Load Egyptian vectors (compressed float16, upcast to float32)
data = np.load("egyptian_aligned_vectors.npz")
vectors = data['vectors'].astype(np.float32)

with open("egyptian_aligned_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load any GloVe model (must be 300d)
glove = KeyedVectors.load_word2vec_format("glove.6B.300d.txt", binary=False, no_header=True)

# Find Egyptian words similar to an English concept
english_vec = glove["sun"]
similarities = vectors @ english_vec  # cosine similarity (vectors are L2 normalized)
top_idx = np.argsort(similarities)[-5:][::-1]

idx_to_word = {v: k for k, v in vocab.items()}
for i in top_idx:
    print(f"{idx_to_word[i]}: {similarities[i]:.3f}")
```

## Using the Lookup Utility

```python
from egyptian_lookup import EgyptianLookup

lookup = EgyptianLookup(
    vectors_path="egyptian_aligned_vectors.npz",
    vocab_path="egyptian_aligned_vocab.pkl",
    glove=glove  # or path to glove .txt file
)

# Single concept lookup
results = lookup.find("water")
# [('mw', 0.61), ('rmw', 0.43), ...]  # mw = "water" in Egyptian

# Combine multiple concepts
results = lookup.find_relationship(["death", "rebirth", "transformation"])

# Vector analogy: king:queen :: god:?
results = lookup.find_analogy("king", "queen", "god")

# Weighted blend
results = lookup.find_blend({"sun": 0.7, "power": 0.3})

# Contrast: power without destruction
results = lookup.find_contrast(positive=["power", "wisdom"], negative=["destruction"])

# Semantic midpoint
results = lookup.find_midpoint("life", "death")
```

## Methodology

### Data Sources

- **Egyptian Corpus**: 100,729 sentences from the BBAW (Berlin-Brandenburg Academy) hieroglyphic text database
- **English Embeddings**: GloVe 6B 300d (400,000 words)
- **Anchor Dictionary**: 8,541 Egyptian-English word pairs extracted via co-occurrence analysis from parallel German translations

### Training Pipeline

1. **FastText Training**: 768-dimensional skip-gram embeddings trained on Egyptian transliteration corpus (10 epochs, window=5)

2. **Visual Feature Extraction**: 768-dimensional ResNet-50 features from hieroglyph images (HamdiJr dataset)

3. **Fusion**: Text embeddings (768d) + Visual embeddings (768d) = 1536d fused representations

4. **Anchor Extraction**: Co-occurrence analysis on parallel texts to identify 8,541 Egyptian-English word pairs

5. **Alignment**: Ridge regression (α=1.0) to learn transformation from Egyptian 1536d space to English GloVe 300d space

6. **Projection**: All 80,662 Egyptian vectors transformed to GloVe space and L2-normalized

### Evaluation

| Metric | Score |
|--------|-------|
| **Top-1 Accuracy** | **30.67%** |
| Top-5 Accuracy | 37.69% |
| Top-10 Accuracy | 41.49% |
| Anchor Coverage | 78.4% (6,700 / 8,541) |
| Visual Match Rate | 0.08% |

Accuracy measured on held-out test set: does the nearest English neighbor match the expected translation?

**Note**: This is the V10 SOTA methodology. The visual match rate is low because most Egyptian vocabulary in the corpus doesn't have direct Gardiner code mappings (transliteration ≠ glyph codes). However, the 1536d architecture (768d text + 768d visual padding) provides better capacity for the alignment model.

### Known Limitations

1. **Hubness**: Common English words like "the" appear as neighbors to many Egyptian words due to GloVe geometry

2. **Domain Bias**: Training corpus is primarily religious/funerary texts, so religious vocabulary is better represented

3. **Polysemy**: Many Egyptian words have multiple meanings; alignment captures dominant usage

4. **Function Words**: Articles, pronouns, and prepositions have lower alignment quality than content words

## Vocabulary

The Egyptian vocabulary uses standard Egyptological transliteration:

| Symbol | Example | Meaning |
|--------|---------|---------|
| `ꜥ` | ꜥnḫ | ayin (life) |
| `ḥ` | ḥr,w | h-dot (Horus) |
| `ḫ` | ḫpr | kh (become) |
| `š` | šms | sh (follow) |
| `ṯ` | nṯr | tj (god) |
| `=` | =f | suffix pronoun (his) |
| `.pl` | nṯr.pl | plural marker (gods) |

### Sample Vocabulary

| Egyptian | English | Confidence |
|----------|---------|------------|
| nṯr | god | 60% |
| wsjr | Osiris | 98% |
| ḥr,w | Horus | 97% |
| mw | water | 74% |
| zꜣ | son | 79% |
| nswt | king | 57% |
| p,t | sky | 63% |
| jb | heart | 48% |

## Version History

This is the **V10 SOTA** of the Heiroglyphy project, representing the culmination of 12 alignment approaches:

| Version | Approach | Accuracy |
|---------|----------|----------|
| V3 | Procrustes baseline | 22.0% |
| V5 | Scaled corpus (10x data) | 24.5% |
| V6 | BERT (failed - vocab mismatch) | 0.5% |
| V7 | FastText 768d | 29.1% |
| V8 | Coptic bridge (regression) | 28.2% |
| V9 | Text + Visual (1536d) | 30.5% |
| **V10** | **V9 + Gardiner mapping** | **30.67%** |

## Hieroglyph Dictionary

The `hieroglyph_dictionary.tsv` provides a lookup table from hieroglyphs to English:

| Column | Description |
|--------|-------------|
| `hieroglyph` | Unicode hieroglyph character(s) |
| `gardiner_codes` | Gardiner sign list codes (e.g., "M17", "G1,X1") |
| `transliteration` | Egyptological transliteration |
| `english` | English translation/meaning |
| `occurrence` | Frequency score in corpus |

Sample entries (sorted by frequency):
```
𓇋  M17   i     I, me, my
𓏏  X1    t     you, your
𓆑  I9    f     he, him, his
𓂋  D21   r     to, at, concerning
𓁹  D4    iri   create, make, do
```

## Concept Categories

The 279 pre-computed concept vectors are organized into 18 semantic categories:

| Category | Examples | Count |
|----------|----------|-------|
| elements | fire, water, earth, air, light | 14 |
| celestial | sun, moon, star, sky, heaven | 14 |
| geography | river, nile, mountain, desert | 14 |
| animals | lion, snake, cobra, falcon, scarab | 22 |
| deities | god, goddess, divine, sacred | 12 |
| afterlife | death, rebirth, soul, spirit, tomb | 16 |
| royalty | king, queen, pharaoh, throne | 13 |
| virtues | truth, justice, wisdom, power | 12 |
| body | head, eye, heart, hand, blood | 15 |
| objects | temple, pyramid, boat, sword, gold | 21 |
| ... | (see concept_categories.json) | ... |

## License

The underlying BBAW corpus data is subject to its original licensing terms. The trained vectors and alignment methodology are provided for research and educational purposes.

## Citation

If you use these vectors, please cite the BBAW hieroglyphic corpus:

```
Thesaurus Linguae Aegyptiae (TLA)
Berlin-Brandenburg Academy of Sciences and Humanities
https://aaew.bbaw.de/tla/
```
