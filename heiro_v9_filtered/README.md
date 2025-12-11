# Heiroglyphy V9: Filtered Alignment & Export

## Overview

V9 exports Egyptian hieroglyphic word vectors aligned to English GloVe space for use in downstream applications. This version builds on V7's proven methodology (29.10% Top-1 accuracy) while experimenting with anchor filtering strategies.

## Key Finding: Filtering Didn't Help

We investigated whether removing "function word pollution" (articles, pronouns, prepositions) from the anchor dictionary would improve alignment quality. **It did not.**

| Approach | Anchors Used | Top-1 Accuracy |
|----------|--------------|----------------|
| V7 Original (all anchors) | 6,700 | **29.10%** |
| V9 Strict filtering (content words only, conf >= 0.60) | 1,768 | 23.16% |
| V9 Hybrid filtering (high-conf function + content) | 3,970 | 24.43% |
| V9 Confidence-weighted alignment | 6,700 | 28.21% |
| **V9 Ridge (V7 replication)** | **6,700** | **29.10%** |

**Conclusion**: The function words, despite being semantically less interesting, provide crucial signal for learning the alignment transformation. Quantity matters.

## Output Files

The `outputs/` directory contains the following files:

### 1. `egyptian_aligned_vectors.npy` (~92 MB)
- Shape: `(80662, 300)`
- 80,662 Egyptian words transformed into 300-dimensional English GloVe space
- L2 normalized
- dtype: `float32`

```python
import numpy as np
vectors = np.load('egyptian_aligned_vectors.npy')
print(vectors.shape)  # (80662, 300)
```

### 2. `egyptian_aligned_vocab.pkl` (~1.5 MB)
- Python dict: `{word: index}`
- Maps Egyptian transliteration strings to vector indices

```python
import pickle
with open('egyptian_aligned_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Look up a word
idx = vocab['wsjr']  # Osiris
vector = vectors[idx]
```

### 3. `procrustes_transform.npy` (~1.8 MB)
- Shape: `(1537, 300)` - includes bias as last row
- The learned Ridge regression transformation
- Can transform new Egyptian vectors to English space

```python
transform = np.load('procrustes_transform.npy')
W = transform[:-1]  # (1536, 300) coefficient matrix
bias = transform[-1]  # (300,) bias vector

# Transform a new vector
new_english = egyptian_vec @ W + bias
```

### Additional Files
- `ridge_coefficients.npy`: Just the W matrix (1536x300)
- `ridge_bias.npy`: Just the bias vector (300,)

## Usage Example

```python
import numpy as np
import pickle
from gensim.models import KeyedVectors

# Load Egyptian vectors and vocab
vectors = np.load('outputs/egyptian_aligned_vectors.npy')
with open('outputs/egyptian_aligned_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load English GloVe for similarity search
glove = KeyedVectors.load_word2vec_format('glove.6B.300d.txt', binary=False, no_header=True)

# Find English words similar to an Egyptian word
egyptian_word = 'nṯr'  # god
if egyptian_word in vocab:
    idx = vocab[egyptian_word]
    egyptian_vec = vectors[idx]

    # Find nearest English words
    similar = glove.similar_by_vector(egyptian_vec, topn=5)
    print(f"'{egyptian_word}' is similar to:")
    for word, score in similar:
        print(f"  {word}: {score:.3f}")
```

## Methodology

### Data Sources
- **Egyptian Embeddings**: V7 FastText model (768d) trained on 100,729 BBAW corpus sentences
- **Fused Embeddings**: 768d text + 768d visual = 1536d (visual component zeros due to mapping issue)
- **English Embeddings**: GloVe 6B 300d (400,000 words)
- **Anchors**: 8,541 Egyptian-English pairs, 6,700 with valid vocabulary overlap

### Alignment Process
1. Load Egyptian (1536d) and English (300d) embeddings
2. Extract anchor pairs where both words exist in vocabularies
3. Train Ridge regression (α=1.0) to map Egyptian → English
4. Apply transformation to all 80,662 Egyptian words
5. L2 normalize output vectors

### Anchor Analysis

The original anchor dictionary has these characteristics:
- **Total anchors**: 8,541
- **Function words**: 51% (articles, pronouns, prepositions, etc.)
- **Content words**: 49% (nouns, verbs, named entities)
- **Average confidence**: 74.7%

Top content word anchors (high confidence):
- `wsjr` → osiris (98%)
- `ppy` → pepi (99%)
- `ḥr,w` → horus (97%)
- `rꜥw` → re (97%)
- `ꜣs,t` → isis (98%)

Function words that dominate by frequency:
- `n` → the (34% confidence)
- `m` → the (37% confidence)
- `=f` → he (38% confidence)

## Experiments

### Strict Filtering (23.16%)
Removed all function words and kept only content words with confidence >= 60%.
Result: Accuracy dropped significantly due to insufficient training examples.

### Hybrid Filtering (24.43%)
Kept content words (conf >= 50%) and high-confidence function words (conf >= 70%).
Result: Still worse than using all anchors.

### Confidence Weighting (28.21%)
Used all anchors but weighted by confidence² during training.
Result: Slightly worse than unweighted.

### Ridge Regression (29.10%)
V7's original approach with all anchors and Ridge regression (α=1.0).
Result: Best performance, successfully replicated.

## Directory Structure

```
heiro_v9_filtered/
├── README.md
├── data/
│   └── processed/
│       ├── filtered_anchors.json        # Strict filtered (3,227 pairs)
│       ├── filtered_anchors_hybrid.json # Hybrid filtered (5,739 pairs)
│       ├── filter_statistics.json
│       ├── filter_statistics_hybrid.json
│       └── alignment_results_v9.json
├── outputs/
│   ├── egyptian_aligned_vectors.npy     # Main output (92 MB)
│   ├── egyptian_aligned_vocab.pkl       # Word -> index mapping
│   ├── procrustes_transform.npy         # Transformation matrix
│   ├── ridge_coefficients.npy           # W matrix only
│   └── ridge_bias.npy                   # Bias vector only
└── scripts/
    ├── 01_filter_anchors.py             # Strict filtering
    ├── 01b_filter_anchors_hybrid.py     # Hybrid filtering
    ├── 02_align_and_export.py           # Alignment with strict filters
    ├── 02b_align_and_export_hybrid.py   # Alignment with hybrid filters
    ├── 03_weighted_alignment.py         # Confidence-weighted alignment
    └── 04_ridge_alignment.py            # V7 replication (SOTA)
```

## Version History

| Version | Accuracy | Notes |
|---------|----------|-------|
| V1-V2 | Failed | Neural approaches unstable |
| V3 | 22.0% | Baseline Procrustes |
| V4 | 15% | CSLS hurt performance |
| V5 | 24.53% | 10x more data helped |
| V6 | 0.47% | BERT vocabulary mismatch |
| **V7** | **29.10%** | **768d FastText, SOTA** |
| V8 | 28.16% | Coptic bridge hurt |
| V9 | 29.10% | Export files, filtering experiments |
