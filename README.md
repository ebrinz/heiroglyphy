# Heiroglyphy: Computational Translation of Ancient Egyptian

**Heiroglyphy** is a digital humanities research project exploring the application of **Vector-to-Vector (vec2vec)** alignment techniques to the translation of Ancient Egyptian Hieroglyphs.

The core research question is: **Can we map the geometric "shape" of the Ancient Egyptian language onto Modern English to discover meanings without a traditional dictionary?**

This repository documents 12 experimental iterations, from complex neural models to elegant linear algebra, achieving **30.67% Top-1 accuracy** on unsupervised hieroglyphic-to-English translation.

## 🧬 The Vec2Vec Hypothesis

All attempts in this project are grounded in the **Distributional Hypothesis**: that words with similar meanings appear in similar contexts. By training vector embeddings (Word2Vec/FastText), we turn languages into geometric shapes.

The "vec2vec" challenge is to find a transformation function $f$ such that:
$$ f(v_{hieroglyph}) \approx v_{english} $$

*   **Attempts 1 & 2** explored **Neural Vec2Vec**: Using deep neural networks to learn non-linear mappings between the spaces.
*   **Attempts 3-12** explored **Linear Alignment (Procrustes)**: Using the analytic solution (SVD) rather than neural networks.

## 📊 Progress Summary

| Attempt | Technique | Top-1 Accuracy | Status |
|---------|-----------|----------------|--------|
| V1 | Neural Vec2Vec (Multi-Space) | - | ❌ Failed (instability) |
| V2 | Unsupervised Neural Vec2Vec | - | ❌ Failed (isomorphism gap) |
| V3 | Linear Procrustes + Anchors | 22.0% | ✅ Baseline |
| V4 | Linear + CSLS | 15.0% | ⚠️ Negative result |
| V5 | Linear + 10x Data | 24.53% | ✅ Scaled baseline |
| V6 | BERT Contextual | 0.47% | ❌ Failed (tokenization) |
| V7 | FastText 768d | 29.10% | ✅ Text-only breakthrough |
| V8 | Coptic Bridge | 28.16% | ⚠️ Negative result |
| V9 | Visual Features (1536d) | 30.52% | ✅ First 30%+ |
| V10 | Vocab Normalization | **30.67%** | ✅ **Current SOTA** 🎉 |
| V11 | MLP + N-grams | 28.76% | ⚠️ Regression |
| V12 | Egyptian→German | 12.90% | 🧪 Exploratory |

**Key Insight**: Simple linear methods with good data outperform complex neural architectures for low-resource ancient language alignment.

## 📂 Project Structure

### [Attempt 1: The Translation Bridge (`heiro_v1`)](./heiro_v1)
*   **Technique**: **Neural Vec2Vec (Multi-Space)**.
*   **Strategy**: Bridging Hieroglyphic and English spaces using German as a pivot.
*   **Outcome**: Complexity of aligning three spaces with neural networks led to instability.

### [Attempt 2: The Purist Geometric Mapping (`heiro_v2`)](./heiro_v2)
*   **Technique**: **Unsupervised Neural Vec2Vec**.
*   **Strategy**: Adversarial alignment based purely on geometric density, zero supervision.
*   **Outcome**: Failed due to the "Isomorphism Gap" - Ancient Egyptian is too different from Modern English.

### [Attempt 3: Anchor-Guided Alignment (`heiro_v3`)](./heiro_v3)
*   **Technique**: **Linear Vec2Vec (Orthogonal Procrustes)**.
*   **Strategy**: ~1,300 anchor points with analytic linear rotation.
*   **Outcome**: **22% accuracy** - First successful alignment, proving linear methods work.

### [Attempt 4: CSLS Refinement (`heiro_v4`)](./heiro_v4)
*   **Technique**: **Linear Vec2Vec + CSLS**.
*   **Strategy**: Cross-Domain Similarity Local Scaling to reduce hubness.
*   **Outcome**: **15% accuracy** ⚠️ - CSLS too aggressive for sparse datasets.

### [Attempt 5: Scaled Corpus Alignment (`heiro_v5_getdata`)](./heiro_v5_getdata)
*   **Technique**: **Procrustes + 10x Data**.
*   **Strategy**: Combined corpus of 104,000 texts, 8,541 anchor pairs from TLA, Ramses, BBAW.
*   **Outcome**: **24.53% accuracy** - 11.5% relative improvement. Perfect hits on deities (Osiris: 61.5%, Horus: 62.1%).

### [Attempt 6: BERT Contextual Embeddings (`heiro_v6_BERT`)](./heiro_v6_BERT)
*   **Technique**: **BERT Contextual Embeddings**.
*   **Strategy**: Context-aware representations for polysemy.
*   **Outcome**: **0.47% accuracy** ❌ - WordPiece tokenizer destroyed hieroglyphic transliteration.

### [Attempt 7: FastText 768d (`heiro_v7_FastTextVisual`)](./heiro_v7_FastTextVisual)
*   **Technique**: **Large-Scale FastText (768d)**.
*   **Strategy**: 2.56x larger embeddings + Ridge regression alignment.
*   **Outcome**: **29.10% accuracy** ✅ - 18.6% relative improvement over V5.

### [Attempt 8: Coptic Bridge (`heiro_v8_use_coptic`)](./heiro_v8_use_coptic)
*   **Technique**: **Coptic Bridge Alignment**.
*   **Strategy**: Using Coptic cognates to expand anchors (+368 pairs).
*   **Outcome**: **28.16% accuracy** ⚠️ - Etymology ≠ Semantics. Quality > Quantity for anchors.

### [Attempt 9: Visual Features Redux (`heiro_v9_use_visuals_again`)](./heiro_v9_use_visuals_again)
*   **Technique**: **Text + Visual Fusion (1536d)**.
*   **Strategy**: ResNet-50 visual features from HamdiJr hieroglyph images fused with FastText.
*   **Outcome**: **30.52% accuracy** ✅ - First model to break 30%! Visual match rate was 0%, but larger dimensionality helped.

### [Attempt 10: Vocabulary Refinement (`heiro_v10_refinement`)](./heiro_v10_refinement)
*   **Technique**: **Vocabulary Normalization + Lexicon Integration**.
*   **Strategy**: Cleaned vocabulary, integrated HamdiJr lexicon, normalized transliteration variants.
*   **Outcome**: **30.67% accuracy** ✅ - **Current SOTA**. Minor gains from data cleaning.

### [Attempt 11: MLP Training (`heiro_v11`)](./heiro_v11)
*   **Technique**: **MLP + N-gram Features**.
*   **Strategy**: Neural approach with cleaner data pipeline.
*   **Outcome**: **28.76% accuracy** ⚠️ - Confirms linear methods beat neural for this task.

### [Attempt 12: Egyptian→German (`heiro_v12`)](./heiro_v12)
*   **Technique**: **Procrustes with German Target**.
*   **Strategy**: Minimal 80-anchor alignment directly to German (original translation language).
*   **Outcome**: **12.90% accuracy** 🧪 - Exploratory work, not SOTA attempt.

---

### [Final Output (`final_output`)](./final_output)

Production-ready Egyptian word vectors aligned to GloVe 300d space:

| File | Size | Description |
|------|------|-------------|
| `egyptian_aligned_vectors.npz` | 43 MB | 80,662 Egyptian words (float16 compressed) |
| `egyptian_aligned_vocab.pkl` | 1.5 MB | Word → vector index mapping |
| `egyptian_lookup.py` | 9 KB | Full lookup utility (requires gensim) |
| `egyptian_lookup_lite.py` | 6 KB | Edge/mobile version (numpy only) |
| `esoteric_glove_vectors.npz` | 62 KB | 113 pre-computed concept vectors |

**Quick Usage:**
```python
from egyptian_lookup import EgyptianLookup

lookup = EgyptianLookup(
    vectors_path="egyptian_aligned_vectors.npz",
    vocab_path="egyptian_aligned_vocab.pkl",
    glove=glove  # any GloVe 300d model
)

# Find Egyptian words for English concepts
lookup.find("sun")  # [('ḥrw-nbw', 0.40), ('ḥrw', 0.39), ...]
lookup.find_relationship(["death", "rebirth"])  # semantic combinations
lookup.find_blend({"power": 0.7, "wisdom": 0.3})  # weighted blends
```

## 🚀 Getting Started

**For using pre-trained vectors**, start with **`final_output`** - production-ready files and lookup utilities.

**For the SOTA methodology**, see **`heiro_v9_use_visuals_again`** (30.52%) or **`heiro_v10_refinement`** (30.67%).

**For understanding the baseline**, **`heiro_v7_FastTextVisual`** documents the 768d FastText approach (29.10%).

**For the data pipeline**, **`heiro_v5_getdata`** covers corpus assembly and anchor extraction.

### Prerequisites
*   Python 3.8+
*   `gensim`, `numpy`, `scikit-learn`, `pandas`, `jupyter`

### Usage
```bash
cd heiro_v5_getdata
jupyter notebook
```

## 📚 Data Sources
*   **Thesaurus Linguae Aegyptiae (TLA)**: Primary source for transliterations and German translations
*   **Ramses Online**: Additional hieroglyphic texts (German translations)
*   **Berlin-Brandenburg Academy (BBAW)**: Large-scale hieroglyphic corpus from HuggingFace
*   **HamdiJr/Egyptian_hieroglyphs**: 4,210 labeled hieroglyph images + lexicon
*   **GloVe**: Pre-trained English word embeddings (300d)

## Key Learnings

1. **Linear > Neural** for low-resource alignment (Procrustes beats MLPs/GANs)
2. **Dimensionality matters** (768d >> 300d, even 1536d helps)
3. **Quality > Quantity** for anchors (Coptic cognates hurt performance)
4. **Modern NLP fails** on ancient languages (BERT tokenization destroys meaning)
5. **Visual features untapped** (0% match rate, but architecture still helped)
