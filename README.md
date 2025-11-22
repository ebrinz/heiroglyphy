# Heiroglyphy: Computational Translation of Ancient Egyptian

**Heiroglyphy** is a digital humanities research project exploring the application of **Vector-to-Vector (vec2vec)** alignment techniques to the translation of Ancient Egyptian Hieroglyphs.

The core research question is: **Can we map the geometric "shape" of the Ancient Egyptian language onto Modern English to discover meanings without a traditional dictionary?**

This repository contains three distinct experimental approaches to this "vec2vec" hypothesis, documenting the evolution from complex neural models to elegant linear algebra.

## 🧬 The Vec2Vec Hypothesis

All three attempts in this project are grounded in the **Distributional Hypothesis**: that words with similar meanings appear in similar contexts. By training vector embeddings (Word2Vec/FastText), we turn languages into geometric shapes.

The "vec2vec" challenge is to find a transformation function $f$ such that:
$$ f(v_{hieroglyph}) \approx v_{english} $$

*   **Attempts 1 & 2** explored **Neural Vec2Vec**: Using deep neural networks (specifically a custom `Vec2VecModel` class) to learn non-linear mappings between the spaces.
*   **Attempt 3** explored **Linear Alignment (Procrustes)**: We **abandoned the neural network** in favor of Linear Algebra (SVD). This is technically a "vector-to-vector" mapping, but it relies on the analytic solution (Orthogonal Procrustes) rather than the neural `Vec2VecModel` architecture used in previous attempts.

## 📊 Progress Summary

| Attempt | Technique | Top-1 Accuracy | Status |
|---------|-----------|----------------|--------|
| V1 | Neural Vec2Vec (Multi-Space) | - | ❌ Failed (instability) |
| V2 | Unsupervised Neural Vec2Vec | - | ❌ Failed (isomorphism gap) |
| V3 | Linear Procrustes + Anchors | 22% | ✅ Baseline |
| V4 | Linear + CSLS | 15% | ⚠️ Negative result |
| V5 | Linear + 10x Data | **24.53%** | ✅ Scaled baseline |
| V6 | BERT Contextual | 0.47% | ❌ Failed (tokenization) |
| V7 | FastText 768d | **29.10%** | ✅ **Current SOTA** |
| V8 | Coptic Bridge | 28.2% | ⚠️ Negative result |

**Key Insight**: Simple linear methods with good data outperform complex neural architectures for low-resource ancient language alignment.

## 📂 Project Structure

### [Attempt 1: The Translation Bridge (`heiro_v1`)](./heiro_v1)
*   **Technique**: **Neural Vec2Vec (Multi-Space)**.
*   **Strategy**: We attempted to bridge the Hieroglyphic space and the English space using a third "pivot" space (German). The model tried to learn a neural mapping that satisfied all three constraints simultaneously.
*   **Outcome**: The complexity of aligning three moving targets with a neural network led to instability.

### [Attempt 2: The Purist Geometric Mapping (`heiro_v2`)](./heiro_v2)
*   **Technique**: **Unsupervised Neural Vec2Vec**.
*   **Strategy**: A "clean room" experiment. We trained independent spaces and used an adversarial neural network (similar to MUSE/GANs) to force them to align based purely on their geometric density, with **zero** supervision.
*   **Outcome**: Failed due to the "Isomorphism Gap." The shape of Ancient Egyptian concepts is too different from Modern English for a neural network to guess the alignment without clues.

### [Attempt 3: Anchor-Guided Alignment (`heiro_v3`)](./heiro_v3)
*   **Technique**: **Linear Vec2Vec (Orthogonal Procrustes)**.
*   **Strategy**: The "Goldilocks" solution. Instead of a black-box neural network, we used the analytic solution to the vec2vec problem. We identified ~1,300 "Anchor" points and calculated the optimal linear rotation to lock them in place.
*   **Outcome**: The most successful iteration (~22% accuracy). It proves that the vector spaces *can* be aligned, but a simple linear rotation with supervision works better than complex unsupervised neural networks for this specific data.

### [Attempt 4: CSLS Refinement (`heiro_v4`)](./heiro_v4)
*   **Technique**: **Linear Vec2Vec + CSLS**.
*   **Strategy**: We attempted to refine the V3 results by replacing Nearest Neighbor search with **Cross-Domain Similarity Local Scaling (CSLS)**. This metric is designed to reduce the "hubness" problem (where common words dominate the nearest neighbor lists).
*   **Outcome**: Interestingly, accuracy *dropped* to ~15%. This negative result suggests that for small, sparse datasets (like our specialized Egyptology corpus), the "hubness" correction might be too aggressive, or that the embedding space lacks the density required for CSLS to be effective. It highlights the difference between "Big Data" NLP techniques and "Low Resource" realities.

### [Attempt 5: Scaled Corpus Alignment (`heiro_v5_getdata`)](./heiro_v5_getdata)
*   **Technique**: **Linear Vec2Vec (Orthogonal Procrustes) + 10x Data**.
*   **Strategy**: Building on V3's success, we assembled a **combined corpus of 104,000 texts** (10x larger than V3) by integrating three major datasets: TLA, Ramses Online, and BBAW. We extracted **8,541 high-confidence anchor pairs** using co-occurrence analysis, translated them from German to English, and trained FastText embeddings on the full hieroglyphic corpus. The alignment used the same Procrustes method as V3 but with significantly more data and anchors.
*   **Outcome**: **24.53% accuracy** - a statistically significant **11.5% relative improvement** over V3's 22%. This proves the vec2vec hypothesis scales with data quality and quantity. Perfect hits on key deities (Osiris: 61.5%, Horus: 62.1%, Re: 54.6%) and semantic concepts (god: 61.3%, water: 57.7%) demonstrate meaningful semantic alignment between Ancient Egyptian and Modern English vector spaces.

### [Attempt 6: BERT Contextual Embeddings (`heiro_v6_BERT`)](./heiro_v6_BERT)
*   **Technique**: **BERT Contextual Embeddings**.
*   **Strategy**: Attempted to improve on V5 by using BERT's context-aware representations instead of static FastText vectors. The hypothesis was that contextual embeddings would better capture polysemy in Ancient Egyptian.
*   **Outcome**: **0.47% accuracy** ❌ - Complete failure. BERT's WordPiece tokenizer fragmented hieroglyphic transliteration into meaningless subwords, destroying semantic information. This negative result proved that modern pre-trained models don't transfer to ancient languages, and simpler models work better for low-resource specialized domains.

### [Attempt 7: FastText + Visual Embeddings (768d) (`heiro_v7_FastTextVisual`)](./heiro_v7_FastTextVisual)
*   **Technique**: **Large-Scale FastText (768d) + Visual Fusion**.
*   **Strategy**: Returned to FastText but with **2.56x larger embeddings** (768d instead of 300d) trained on hieroglyphic transliteration. Combined with 768d visual embeddings from ResNet-50 for a symmetrical 1536d fusion architecture. Used Ridge regression to align to 300d English GloVe space.
*   **Outcome**: **29.10% accuracy** ✅ - **New state-of-the-art!** An **18.6% relative improvement** over V5's 24.53%. Achieved 36.57% Top-5 accuracy and 41.19% Top-10 accuracy with 78.4% anchor coverage (6,700/8,541 valid pairs). This proves that larger embedding dimensions capture richer semantic relationships for Ancient Egyptian, and simple architectures outperform complex models for low-resource languages. Note: Visual features currently unused due to vocabulary mismatch (transliteration vs glyphs).

### [Attempt 8: Coptic Bridge (`heiro_v8_use_coptic`)](./heiro_v8_use_coptic)
*   **Technique**: **Coptic Bridge Alignment**.
*   **Strategy**: Used Coptic (the direct descendant of Ancient Egyptian) as a bridge to expand the anchor dictionary. Extracted 368 new Egyptian-English anchors from ThotBank's Egyptian-Coptic cognates, increasing coverage from 8,541 to 8,909 anchors (+4.31%).
*   **Outcome**: **28.16% accuracy** ⚠️ - Slight regression (-0.94%) from V7's 29.10%. Despite adding more anchors, the Coptic-derived meanings introduced semantic drift (1,000+ year gap) and domain mismatch (biblical vs literary texts). Key learning: **Etymology ≠ Semantics** - cognates don't guarantee identical vector space positions. Quality > Quantity for anchor dictionaries.

## 🚀 Getting Started

We recommend starting with **`heiro_v7_FastTextVisual`** as it represents the current state-of-the-art, achieving **29.10% accuracy** with 768d FastText embeddings.

For understanding the data collection and baseline methodology, **`heiro_v5_getdata`** provides comprehensive documentation of the corpus assembly and anchor extraction process.

For a simpler introduction to the core alignment technique, **`heiro_v3`** offers the most accessible starting point.

### Prerequisites
*   Python 3.8+
*   `gensim`, `numpy`, `scikit-learn`, `pandas`, `jupyter`

### Usage
Navigate to `heiro_v5_getdata` and launch the Jupyter Notebooks to follow the step-by-step replication of our results.

```bash
cd heiro_v5_getdata
jupyter notebook
```

## 📚 Data Sources
*   **Thesaurus Linguae Aegyptiae (TLA)**: The primary source for Hieroglyphic transliterations and German translations.
*   **Ramses Online**: Additional hieroglyphic texts with German translations (V5).
*   **Berlin-Brandenburg Academy (BBAW)**: Large-scale hieroglyphic corpus from HuggingFace (V5).
*   **Wikipedia**: Used for training the Modern English embedding space.
*   **GloVe**: Pre-trained English word embeddings (V5).


