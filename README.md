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

## 🚀 Getting Started

We recommend starting with **`heiro_v3`** as it represents the most mature and reproducible version of the research.

### Prerequisites
*   Python 3.8+
*   `gensim`, `numpy`, `scikit-learn`, `pandas`, `jupyter`

### Usage
Navigate to `heiro_v3` and launch the Jupyter Notebooks to follow the step-by-step replication of our results.

```bash
cd heiro_v3
jupyter notebook
```

## 📚 Data Sources
*   **Thesaurus Linguae Aegyptiae (TLA)**: The primary source for Hieroglyphic transliterations and German translations.
*   **Wikipedia**: Used for training the Modern English embedding space.

---
*Research conducted by [Your Name/Team]*
