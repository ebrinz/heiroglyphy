# Heiroglyphy V4: CSLS Refinement

## 🎯 Goal
Attempt 4 builds directly on the success of Attempt 3 (Anchor-Guided Alignment).
We aimed to improve the translation accuracy by replacing the standard **Nearest Neighbor (NN)** search with **Cross-Domain Similarity Local Scaling (CSLS)**.

## 🧪 The Hypothesis
In high-dimensional vector spaces, certain "hub" words (like common stopwords) tend to appear as nearest neighbors to *everything*, polluting the translation results.
**CSLS** is a metric designed to mitigate this by penalizing words that are dense hubs. It asks: *"I know you are close to me, but are you also close to everyone else? If so, you are less likely to be my specific translation."*

## 📓 Notebooks
*   **`01_csls_alignment.ipynb`**:
    *   Loads the models trained in V3.
    *   Re-calculates the Procrustes Rotation.
    *   Implements the CSLS metric from scratch.
    *   Evaluates performance.

## 📊 Results
*   **Top-10 Accuracy**: **~15%** (Lower than V3's 22%).

### Analysis
Contrary to expectations, CSLS *reduced* accuracy in this specific context.
*   **Reason**: Our English embedding space (trained on TLA translations) is small and sparse compared to the massive datasets CSLS is usually used on (like Wikipedia). In a sparse space, "hubness" might actually be a useful signal for broad concepts, and penalizing it too aggressively removes valid matches.
*   **Conclusion**: Advanced techniques like CSLS require high-quality, dense embedding spaces to shine. For low-resource languages, simple Procrustes + NN might be more robust.

## 🛠️ Usage
```bash
jupyter notebook 01_csls_alignment.ipynb
```
