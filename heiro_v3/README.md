# Heiroglyphy V3: Anchor-Guided Alignment

## 🏺 Can AI Translate Ancient Egyptian using Math?

**Heiroglyphy** is a research project exploring whether we can translate Ancient Egyptian Hieroglyphs into Modern English using **Vector Space Alignment**. 

Instead of relying on a traditional dictionary, we try to align the "geometry" of the two languages. The idea is simple: if the relationship between "King" and "Queen" is a specific direction in English, that same direction should exist between the Hieroglyphs for "Pharaoh" and "Great Royal Wife".

## 🚀 How It Works

This version (V3) uses an approach called **Anchor-Guided Alignment**:

1.  **Embeddings**: We train AI models to convert words into vectors (lists of numbers).
    *   **Hieroglyphs**: We use *FastText* to understand the sub-structure of transliterated words (e.g., `nfr`, `nfr.t`).
    *   **English**: We use *Word2Vec* to understand the semantic meaning of English words.
2.  **Anchors**: We identify a small set of "known" translations (about 1,300 pairs) to serve as our guide stars.
3.  **Alignment**: We use Linear Algebra (Procrustes Analysis) to "rotate" the entire Hieroglyphic universe until it overlaps with the English universe, locking our anchors in place.
4.  **Translation**: Once aligned, we can look at where *unknown* Hieroglyphic words land in the English space to guess their meaning.

## 📂 Project Structure

The project is broken down into three educational Jupyter Notebooks:

*   **`01_data_preparation.ipynb`**: 
    *   Loads the raw TLA (Thesaurus Linguae Aegyptiae) dataset.
    *   Cleans the text and extracts our "Anchor" dictionary.
*   **`02_embedding_training.ipynb`**:
    *   Trains the FastText and Word2Vec models from scratch.
    *   Visualizes how the models learn concepts like "God" or "King" independently.
*   **`03_alignment_and_analysis.ipynb`**:
    *   Performs the mathematical alignment.
    *   Evaluates the accuracy (currently ~22% Top-10 accuracy).
    *   **The Fun Part**: A translation function that lets you input Hieroglyphs and see what the AI thinks they mean.

## 🛠️ Getting Started

### Prerequisites
You need Python 3 and the following libraries:
```bash
pip install gensim numpy scikit-learn pandas tqdm
```

### Running the Project
1.  Open the notebooks in order (01 -> 02 -> 03).
2.  Run the cells to see the process unfold.
3.  In Notebook 3, try the `translate()` function with your own words!

## 📊 Results

This approach shows that even with a small dataset (~12k sentences), we can achieve statistically significant alignment. While it won't replace a human Egyptologist yet, it successfully maps core concepts:

*   `pr-aa` (Pharaoh) $\rightarrow$ maps near `subjects`, `writing`, `sacrifice`.
*   `nfr` (Good/Beautiful) $\rightarrow$ maps near `wemetetka` (a specific name/concept), `worried` (noise), but captures the general "adjective" cluster.

## 📜 Credits
*   Data Source: **Thesaurus Linguae Aegyptiae (TLA)**
*   Methodology: Inspired by **MUSE (Multilingual Unsupervised and Supervised Embeddings)**.
