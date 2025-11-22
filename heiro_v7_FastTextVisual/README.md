# Heiroglyphy V7: FastText + Visual Embeddings (768d)

## Overview

**V7** successfully improved on V5's 24.53% accuracy by training larger FastText embeddings (768d instead of 300d) on hieroglyphic transliteration. This approach achieved **29.10% Top-1 accuracy**, representing an **18.6% relative improvement** over the V5 baseline.

## Key Innovation

Instead of using complex contextual models (V6's failed BERT approach), V7 returns to the proven FastText architecture but with:
- **Larger embeddings**: 768d vectors (matching visual embedding dimensions)
- **Transliteration-based training**: Using the `transcription` column from BBAW data
- **Symmetrical fusion architecture**: 768d text + 768d visual = 1536d total

## Results Analysis

### Performance Breakdown
- **Top-1 Accuracy**: 29.10% (Text-Only)
- **Top-5 Accuracy**: 36.57%
- **Top-10 Accuracy**: 41.19%
- **Anchor Coverage**: 78.4% (6,700/8,541)

### Comparison to Baselines
| Metric | V5 Baseline | V6 BERT | **V7 (This)** |
|--------|-------------|---------|---------------|
| **Top-1 Accuracy** | 24.53% | 0.47% | **29.10%** ✅ |
| **Improvement** | - | - | **+4.57%** (Absolute) |
| **Relative Gain** | - | - | **+18.6%** |

### Why V7 Succeeded (Where V6 Failed)
1. **Vocabulary Alignment**: V7 trained on `transcription` (e.g., "nfr"), which matches the anchor dictionary. V6 trained on MdC codes ("F35"), which had almost zero overlap with the anchors (0.18% coverage).
2. **Model Simplicity**: FastText's subword information captured morphological variants better than BERT's tokenization for this low-resource language.
3. **Dimension Scaling**: Increasing dimensions from 300d (V5) to 768d (V7) provided more capacity for semantic separation.

### The "Silent Failure" of Visuals
Despite the "Fused" architecture, V7 is effectively **text-only**.
- **Issue**: Visual embeddings were keyed by Unicode/Gardiner codes (e.g., `U+13000`), while FastText used transliteration (`nfr`).
- **Result**: The fusion step found **0 matches**, resulting in zero-vectors for the visual component.
- **Implication**: The 29.10% accuracy is achieved purely by the 768d FastText model. This sets a strong baseline for V9, where we will properly integrate visuals.

## Approach

### Pipeline
1. **Data Cleaning**: Extract transliteration from BBAW parquet (`transcription` column)
2. **FastText Training**: Train 768d skip-gram embeddings (10 epochs, window=5)
3. **Visual Fusion**: Concatenate FastText (768d) + Visual (768d) = 1536d
4. **Alignment**: Linear regression (Ridge) to map 1536d → 300d English GloVe space
5. **Evaluation**: Test on 8,541 anchor pairs

### Technical Details
- **Model**: FastText skip-gram (sg=1)
- **Dimensions**: 768d (2.56x larger than V5's 300d)
- **Training Data**: 100,729 sentences from BBAW corpus
- **Vocabulary**: 80,662 unique transliteration tokens
- **Training Time**: ~71 seconds (10 epochs)
- **Alignment Method**: Ridge regression (α=1.0)

### Why It Works
1. **Larger Embedding Space**: 768d vectors capture more nuanced semantic relationships than 300d
2. **Transliteration Vocabulary Match**: Training on `transcription` ensures vocabulary alignment with anchors (78.4% coverage vs V5's 0.18%)
3. **Skip-Gram Architecture**: FastText's subword-aware skip-gram handles Egyptian morphology well (captures variants like nṯr, nṯrw, nṯr.j)

## Data
- **Corpus**: BBAW Hieroglyphic Corpus (`data/raw/bbaw_huggingface.parquet`)
  - 100,729 sentences with transliteration
  - 789,159 tokens
  - 80,662 unique words
- **Anchors**: `data/processed/anchors.json` (8,541 pairs, 6,700 valid)
- **Visual Embeddings**: `data/processed/visual_embeddings_768d.pkl` (ResNet-50, unused)
- **English Embeddings**: GloVe 6B 300d

## Notebooks
- **[01_complete_pipeline.ipynb](notebooks/01_complete_pipeline.ipynb)**: Full pipeline from data cleaning to evaluation
  - Includes all 4 scripts inline for learning
  - Shows intermediate results and analysis
  - Documents the vocabulary mismatch discovery

## Next Steps
- **V8**: Attempted Coptic bridge (28.16% accuracy - slight regression).
- **V9**: Fix the visual pipeline to properly fuse ResNet-50 features (Goal: >30%).

