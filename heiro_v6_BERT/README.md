# Heiroglyphy V6: Context-Aware Alignment

## Overview

**V6** builds on V5's success (24.53% accuracy) by addressing its key limitation: **context independence**. While V5 proved that linear alignment works at scale, single-word embeddings cannot capture the polysemous nature of Ancient Egyptian, where the same word has different meanings in different contexts.

V6 explores four enhancement strategies to push beyond the ~25% ceiling:

1. **Transliteration Normalization**: Merge variant spellings (ḥr.w = ḥr,w)
2. **Larger Embeddings**: Train 500d vectors instead of 300d
3. **Context-Aware Embeddings**: Use BERT-style contextual representations
4. **Visual Features**: Incorporate hieroglyph visual similarity

## Goals

- **Target Accuracy**: >30% (25% relative improvement over V5)
- **Primary Focus**: Context-aware embeddings (BERT/mBERT)
- **Secondary**: Transliteration normalization and embedding size experiments

## Project Structure

```
heiro_v6_BERT/
├── data/
│   ├── raw/                    # Reuse V5 datasets (symlinks)
│   │   ├── tla_raw.json       
│   │   ├── ramses_raw.json    
│   │   └── bbaw_huggingface.parquet
│   └── processed/              # V6-specific processed data
│       ├── normalized_corpus.txt      # Transliteration variants merged
│       ├── contextual_embeddings/     # BERT embeddings
│       └── visual_features/           # Hieroglyph image features
├── notebooks/
│   ├── 01_transliteration_normalization.ipynb
│   ├── 02_embedding_size_experiments.ipynb
│   ├── 03_bert_contextual_embeddings.ipynb
│   └── 04_visual_feature_extraction.ipynb
├── models/
│   └── (BERT models, custom architectures)
└── README.md
```

## Methodology

### Phase 1: Transliteration Normalization
**Problem**: V5 treats `ḥr.w` and `ḥr,w` as different words, fragmenting the embedding space.

**Solution**:
1. Analyze transliteration variants in corpus
2. Create normalization rules (. → , or remove punctuation)
3. Retrain FastText on normalized corpus
4. Measure accuracy improvement

**Expected Impact**: +2-3% accuracy

### Phase 2: Embedding Size Experiments
**Problem**: 300d may be insufficient for capturing Egyptian semantic complexity.

**Solution**:
1. Train FastText with 500d vectors
2. Use GloVe 300d → 500d (or train custom English embeddings)
3. Compare 300d vs 500d alignment accuracy

**Expected Impact**: +1-2% accuracy

### Phase 3: Context-Aware Embeddings (Primary Focus)
**Problem**: Single-word embeddings ignore context. "ḥr" can mean "face," "upon," or "Horus" depending on usage.

**Solution**:
1. **Option A: Multilingual BERT (mBERT)**
   - Fine-tune mBERT on hieroglyphic corpus
   - Extract contextual embeddings for each word occurrence
   - Align with English BERT embeddings
   
2. **Option B: Custom Transformer**
   - Train small transformer on hieroglyphic corpus
   - Use attention mechanisms to capture context
   - Align with pre-trained English transformer

**Expected Impact**: +5-10% accuracy (biggest potential gain)

### Phase 4: Visual Features (Exploratory)
**Problem**: Hieroglyphs are visual symbols; shape similarity may aid alignment.

**Solution**:
1. Extract hieroglyph images from Gardiner's sign list
2. Use CNN to extract visual features
3. Combine visual + textual embeddings
4. Test if visual similarity improves alignment

**Expected Impact**: +1-3% accuracy

## Data Sources

- **Corpus**: Reuse V5's 104k texts (TLA + Ramses + BBAW)
- **Anchors**: Reuse V5's 8,541 English anchor pairs
- **BERT Models**: HuggingFace transformers (mBERT, XLM-R)
- **Hieroglyph Images**: Gardiner's sign list, Unicode charts

## Technical Stack

### Core Libraries
- `transformers`: BERT/mBERT models
- `torch`: Deep learning framework
- `gensim`: FastText baseline
- `scikit-learn`: Alignment and evaluation
- `pillow`, `opencv`: Image processing (Phase 4)

### Installation
```bash
pip install transformers torch gensim scikit-learn pandas numpy tqdm jupyter
pip install pillow opencv-python  # For visual features
```

## Success Metrics

| Metric | V5 Baseline | V6 Target | Stretch Goal |
|--------|-------------|-----------|--------------|
| Overall Accuracy | 24.53% | >30% | >35% |
| Deity Accuracy | ~60% | >70% | >80% |
| Common Words | ~50% | >60% | >70% |

## Key Hypotheses

1. **Context matters**: Contextual embeddings will significantly outperform static embeddings
2. **Normalization helps**: Merging variants will improve embedding quality
3. **Size has limits**: 500d may help, but context is more important than dimensionality
4. **Visual is complementary**: Visual features alone won't help much, but may boost combined models

## Timeline

- **Week 1**: Transliteration normalization + 500d experiments
- **Week 2-3**: BERT contextual embeddings (primary focus)
- **Week 4**: Visual features (if time permits)
- **Week 5**: Evaluation and documentation

## Expected Challenges

1. **Computational cost**: BERT is much slower than FastText
2. **Context extraction**: Need to handle sentence boundaries properly
3. **Alignment complexity**: Contextual embeddings are dynamic, not static
4. **Data sparsity**: 104k texts may be small for transformer training

## Fallback Plan

If BERT proves too complex/slow:
1. Use **sentence embeddings** instead of word embeddings
2. Average contextualized representations across corpus
3. Fall back to improved FastText (normalized + 500d)

---

**Status**: Planning phase  
**Next Step**: Create `01_transliteration_normalization.ipynb`
