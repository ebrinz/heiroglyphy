# Heiroglyphy V5: Scaled Corpus Alignment

## Overview

**V5** represents the most successful iteration of the Heiroglyphy project, achieving **24.53% accuracy** in aligning Ancient Egyptian hieroglyphic and Modern English vector spaces - an **11.5% relative improvement** over V3's 22% baseline.

Building on V3's proven Orthogonal Procrustes approach, V5 scales the methodology with:
- **10x larger corpus**: 104,000 texts (vs. V3's 12,000)
- **6x more anchors**: 8,541 high-confidence pairs (vs. V3's 1,362)
- **Three integrated datasets**: TLA, Ramses Online, and BBAW

## Key Results

### Accuracy
- **Overall**: 24.53% (1,833 / 7,471 test anchors)
- **Improvement**: +2.53 percentage points over V3 (22.0%)
- **Statistical significance**: Robust with 7,471 test cases

### Perfect Hits (Top-1 Matches)
| Hieroglyphic | Expected | Score | Status |
|--------------|----------|-------|--------|
| `wsjr` | osiris | 61.5% | ✓ Perfect |
| `ḥr,w` | horus | 62.1% | ✓ Perfect |
| `rꜥw` | re | 54.6% | ✓ Perfect |
| `nṯr` | god | 61.3% | ✓ Perfect |
| `zꜣ` | son | 47.4% | ✓ Perfect |
| `mw` | water | 57.7% | ✓ Perfect |

## Project Structure

```
heiro_v5_getdata/
├── data/
│   ├── raw/                    # Source datasets
│   │   ├── tla_raw.json       # TLA corpus (12k texts)
│   │   ├── ramses_raw.json    # Ramses Online (4k texts)
│   │   └── bbaw_huggingface.parquet  # BBAW corpus (100k texts)
│   └── processed/              # Derived data
│       ├── german_anchors.pkl  # 8,541 hieroglyphic↔German pairs
│       ├── english_anchors.pkl # 8,541 hieroglyphic↔English pairs
│       ├── hieroglyphic_corpus.txt  # Training corpus
│       ├── hieroglyphic_vectors.kv  # FastText embeddings
│       ├── procrustes_matrix.npy    # Transformation matrix
│       └── alignment_results.json   # Final accuracy metrics
├── notebooks/
│   ├── 01_dataset_assembly.ipynb       # Phase 1: Data integration
│   ├── 02_bert_dataset_exploration.ipynb
│   ├── 03_v5_data_strategy.ipynb
│   ├── 04_anchor_extraction.ipynb      # Phase 2: Anchor extraction
│   ├── 05_translate_anchors.ipynb      # Phase 2: German→English
│   ├── 06_embedding_training.ipynb     # Phase 3: FastText training
│   └── 07_procrustes_alignment_optimized.ipynb  # Phase 4: Alignment
└── scrapers/
    ├── tla_scraper.py          # TLA data collection
    └── ramses_scraper.py       # Ramses Online scraper
```

## Methodology

### Phase 1: Data Assembly
1. **TLA Dataset**: 12,773 texts from HuggingFace
2. **Ramses Online**: 4,000+ texts via custom scraper
3. **BBAW Dataset**: 100,736 texts from HuggingFace
4. **Deduplication**: Reduced to 104,000 unique texts

### Phase 2: Anchor Extraction
1. **Co-occurrence Analysis**: Built hieroglyphic↔German matrix
2. **Confidence Filtering**: Extracted 8,541 high-probability pairs
3. **Translation**: German→English using DeepL + dictionary fallback

### Phase 3: Embedding Training
1. **Hieroglyphic**: FastText (300d) on 104k corpus
2. **English**: Pre-trained GloVe (300d, 6B tokens)

### Phase 4: Procrustes Alignment
1. **Anchor Vectors**: Extracted 7,471 valid pairs (87.5% coverage)
2. **SVD Transformation**: Computed optimal rotation matrix
3. **Evaluation**: Vectorized top-1 accuracy on test set

## Data Sources

- **Thesaurus Linguae Aegyptiae (TLA)**: https://thesaurus-linguae-aegyptiae.de/
- **Ramses Online**: http://ramses.ulg.ac.be/
- **Berlin-Brandenburg Academy (BBAW)**: HuggingFace dataset
- **GloVe Embeddings**: Stanford NLP pre-trained vectors

## Getting Started

### Prerequisites
```bash
pip install gensim numpy scipy scikit-learn pandas tqdm jupyter
```

### Quick Start
```bash
cd notebooks
jupyter notebook
```

Start with `07_procrustes_alignment_optimized.ipynb` to see the final results, or begin with `01_dataset_assembly.ipynb` to follow the full pipeline.

## Key Findings

1. **Data scaling works**: 10x corpus → 11.5% accuracy improvement
2. **Anchor quality matters**: 8.5k high-confidence anchors > 1.3k
3. **Linear alignment is sufficient**: Procrustes outperforms complex neural methods
4. **Semantic clusters align**: Egyptian deities map to their English names
5. **Context-independent limitations**: Single-word alignment caps at ~25% due to polysemy

## Ethical Considerations

- **Respect robots.txt**: All scrapers honor site policies
- **Rate limiting**: 1-2 requests/second maximum
- **Attribution**: All sources properly credited
- **Academic use**: Research purposes only

## Future Directions (V6)

1. Normalize transliteration variants (ḥr.w = ḥr,w)
2. Train larger embeddings (500d instead of 300d)
3. Use context-aware embeddings (BERT-style)
4. Add hieroglyph visual features

---

**Research conducted as part of the Heiroglyphy project**  
*Exploring computational methods for Ancient Egyptian translation*
