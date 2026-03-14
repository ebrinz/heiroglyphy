# Heiroglyphy V10: Vocabulary Refinement (Current SOTA)

## Overview

**V10** achieved the project's best accuracy — **30.67% Top-1** — through vocabulary normalization and lexicon integration, building on V9's 1536d fused architecture. Three sub-iterations explored different strategies for improving the transliteration-to-Gardiner mapping.

## Results

| Version | Technique | Top-1 | Top-5 | Top-10 | Visual Match Rate |
|---------|-----------|-------|-------|--------|-------------------|
| V10.0 | Gardiner mapping (Wikipedia) | **30.67%** | 37.69% | 41.49% | 0.08% |
| V10.1 | + Vocabulary normalization | 30.07% | 38.13% | 41.64% | 1.58% |
| V10.2 | + HamdiJr lexicon (4,282 entries) | 30.30% | 37.91% | 41.87% | 0.63% |

**V10.0 remains SOTA.** The normalization and lexicon approaches improved visual match rates but slightly degraded Top-1 accuracy — likely because the new mappings introduced noise that offset the coverage gains.

### Comparison to Baselines

| Metric | V7 (Text-Only) | V9 (1536d) | **V10.0 (This)** |
|--------|-----------------|------------|-------------------|
| **Top-1** | 29.10% | 30.52% | **30.67%** |
| **Top-5** | 36.57% | 37.54% | **37.69%** |
| **Top-10** | 41.19% | 41.79% | **41.49%** |

## Approach

V10 uses the same architecture as V9 (768d FastText + 768d Visual = 1536d → Ridge Regression → 300d GloVe) but focuses on improving the input data:

1. **V10.0**: Scraped Wikipedia for 230 Gardiner code mappings to bridge the transliteration/Gardiner vocabulary gap
2. **V10.1**: Normalized vocabulary (suffix stripping, parentheses removal) to increase visual match coverage from 0.08% to 1.58%
3. **V10.2**: Integrated HamdiJr lexicon (4,282 entries) for direct transliteration → Gardiner lookup

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_fusion_v10.ipynb` | Baseline fusion with Wikipedia Gardiner mapping |
| `02_fusion_v10.1_normalized.ipynb` | Vocabulary normalization experiment |
| `03_fusion_v10.2_lexicon.ipynb` | HamdiJr lexicon integration |

## Scripts

| Script | Purpose |
|--------|---------|
| `fetch_gardiner_mapping.py` | Scrape Gardiner codes from Wikipedia |
| `parse_lexicon.py` | Parse HamdiJr Lexicon.txt |
| `merge_mappings.py` | Combine multiple mapping sources |
| `normalize_vocab.py` | Vocabulary normalization utilities |
| `analyze_vocab.py` | Vocabulary coverage statistics |
| `find_top_missing.py` | Identify highest-impact unmapped words |
| `add_cleanup.py` / `add_cleanup_all.py` | Data cleaning utilities |

## Data

| File | Description |
|------|-------------|
| `gardiner_mapping.json` | Wikipedia-scraped Gardiner → transliteration (230 codes) |
| `Lexicon.txt` | Raw HamdiJr lexicon |
| `lexicon_trans_to_codes.json` | Transliteration → Gardiner codes |
| `lexicon_code_to_trans.json` | Gardiner codes → transliteration |
| `merged_mapping.json` | Combined mapping from all sources |
| `manual_mappings.json` | Hand-curated corrections |

## Key Findings

1. **Diminishing returns on mapping quality**: More visual matches didn't translate to better accuracy — the new mappings introduced enough noise to offset coverage gains
2. **V10.0's Wikipedia mapping was surprisingly effective**: A simple 230-code scrape produced the best result
3. **The visual match bottleneck remains**: Even at 1.58% (V10.1), coverage is far too low to materially impact alignment

## Next Steps

See [TODO.md](./TODO.md) for the roadmap to 50% accuracy, including multi-glyph decomposition, architecture improvements, and data augmentation strategies.

---

**Status**: Current SOTA (30.67%)
**Previous**: [V9 Visual Features](../heiro_v9_use_visuals_again/) — 30.52%
**Next Attempt**: [V11 MLP Training](../heiro_v11/) — 28.76% (regression)
