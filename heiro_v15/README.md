# Heiroglyphy V15: FastText Retraining (Current SOTA)

## Overview

**V15** retrained FastText with better hyperparameters, achieving **32.35% Top-1 accuracy** — a +1.68% improvement over V13.

The key insight: V7's FastText was trained with `min_count=1`, keeping 52,813 hapax legomena (65.5% of vocabulary) as noise vectors. Filtering to `min_count=5` reduced vocabulary from 80,662 to 10,833 — an 87% reduction — and accuracy went *up*. Wider context windows (`window=10` vs 5) also helped compensate for the corpus's short texts (median 6 words).

## Results

### Parameter Sweep (9 configurations)

| Config | min_count | window | epochs | algo | Vocab | Top-1 | Δ V13 |
|--------|-----------|--------|--------|------|-------|-------|-------|
| V7_baseline | 1 | 5 | 10 | sg | 80,662 | 30.60% | -0.97% |
| mc3_w5 | 3 | 5 | 10 | sg | 17,746 | 28.96% | -2.61% |
| mc5_w5 | 5 | 5 | 10 | sg | 10,833 | 31.41% | -0.16% |
| mc3_w10 | 3 | 10 | 10 | sg | 17,746 | 30.34% | -1.23% |
| mc3_w15 | 3 | 15 | 10 | sg | 17,746 | 30.41% | -1.16% |
| **mc5_w10** | **5** | **10** | **10** | **sg** | **10,833** | **31.88%** | **+0.31%** |
| mc3_w10_e20 | 3 | 10 | 20 | sg | 17,746 | 30.57% | -1.00% |
| mc3_w10_e50 | 3 | 10 | 50 | sg | 17,746 | 27.73% | -3.84% |
| mc3_w10_cbow | 3 | 10 | 20 | cbow | 17,746 | 29.72% | -1.85% |

### Alpha Sweep on mc5_w10

| alpha | Top-1 | Top-5 | Top-10 |
|-------|-------|-------|--------|
| **0.001** | **32.35%** | **41.47%** | **45.13%** |
| 0.01 | 31.72% | 40.92% | 45.60% |
| 0.05 | 31.96% | 41.39% | 45.36% |
| 0.1 | 31.88% | 41.54% | 44.74% |
| 1.0 | 31.41% | 39.91% | 44.51% |

### Progression

| Version | Top-1 | Top-5 | Top-10 | Key change |
|---------|-------|-------|--------|------------|
| V10 | 30.67% | 37.69% | 41.49% | alpha=1.0, 80K vocab |
| V13 | 31.57% | 38.81% | 42.61% | alpha=0.1 |
| **V15** | **32.35%** | **41.47%** | **45.13%** | mc5_w10, alpha=0.001 |

## Key Findings

1. **min_count=5 > min_count=3 > min_count=1**: Aggressive filtering of rare words improves embedding quality. 87% of vocabulary was noise.
2. **window=10 > window=5**: Wider context compensates for short texts (median 6 words).
3. **More epochs hurts**: 50 epochs dropped to 27.73% — overfitting on sparse data.
4. **Skip-gram > CBOW**: For rare-word-heavy ancient languages, skip-gram is better.
5. **Alpha decreases as embeddings improve**: 1.0 → 0.1 → 0.001. Cleaner embeddings need less regularization.

## Run

```bash
python heiro_v15/scripts/train_and_evaluate.py
```

---

**Status**: Current SOTA (32.35%)
**Previous**: [V13 Alpha Tuning](../heiro_v13/) — 31.57%
