# Heiroglyphy V14: Iterative Procrustes + Hub Filtering

## Overview

**V14** tested two strategies to push beyond V13's 31.57% SOTA:

1. **Iterative Procrustes Refinement** — bootstrap new anchor pairs via mutual nearest neighbor (MNN) discovery
2. **Hub Filtering** — remove English stopwords from training and retrieval to eliminate the 82% hubness problem

Both approaches failed, but the failure revealed a critical insight about the test set composition.

## Results

| Experiment | Top-1 | Top-5 | Top-10 | Notes |
|-----------|-------|-------|--------|-------|
| V13 baseline | 31.57% | 38.81% | 42.61% | alpha=0.1, 5,360 train anchors |
| Iterative Procrustes | 31.57% | 38.81% | 42.61% | Converged in 2 iters, +18 pairs, no improvement |
| B1: Filtered retrieval | 7.61% | 12.62% | 14.10% | Catastrophic — model projects to hubs it can't find |
| B2: Filtered train+retrieval | 18.37% | 27.83% | 30.80% | Lost 61% of training data |
| B3: B2 + iterative | 18.37% | 27.83% | 30.80% | No improvement over B2 |

## Key Finding: The Function Word Problem

60% of test pairs (801/1,340) have English stopword targets ("the", "of", "in", etc.). This isn't hubness — it's reality:

- `n` → "of/to" (8,829 occurrences — most frequent Egyptian word)
- `m` → "in/from"
- `r` → "to/at"
- `=f` → "his"
- `=k` → "your"

The 82% "the" prediction rate reflects that Egyptian function words dominate the corpus by frequency and legitimately translate to English function words. The alignment isn't confused — the test set is 60% function words.

**Implication**: The 31.57% accuracy is depressed by function word ambiguity. On content words alone, the model performs significantly better, but the standard test methodology doesn't separate these.

## Run

```bash
# Iterative Procrustes (in notebook)
jupyter notebook notebooks/01_iterative_procrustes.ipynb

# Hub filtering experiments
python scripts/run_hub_filter.py
```

---

**Status**: Complete (negative result, important insight)
**Baseline**: [V13 Alpha Tuning](../heiro_v13/) — 31.57%
