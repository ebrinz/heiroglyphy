# Heiroglyphy V13: Accuracy Push

## Overview

**V13** attempts to push accuracy from V10's 30.67% toward 40% through 5 incremental experiments on the existing pipeline. No new data sources or architectures — just better tuning, retrieval, and target vectors.

## Experiments

| # | Experiment | Hypothesis | Result |
|---|-----------|-----------|--------|
| 1 | Ridge Alpha CV | alpha=1.0 is suboptimal for ~86:1 param/sample ratio | TBD |
| 2 | Confidence-Weighted Ridge | High-confidence anchors should count more | TBD |
| 3 | CSLS Retrieval | Penalizing hub words improves Top-1 | TBD |
| 4 | 768d Ablation | 1536d zeros may only help via regularization | TBD |
| 5 | GloVe 840B Target | Larger English vocab recovers lost anchors | TBD |

## Run

```bash
# Experiments 1-4:
jupyter notebook notebooks/01_v13_experiments.ipynb

# Experiment 5 prerequisite (downloads ~2GB):
python scripts/download_glove_840b.py
```

## Results

See `results/experiment_results.json` after running the notebook.

---

**Status**: In Progress
**Baseline**: [V10 Vocabulary Refinement](../heiro_v10_refinement/) — 30.67%
