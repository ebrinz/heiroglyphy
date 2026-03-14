# V13: Accuracy Push from 30.67% toward 40%

**Date**: 2026-03-13
**Baseline**: V10 SOTA — 30.67% Top-1, 37.69% Top-5, 41.49% Top-10
**Goal**: Push Top-1 accuracy toward 40% through 5 incremental experiments

---

## Diagnosis

The V10 plateau at 30.67% has three root causes:

1. **Ridge alpha never tuned** — alpha=1.0 was chosen arbitrarily. With an ~86:1 parameter-to-sample ratio (1536 x 300 = 460,800 params, 5,360 training pairs), regularization strength is critical. Optimal alpha in high-dimensional low-sample regimes is often much larger than 1.0.
2. **Possible hubness in retrieval** — plain nearest-neighbor ranking may let common English words dominate neighbor lists. CSLS penalizes hub words. However, V4 showed CSLS caused a 7-point regression (22% → 15%) at 300d — this must be validated before assuming it helps at 768d/1536d scale.
3. **English target space is weak** — GloVe 6B 300d (400K vocab, 6B tokens) loses 21.6% of anchors to coverage gaps. GloVe 840B 300d (2.2M vocab, 840B tokens) should recover most.

Secondary factors: unweighted anchors (0.30-1.0 confidence treated equally), and an open question about whether the 1536d zero-padded architecture actually helps or just adds parameters.

---

## Architecture

```
heiro_v13/
├── README.md
├── notebooks/
│   └── 01_v13_experiments.ipynb    # Experiments 1-4 (interactive)
├── scripts/
│   └── download_glove_840b.py      # Experiment 5 (GloVe 840B download + checksum)
├── data/                           # Downloaded GloVe 840B
└── results/
    └── experiment_results.json     # All results
```

The notebook loads the V10 pipeline once (FastText, visual embeddings, anchors, GloVe 6B), then runs experiments sequentially. Each builds on the best result so far.

---

## Sanity Check (Cell 0)

Before any experiment, reproduce the V10 baseline:
- Load the exact V10 config (1536d, alpha=1.0, no weighting, plain NN)
- Verify Top-1 = 30.67% +/- 0.1%
- If not, halt and investigate — data files may have drifted from V10.1/V10.2 modifications
- Log: total anchors loaded, valid anchor count, train/test split sizes, FastText vocab size

---

## Experiments

### Experiment 1: Ridge Alpha Cross-Validation

**Hypothesis**: alpha=1.0 is suboptimal for the ~86:1 parameter-to-sample ratio.

**Method**:
- Load 1536d fused Egyptian embeddings + 300d GloVe anchor pairs
- 80/20 split, `random_state=42` (same as V10)
- `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0], cv=5)` with 5-fold CV (LOO is computationally expensive for multi-output regression; 5-fold is sufficient and fast)
- Evaluate Top-1/5/10 on held-out test set
- Log best alpha and per-fold variance

**Expected gain**: +1-2%

### Experiment 2: Confidence-Weighted Ridge

**Hypothesis**: High-confidence anchors (0.9) should matter more than low-confidence ones (0.3) during training.

**Anchor confidence distribution** (from anchors.json):
- Mean: 0.747, Std: 0.214
- Below 0.5: ~16% of anchors
- Above 0.8: ~47% of anchors
- Range: 0.30 to 1.00

**Method**:
- Same data, best alpha from Exp 1
- `Ridge(alpha=best_alpha).fit(X, Y, sample_weight=confidence_scores)`
- Compare Top-1/5/10 to unweighted

**Limitation**: Alpha was tuned without weighting. The interaction between alpha and confidence weighting is not explored. If Exp 2 shows improvement, a joint alpha+weighting search could extract more, but this is deferred to keep the experiment count manageable.

**Expected gain**: +0.5-1%

### Experiment 3: CSLS Retrieval

**Hypothesis**: Plain cosine NN suffers from hubness — common English words appear as nearest neighbors to many Egyptian words regardless of meaning.

**Prior result warning**: V4 applied CSLS to Procrustes-aligned 300d vectors and accuracy dropped 22% → 15%. V4's README attributed this to CSLS being "too aggressive for sparse data." Conditions differ now (Ridge vs Procrustes, 1536d vs 300d, 6,700 vs 1,300 anchors), but this is not guaranteed to help.

**Go/no-go check**: Before running CSLS, log the 20 most frequent top-1 predictions across the full test set. If stopwords/function words appear in fewer than 10% of test predictions, hubness is not the bottleneck and CSLS is unlikely to help. In that case, log the finding and skip to Exp 4.

**Method** (if go):
- Take best model from Exp 1 or 2
- Project all 80,662 Egyptian vectors to GloVe space
- Replace NN retrieval with CSLS(k=10): for each query, subtract mean similarity to its k nearest English neighbors
- Re-evaluate Top-1/5/10

**Expected gain**: +1-3% if hubness is confirmed, +0% or negative if not

### Experiment 4: Pure 768d Ablation

**Hypothesis**: The 1536d architecture (768d text + 768d zeros) may only help via implicit regularization. A properly tuned 768d Ridge might match it.

**Method**:
- Strip visual padding, use raw 768d FastText vectors
- Run its own `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0], cv=5)` on the 768d input — alpha from Exp 1 is not transferable because the input dimensionality and parameter-to-sample ratio are different (768 x 300 = 230,400 params, ratio ~43:1)
- Compare to 1536d at its own best alpha

**Expected gain**: Informational — resolves open question from DISCOVERIES.md. If 768d matches 1536d with tuned alpha, the zeros were only helping via regularization and we can simplify the pipeline.

### Experiment 5: GloVe 840B Target Space

**Hypothesis**: GloVe 840B (2.2M vocab, 840B tokens) will recover anchor coverage lost to GloVe 6B's 400K vocab and provide better target vectors.

**Method**:
- `download_glove_840b.py` fetches `glove.840B.300d.zip` (~2GB) to `heiro_v13/data/`
- Script verifies SHA256 checksum after download; raises exception on corruption
- **Retrain Ridge end-to-end** using GloVe 840B anchor vectors as targets (not just swap the retrieval vocabulary — the Ridge model must be trained against 840B geometry)
- Use best alpha and weighting config from Exps 1-2, re-tuning alpha via RidgeCV against 840B targets
- Log: anchor coverage delta (how many anchors gained/lost vs 6B), new valid anchor count
- Evaluate Top-1/5/10

**Expected gain**: +1-2%

---

## Dependency Chain

```
Sanity Check (reproduce 30.67%)
    ↓
Exp 1 (Ridge CV) → best_alpha_1536d
    ↓                       ↘
Exp 2 (Weighted)         Exp 4 (768d ablation, own RidgeCV)
    ↓                       (informational, parallel to 2/3)
Exp 3 (CSLS, with go/no-go check)
    ↓
Exp 5 (GloVe 840B, retrain end-to-end with best config from 1-3)
```

Exp 4 branches from Exp 1 independently — it does not depend on Exp 2 or 3.

---

## Evaluation Protocol

- **Test set**: 20% holdout, `random_state=42`, same as V10
- **Metrics**: Top-1, Top-5, Top-10 accuracy + delta vs V10 baseline
- **Sanity check**: Must reproduce 30.67% before any experiment runs
- **Results format**: JSON saved to `results/experiment_results.json`

```json
{
  "sanity_check": {
    "top1": null, "top5": null, "top10": null,
    "anchors_loaded": null, "valid_anchors": null,
    "train_size": null, "test_size": null,
    "passed": false
  },
  "baseline_v10": { "top1": 30.67, "top5": 37.69, "top10": 41.49 },
  "exp1_ridge_cv": {
    "best_alpha": null, "cv_folds": 5,
    "top1": null, "top5": null, "top10": null,
    "delta_v10": null
  },
  "exp2_weighted": {
    "weighting": true, "alpha": null,
    "top1": null, "top5": null, "top10": null,
    "delta_v10": null
  },
  "exp3_csls": {
    "csls_k": 10, "hubness_check_passed": null,
    "top20_hub_words": null, "hub_pct": null,
    "top1": null, "top5": null, "top10": null,
    "delta_v10": null
  },
  "exp4_768d_ablation": {
    "input_dim": 768, "best_alpha_768d": null,
    "top1": null, "top5": null, "top10": null,
    "delta_v10": null
  },
  "exp5_glove_840b": {
    "glove_vocab_size": null, "anchor_coverage_840b": null,
    "anchor_coverage_delta": null, "best_alpha_840b": null,
    "top1": null, "top5": null, "top10": null,
    "delta_v10": null
  },
  "best_config": { "description": null, "top1": null }
}
```

---

## Promotion Criteria

If V13 beats V10:
- Manually re-run `regenerate_final_output.py` with winning config
- Update `final_output/metadata.json`
- Update root `README.md` progress table

The notebook reports results — promotion is a manual decision.

---

## Data Dependencies

| Data | Path | Size | Status |
|------|------|------|--------|
| FastText V7 model | `heiro_v7_FastTextVisual/models/fasttext_v7.model` | ~1.6GB | Exists (gitignored) |
| Visual embeddings | `heiro_v9_use_visuals_again/data/processed/visual_embeddings_768d.pkl` | ~1MB | Exists (gitignored) |
| Anchors | `heiro_v6_BERT/data/processed/anchors.json` | ~2MB | Exists (committed) |
| Lexicon mapping | `heiro_v10_refinement/data/lexicon_trans_to_codes.json` | ~500KB | Exists (gitignored) |
| GloVe 6B 300d | `heiro_v5_getdata/data/processed/glove.6B.300d.txt` | ~1GB | Exists (gitignored) |
| GloVe 840B 300d | `heiro_v13/data/glove.840B.300d.txt` | ~2GB | Downloaded by script |
