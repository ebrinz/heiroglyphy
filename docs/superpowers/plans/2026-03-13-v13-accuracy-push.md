# V13: Accuracy Push Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push Heiroglyphy accuracy from 30.67% toward 40% through 5 incremental experiments on the V10 pipeline.

**Architecture:** Single experiment notebook (`01_v13_experiments.ipynb`) loads V10 data once and runs 5 sequential experiments (Ridge CV, confidence weighting, CSLS, 768d ablation, GloVe 840B). A standalone download script handles the 2GB GloVe 840B fetch. Results accumulate in a JSON file.

**Tech Stack:** Python 3.8+, numpy, scikit-learn (Ridge, RidgeCV, train_test_split), gensim (FastText, KeyedVectors), tqdm, urllib (download)

**Spec:** `docs/superpowers/specs/2026-03-13-v13-accuracy-push-design.md`

**Note:** Pickle is used to load `visual_embeddings_768d.pkl` — this is the project's own serialized data, not untrusted external content.

---

## Chunk 1: Scaffolding & Data Loading

### Task 1: Create V13 directory structure

**Files:**
- Create: `heiro_v13/README.md`
- Create: `heiro_v13/notebooks/` (directory)
- Create: `heiro_v13/scripts/` (directory)
- Create: `heiro_v13/data/` (directory)
- Create: `heiro_v13/results/` (directory)

- [ ] **Step 1: Create directories**

```bash
mkdir -p heiro_v13/{notebooks,scripts,data,results}
```

- [ ] **Step 2: Create README.md**

Create `heiro_v13/README.md` with experiment table, run instructions, and links to V10 baseline.

- [ ] **Step 3: Add v13 to .gitignore**

Append to `.gitignore`:

```
# ============================================
# V13 - Accuracy Push
# ============================================
heiro_v13/data/
heiro_v13/notebooks/*.ipynb
```

- [ ] **Step 4: Commit**

```bash
git add heiro_v13/README.md .gitignore
git commit -m "feat: scaffold V13 accuracy push experiment"
```

---

### Task 2: Create the GloVe 840B download script

**Files:**
- Create: `heiro_v13/scripts/download_glove_840b.py`

Script downloads `glove.840B.300d.zip` (~2GB) from HuggingFace Stanford NLP mirror, verifies file size (>2GB sanity check), extracts to `heiro_v13/data/glove.840B.300d.txt`, removes zip. Includes progress reporting and skip-if-exists logic.

- [ ] **Step 1: Write download script**
- [ ] **Step 2: Verify script syntax** (`python3 -c "import ast; ast.parse(...);"`)
- [ ] **Step 3: Commit**

```bash
git add heiro_v13/scripts/download_glove_840b.py
git commit -m "feat: add GloVe 840B download script for V13"
```

---

## Chunk 2: Experiment Notebook — Sanity Check + Experiments 1-2

### Task 3: Create experiment notebook with data loading and sanity check

**Files:**
- Create: `heiro_v13/notebooks/01_v13_experiments.ipynb`

**Notebook structure (cells):**

**Cell 1-2: Header + Imports/Config**
- Import numpy, json, sklearn, gensim, tqdm, collections.Counter, pathlib
- Set REPO_ROOT (handle running from notebooks/ or repo root)
- Initialize RESULTS dict with sanity_check and baseline_v10 entries
- `save_results()` helper that writes to `heiro_v13/results/experiment_results.json`

**Cell 3: Load all data**
- V7 FastText model (768d) from `heiro_v7_FastTextVisual/models/fasttext_v7.model`
- V9 visual embeddings from `heiro_v9_use_visuals_again/data/processed/visual_embeddings_768d.pkl`
- V10 Gardiner mapping from `heiro_v10_refinement/data/gardiner_mapping.json` — build reverse mapping (transliteration -> Gardiner codes)
- Anchors from `heiro_v5_getdata/data/processed/english_anchors.json` (8,541 pairs, fields: hieroglyphic, english, german, confidence, frequency)
- GloVe 6B from `heiro_v5_getdata/data/processed/glove.6B.300d.txt`

**Cell 4: Create fused embeddings + train/test split**
- Build 1536d fused vectors (768d text + 768d visual/zeros) for all 80,662 words
- Build anchor arrays: X_all (1536d), Y_all (300d), conf_all (float), anchor_pairs (list of tuples)
- `train_test_split(X_all, Y_all, conf_all, anchor_pairs, test_size=0.2, random_state=42)`
- Print valid anchor count and train/test sizes

**Cell 5: Evaluation helpers**
- `evaluate(Y_pred, Y_test, pairs_test, english_kv, topn=10)` — NN retrieval, returns dict with top1/top5/top10 and predictions list
- `evaluate_csls(Y_pred, Y_test, pairs_test, english_kv, k=10, topn=10)` — CSLS retrieval. For each prediction: compute cosine sims to full GloVe vocab, compute r_T (mean sim to k nearest English neighbors), CSLS score = 2*sim - r_T, rank by CSLS score. Returns dict with top1/top5/top10.

**Cell 6-7: Sanity check**
- Train Ridge(alpha=1.0) on train set, predict test set
- Evaluate and compare to V10 baseline (30.67%)
- HALT if delta > 0.1% — data may have drifted
- Save to RESULTS["sanity_check"]

- [ ] **Step 1: Create notebook with cells 1-7**
- [ ] **Step 2: Verify sanity check passes** (run cell 7)

---

### Task 3b: Add Experiment 1 (Ridge Alpha CV)

**Append to notebook:**

**Cell 8-9: Experiment 1**
- `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0], cv=5)`
- Report best alpha
- Also run full alpha sweep individually and print table (each alpha → Top-1/Top-5)
- Save to RESULTS["exp1_ridge_cv"] with best_alpha, cv_folds, top1/5/10, delta_v10

- [ ] **Step 1: Add Exp 1 cells**
- [ ] **Step 2: Run and verify**

---

### Task 3c: Add Experiment 2 (Confidence-Weighted Ridge)

**Append to notebook:**

**Cell 10-11: Experiment 2**
- Print confidence distribution (mean, std, %<0.5, %>0.8)
- `Ridge(alpha=best_alpha_exp1).fit(X_train, Y_train, sample_weight=conf_train)`
- Compare to Exp 1 (unweighted)
- Note limitation: alpha not re-tuned jointly with weighting
- Save to RESULTS["exp2_weighted"]

- [ ] **Step 1: Add Exp 2 cells**
- [ ] **Step 2: Run and verify**
- [ ] **Step 3: Commit notebook**

```bash
git add heiro_v13/notebooks/01_v13_experiments.ipynb
git commit -m "feat(v13): notebook with sanity check, Ridge CV, and weighted Ridge"
```

---

## Chunk 3: Experiments 3-4 (CSLS + 768d Ablation)

### Task 4: Add Experiment 3 (CSLS Retrieval)

**Append to notebook:**

**Cell 12-13: Hub analysis (go/no-go)**
- Get predictions from best model so far (Exp 1 or 2)
- Count top-1 prediction frequencies
- Print top 20 most frequent predictions
- Flag stopwords (the, of, and, to, a, in, is, it, for, on, ...)
- If stopword hub rate < 10%: warn that CSLS unlikely to help, but proceed anyway

**Cell 14: Run CSLS**
- Call `evaluate_csls()` on best model predictions
- Compare to NN results
- Save to RESULTS["exp3_csls"] with csls_k, hubness_check_passed, top20_hub_words, hub_pct, top1/5/10

- [ ] **Step 1: Add Exp 3 cells**
- [ ] **Step 2: Commit**

```bash
git add heiro_v13/notebooks/01_v13_experiments.ipynb
git commit -m "feat(v13): add CSLS retrieval experiment with hub analysis"
```

---

### Task 5: Add Experiment 4 (768d Ablation)

**Append to notebook:**

**Cell 15-16: 768d Ablation**
- Slice: `X_train_768 = X_train[:, :768]`, `X_test_768 = X_test[:, :768]`
- Print param count comparison (768x300=230K vs 1536x300=460K)
- Run its OWN `RidgeCV(alphas=..., cv=5)` on 768d data — do NOT reuse 1536d alpha
- Compare to 1536d Exp 1 result
- Interpret: negligible diff = zeros were regularization; 768d better = zeros were noise; 1536d better = capacity benefit
- Save to RESULTS["exp4_768d_ablation"] with input_dim, best_alpha_768d, top1/5/10

- [ ] **Step 1: Add Exp 4 cells**
- [ ] **Step 2: Commit**

```bash
git add heiro_v13/notebooks/01_v13_experiments.ipynb
git commit -m "feat(v13): add 768d ablation experiment"
```

---

## Chunk 4: Experiment 5 (GloVe 840B) + Summary

### Task 6: Add Experiment 5 (GloVe 840B)

**Append to notebook:**

**Cell 17-18: GloVe 840B**
- Check if `heiro_v13/data/glove.840B.300d.txt` exists; if not, print instructions and skip
- Load GloVe 840B via `KeyedVectors.load_word2vec_format()`
- Rebuild anchor arrays against 840B (new X_840, Y_840, conf_840, pairs_840)
- Log anchor coverage delta vs 6B
- New train/test split with `random_state=42`
- **Retrain Ridge end-to-end** with RidgeCV against 840B targets — do NOT reuse 6B-trained model
- Carry forward weighting if Exp 2 improved over Exp 1
- Evaluate and compare
- Save to RESULTS["exp5_glove_840b"] with glove_vocab_size, anchor_coverage_840b, anchor_coverage_delta, best_alpha_840b, weighted, top1/5/10

- [ ] **Step 1: Add Exp 5 cells**
- [ ] **Step 2: Commit**

```bash
git add heiro_v13/notebooks/01_v13_experiments.ipynb
git commit -m "feat(v13): add GloVe 840B experiment"
```

---

### Task 7: Add summary cell and finalize

**Append to notebook:**

**Cell 19-20: Summary**
- Print formatted table: all experiments with Top-1/5/10 and delta vs V10
- Determine best config (highest Top-1 across all experiments)
- Save RESULTS["best_config"]
- Print path to full results JSON

- [ ] **Step 1: Add summary cells**
- [ ] **Step 2: Final commit**

```bash
git add heiro_v13/
git commit -m "feat(v13): complete V13 accuracy push experiment suite"
```

---

## Chunk 5: Verify

### Task 8: Verify all files and run

- [ ] **Step 1: Verify file structure**

```bash
find heiro_v13 -type f | sort
```

Expected:
```
heiro_v13/README.md
heiro_v13/notebooks/01_v13_experiments.ipynb
heiro_v13/scripts/download_glove_840b.py
```

- [ ] **Step 2: Run experiments 1-4** (no GloVe 840B download needed)

```bash
cd heiro_v13/notebooks && jupyter nbconvert --to notebook --execute 01_v13_experiments.ipynb --output 01_v13_experiments_executed.ipynb
```

- [ ] **Step 3: Check results**

```bash
cat heiro_v13/results/experiment_results.json | python3 -m json.tool
```

Verify: sanity check passed, all 4 experiments have results, exp5 shows skipped.
