# TODO: Path to 50% Accuracy

## Current Status
- ✅ v10 baseline: 30.67% accuracy
- ✅ Visual match rate: 0.08% (68/80,662 words)
- 🎯 **Target**: 50% accuracy

---

## Phase 1: Expand Mapping Coverage (Target: +10-15%)

### 1.1 Use TLA Lexicon
- [ ] Parse `heiro_v5_getdata` corpus for Gardiner annotations
- [ ] Build Gardiner → Transliteration dictionary from TLA data
- [ ] Merge with existing Wikipedia mapping
- [ ] Test coverage improvement
- **Expected Gain**: 5-10% | **Effort**: Medium

### 1.2 Normalize Vocabulary ⭐ QUICK WIN
- [ ] Implement suffix stripping (`=f` → `f`, `=k` → `k`)
- [ ] Handle parentheses removal (`n(,j)` → `n`)
- [ ] Strip inflection markers (`.n`, `.t`)
- [ ] Update fusion notebook to use normalization
- [ ] Re-run training and measure improvement
- **Expected Gain**: 3-5% | **Effort**: Low

### 1.3 Manual High-Impact Dictionary
- [ ] Extract top 100 missing words from analysis
- [ ] Research Gardiner codes for top 50 manually
- [ ] Create supplemental mapping file
- [ ] Integrate with main mapping
- [ ] Validate and test
- **Expected Gain**: 5-8% | **Effort**: High

---

## Phase 2: Multi-Glyph Decomposition (Target: +5-10%)

### 2.1 Character-Level Segmentation
- [ ] Identify compound word patterns
- [ ] Implement word segmentation logic
- [ ] Map each segment to Gardiner codes
- [ ] Average visual features for multi-glyph words
- [ ] Test on compound words
- **Expected Gain**: 5-8% | **Effort**: Medium

---

## Phase 3: Architecture Improvements (Target: +5-10%)

### 3.1 Weighted Fusion
- [ ] Add learnable fusion parameter
- [ ] Implement attention mechanism
- [ ] Train with validation set
- [ ] Compare with baseline
- **Expected Gain**: 2-4% | **Effort**: Low

### 3.2 Non-Linear Alignment
- [ ] Replace Ridge Regression with MLP
- [ ] Design 2-layer architecture (1536→512→300)
- [ ] Train with early stopping
- [ ] Evaluate on test set
- **Expected Gain**: 3-5% | **Effort**: Medium

### 3.3 Curriculum Learning
- [ ] Sort anchors by confidence/frequency
- [ ] Implement staged training
- [ ] Test learning curve
- **Expected Gain**: 2-3% | **Effort**: Low

---

## Phase 4: Data Augmentation (Target: +3-5%)

### 4.1 Synonym Expansion
- [ ] Extract German→English synonyms from TLA
- [ ] Generate additional anchor pairs
- [ ] Validate quality
- **Expected Gain**: 2-3% | **Effort**: Medium

### 4.2 Back-Translation
- [ ] Set up translation pipeline
- [ ] Generate paraphrases
- [ ] Filter low-quality pairs
- **Expected Gain**: 1-2% | **Effort**: Medium

---

## Iteration Plan

### ✅ Iteration 0: Setup (Completed)
- ✅ Created v10 environment
- ✅ Scraped Wikipedia Gardiner mapping (230 codes)
- ✅ Baseline run: 30.67% accuracy

### 🔄 Iteration 1: Quick Wins (Current) - Est. 1-2 hours
- [ ] **1.2** Normalize vocabulary → Target: 33-35%
- [ ] **1.3** Manual top-50 dictionary → Target: 36-38%

### Iteration 2: Core Fix - Est. 3-4 hours
- [ ] **1.1** Extract TLA mappings → Target: 40-43%
- [ ] **2.1** Multi-glyph decomposition → Target: 43-47%

### Iteration 3: Architecture - Est. 2-3 hours
- [ ] **3.1** Weighted fusion → Target: 45-49%
- [ ] **3.2** Non-linear alignment → Target: 48-52%

---

## Success Metrics

| Metric | Baseline | Iteration 1 | Iteration 2 | Iteration 3 | Target |
|--------|----------|-------------|-------------|-------------|--------|
| Visual Match Rate | 0.08% | 5-10% | 20-30% | 30-40% | 50%+ |
| Top-1 Accuracy | 30.67% | 36-38% | 43-47% | 48-52% | 50%+ |
| Top-5 Accuracy | 37.69% | 44-46% | 51-55% | 56-60% | 60%+ |
