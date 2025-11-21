# Heiroglyphy V8: Coptic Bridge Alignment

## Overview

**V8** explores using Coptic as an intermediate "bridge" language to improve hieroglyphic-to-English alignment. Coptic is the direct descendant of Ancient Egyptian and preserves phonological information lost in hieroglyphic transliteration.

## Status

🚧 **In Planning** - This is a research proposal for the next iteration.

## Motivation

### Why Coptic?

**Coptic** is the final stage of the Ancient Egyptian language, written in a modified Greek alphabet with some Demotic characters. It offers several unique advantages:

1. **Direct Linguistic Continuity**
   - Coptic evolved directly from Late Egyptian → Demotic → Coptic
   - ~60-70% of Coptic vocabulary derives from Egyptian
   - Grammatical structures are preserved

2. **Phonological Information**
   - Coptic includes **vowels** (hieroglyphs don't!)
   - Example: Egyptian `nfr` → Coptic `ⲛⲟⲩϥⲣⲉ` (noufe) → English "good"
   - Pronunciation helps semantic alignment

3. **More Training Data**
   - **Biblical translations**: Coptic Bible parallel with Greek/English
   - **Liturgical texts**: Extensive church literature
   - **Nag Hammadi Codices**: Gnostic texts with translations
   - **Coptic SCRIPTORIUM**: Large annotated corpus

4. **Existing Resources**
   - Egyptian-Coptic lexicons (Crum's dictionary)
   - Coptic-English dictionaries
   - Digital corpora with annotations

## Proposed Approach

### Strategy 1: Triangulated Alignment

Instead of direct Egyptian → English alignment, use two smaller gaps:

```
Egyptian embeddings ──┐
                      ├──> Coptic embeddings ──> English embeddings
Coptic corpus ────────┘
```

**Steps:**
1. Train FastText on Egyptian transliteration (reuse V7)
2. Train FastText on Coptic corpus
3. Train FastText on English (reuse GloVe or V7)
4. Create Egyptian-Coptic anchors (from lexicons)
5. Create Coptic-English anchors (from Bible translations)
6. Align Egyptian → Coptic → English

**Expected Benefit**: Smaller semantic gaps, better alignment

### Strategy 2: Coptic-Enhanced Anchors

Use Coptic cognates to expand the anchor dictionary:

```
Egyptian: nfr → Coptic: ⲛⲟⲩϥⲣⲉ → English: good
Egyptian: nṯr → Coptic: ⲛⲟⲩⲧⲉ → English: god
```

**Steps:**
1. Extract Egyptian-Coptic cognate pairs from Crum's dictionary
2. Map Coptic words to English via Bible translations
3. Create expanded Egyptian-English anchors
4. Retrain V7 alignment with larger anchor set

**Expected Benefit**: Higher anchor coverage, especially for religious/common terms

### Strategy 3: Joint Multilingual Embeddings

Train a single embedding space for all three languages:

**Steps:**
1. Combine Egyptian + Coptic + English corpora
2. Train multilingual FastText (like Facebook's MUSE)
3. Use anchor pairs to guide alignment during training
4. Evaluate on Egyptian-English test set

**Expected Benefit**: Shared semantic space, implicit knowledge transfer

## Data Sources

### Coptic Corpora
- **Coptic SCRIPTORIUM**: ~1.5M words, annotated
- **Coptic Bible**: Sahidic/Bohairic dialects, parallel with Greek/English
- **Nag Hammadi Codices**: ~50 texts with English translations
- **Crum's Coptic Dictionary**: ~10k entries with Egyptian etymologies

### Egyptian-Coptic Mappings
- **Crum's Dictionary**: Egyptian → Coptic cognates
- **Vycichl's Etymological Dictionary**: Detailed etymologies
- **TLA**: Some entries include Coptic cognates

### Coptic-English Mappings
- **Bible translations**: Direct parallel text
- **Coptic dictionaries**: Coptic → English definitions

## Technical Challenges

1. **Orthographic Mismatch**
   - Egyptian: Latin transliteration (`nfr`)
   - Coptic: Greek alphabet (`ⲛⲟⲩϥⲣⲉ`)
   - Need robust character encoding handling

2. **Temporal Gap**
   - Middle Egyptian (2000 BCE) → Coptic (300-1400 CE)
   - ~2000+ year gap, vocabulary drift
   - Some words changed meaning or disappeared

3. **Dialect Variation**
   - Coptic has 6+ dialects (Sahidic, Bohairic, Fayyumic, etc.)
   - Need to choose primary dialect or combine

4. **Data Sparsity**
   - Egyptian-Coptic parallel texts are rare
   - Most mappings are lexicon-based, not corpus-based

## Success Metrics

| Metric | V7 Baseline | V8 Target | Stretch Goal |
|--------|-------------|-----------|--------------|
| **Top-1 Accuracy** | 29.10% | >32% | >35% |
| **Anchor Coverage** | 78.4% | >85% | >90% |
| **Deity Accuracy** | - | >70% | >80% |

## Project Structure

```
heiro_v8_use_coptic/
├── data/
│   ├── raw/
│   │   ├── coptic_scriptorium/      # Coptic corpus
│   │   ├── coptic_bible/            # Parallel Bible texts
│   │   └── crum_dictionary.json     # Egyptian-Coptic lexicon
│   └── processed/
│       ├── coptic_embeddings.kv     # Coptic FastText
│       ├── egyptian_coptic_anchors.json
│       └── coptic_english_anchors.json
├── notebooks/
│   ├── 01_coptic_data_collection.ipynb
│   ├── 02_egyptian_coptic_mapping.ipynb
│   ├── 03_coptic_embeddings.ipynb
│   └── 04_triangulated_alignment.ipynb
├── scripts/
│   └── (alignment scripts)
└── README.md
```

## Next Steps

1. **Data Collection**
   - Download Coptic SCRIPTORIUM corpus
   - Extract Coptic Bible parallel texts
   - Digitize Crum's dictionary entries

2. **Proof of Concept**
   - Test Egyptian-Coptic cognate coverage
   - Measure Coptic-English alignment quality
   - Compare direct vs triangulated alignment

3. **Full Implementation**
   - Train Coptic embeddings
   - Create anchor dictionaries
   - Implement triangulated alignment
   - Evaluate and compare to V7

## Expected Outcomes

**Optimistic**: 32-35% accuracy (10-20% relative improvement over V7)
**Realistic**: 30-32% accuracy (3-10% relative improvement)
**Pessimistic**: 29% accuracy (no improvement, but valuable negative result)

Even if accuracy doesn't improve, this experiment will provide insights into:
- The role of phonology in semantic alignment
- Whether linguistic continuity helps cross-lingual embeddings
- The value of intermediate languages for distant language pairs

---

**Status**: 🚧 Planning  
**Previous**: [V7 FastText + Visuals](../heiro_v7_FastTextVisual/) - 29.10% accuracy  
**Next Step**: Collect Coptic corpus and create proof-of-concept notebook
