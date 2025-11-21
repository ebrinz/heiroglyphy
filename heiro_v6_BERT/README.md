# Heiroglyphy V6: BERT Contextual Embeddings

## Overview

**V6** attempted to improve on V5's 24.53% accuracy by using BERT contextual embeddings instead of static FastText vectors. The hypothesis was that context-aware representations would better capture the polysemous nature of Ancient Egyptian.

## Approach

- **Model**: BERT (bert-base-uncased)
- **Data**: BBAW hieroglyphic corpus (100k sentences)
- **Method**: Extract BERT embeddings for hieroglyphic transliteration, align to GloVe English embeddings using Procrustes

## Results

- **Top-1 Accuracy**: 0.47% ❌
- **Anchor Coverage**: Very low
- **Status**: Failed

## Why It Failed

1. **Vocabulary Mismatch**: BERT's WordPiece tokenizer splits hieroglyphic transliteration poorly
   - Example: `ḥr,w` → `['ḥ', '##r', '##,', '##w']` (fragmented)
   - Lost semantic meaning through subword tokenization
   
2. **Semantic Mismatch**: BERT trained on modern English, not suited for ancient languages
   - Pre-trained weights encode modern concepts, not ancient Egyptian semantics
   
3. **Poor Anchor Coverage**: Most hieroglyphic words not properly represented
   - Subword fragmentation prevented meaningful vector lookup

## Lessons Learned

1. **Pre-trained models don't transfer to ancient languages**: Modern NLP models encode contemporary concepts
2. **Vocabulary alignment is critical**: Subword tokenization breaks semantic units in transliteration
3. **Simpler is better for low-resource languages**: FastText outperforms BERT for specialized domains
4. **Static embeddings work**: Context may not be as important as vocabulary coverage for alignment tasks

## Data

- **Source**: BBAW Hieroglyphic Corpus (`data/raw/bbaw_huggingface.parquet`)
- **Anchors**: `data/processed/anchors.json` (8,541 hieroglyphic-English pairs)
- **Visual Embeddings**: `data/processed/visual_embeddings_768d.pkl` (ResNet-50 features, unused)

## Next Steps

See **[V7 FastText + Visuals](../heiro_v7_FastTextVisual/)** for the successful approach that achieved **29.10% accuracy** by returning to FastText with larger (768d) embeddings.
