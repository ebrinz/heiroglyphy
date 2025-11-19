# Hieroglyphic-to-English Translation using Vec2Vec

This project implements cross-temporal semantic mapping between ancient Egyptian hieroglyphic concepts and modern English using vec2vec neural translation, with completely separate training corpora to avoid translation bias.

## Overview

The approach trains two completely separate embedding spaces:
- **Hieroglyphic embeddings**: Trained only on transliterated hieroglyphic texts
- **Modern English embeddings**: Trained on contemporary English (Wikipedia, news, social media) with NO Egyptian/ancient content

Then uses vec2vec to discover mappings between these spaces, revealing fascinating connections like:
- What modern English concept is closest to hieroglyphic "𓈖𓄿𓇳" (divine/solar)?
- How do ancient titles map to modern concepts?
- What semantic evolution has occurred over millennia?

## Key Features

### No Translation Bias
Unlike traditional approaches that use parallel corpora or dictionaries, this project deliberately trains on **completely separate** datasets:
- Hieroglyphic space: Pure TLA transliterations
- English space: Modern text with ancient Egyptian content explicitly filtered out

This tests whether vec2vec can discover semantic mappings purely from geometric structure.

### Pure Geometric Mapping
The vec2vec model learns to map between spaces using only:
- Distributional semantics (words appearing in similar contexts)
- Geometric relationships (analogies, clusters)
- No supervised translation pairs

## Project Structure

```
heiro_v2/
├── data/           # Processed datasets (excluded from git)
├── models/         # Trained models (excluded from git)
├── results/        # Analysis results (excluded from git)
└── src/            # ACCIDENTALLY ERASED :/
```

## Note

This was Attempt 2 in the Heiroglyphy project series. The source code was not committed to the repository, but the approach and methodology are documented here. For a working implementation, see:
- **heiro_v1**: Full implementation with source code
- **heiro_v3**: Successful anchor-guided alignment approach
- **heiro_v4**: CSLS refinement

## Outcome

This "pure unsupervised" approach highlighted the limitations of geometric alignment without anchors. The semantic gap between Ancient Egyptian (agrarian, religious, 2000 BCE) and Modern English (technological, global, 2024 CE) proved too large for unsupervised vec2vec to bridge effectively.

This negative result was valuable - it led directly to the successful "Anchor-Guided Alignment" approach in heiro_v3.
