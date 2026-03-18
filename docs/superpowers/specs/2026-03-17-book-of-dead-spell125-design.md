# Book of the Dead — Spell 125 (Papyrus of Ani): Geometric Translation

## Summary

Apply the heiroglyphy geometric translation pipeline to Spell 125 of the Book of the Dead (Papyrus of Ani), using the rosetta decree LaTeX template as a structural base. The document presents Budge's (1895) standard translation alongside geometric translations generated from the V15 aligned embedding space, with per-glyph analysis showing nearest neighbors and cosine similarities.

## Source Material

- **Hieroglyphic text & standard translation**: Budge, E.A.W. (1895). *The Book of the Dead: The Papyrus of Ani*. British Museum. (Public domain)
- **Supplementary scholarly reference**: Faulkner, R.O. (1972). *The Ancient Egyptian Book of the Dead*. British Museum Press. Cited where his reading materially differs from or clarifies Budge.
- **Transliteration**: Derived from Budge's published hieroglyphic text

## Spell 125 Structure

Three sections:

1. **Introduction** — The deceased (Ani) enters the Hall of Two Truths (mAat), addresses Osiris
2. **The 42 Negative Confessions** — Declarations of innocence before 42 assessor gods (e.g., "I have not stolen," "I have not killed")
3. **Post-judgment declarations** — The deceased addresses the gods after weighing of the heart

## Document Structure (LaTeX)

### Template adaptations from rosetta decree

- **Remove**: Greek font (`\gk{}`), Demotic references, all Greek/Demotic lines from `\glyphanalysis`
- **Simplify `\glyphanalysis`** from 8 args to 6: `{glyph}{gardiner}{translit}{standard}{geometric}{analysis}`
- **Rename** `\decreerow` → `\spellrow` (same 4-layer interlinear: hieroglyphs, transliteration, standard, geometric)
- **Keep** `\reconstructedrow` for damaged/uncertain passages
- **Keep** color accent as rosetta terracotta — consistent project identity
- **Keep** glyph index appendix
- **Keep** commentary section

### Document sections

1. Title page — "The Geometry of the Afterlife: Spell 125 of the Book of the Dead Through Semantic Alignment"
2. Introduction — context on Spell 125, the Papyrus of Ani, methodology note
3. Source Text — attribution to Budge, note on Faulkner
4. The Spell: Introduction (lines entering the Hall)
5. The Spell: The 42 Negative Confessions
6. The Spell: Post-judgment declarations
7. Commentary — where geometry diverges from Budge, thematic analysis (afterlife/judgment/moral vocabulary)
8. Glyph Index appendix
9. Bibliography

## Geometric Translation Pipeline

1. Extract transliterated Egyptian words from Budge's Spell 125 text
2. For each word, look up in `final_output/egyptian_aligned_vectors.npz` + `egyptian_aligned_vocab.pkl`
3. Find nearest English neighbors in GloVe space using the aligned vectors
4. Record top-N neighbors with cosine similarities for glyph analysis blocks
5. Compose geometric sentence-level translations informed by neighbor clusters
6. Flag words not in the V15 vocabulary (10,833 words) — these get standard translations only

### Expected high-interest divergences

- **mAat** — already documented as "truth-and-power, the binding force" vs. standard "truth/justice"
- **bA** (soul/ba) — geometric reading may reveal whether Egyptians conceptualized the ba as we understand "soul"
- **ib** (heart) — the organ weighed against mAat; neighbor cluster may reveal heart-as-conscience vs heart-as-organ
- **Ax** (akh/spirit) — the transformed state after judgment
- **wab** — "priest" vs. "administrator" (already documented in rosetta decree)
- **nTr/nTrw** — divine vocabulary, gold/divinity cluster

## File Output

- `docs/paper/book_of_dead_spell125.tex` — main document
- `docs/paper/book_of_dead_spell125.pdf` — compiled PDF

## Bibliography

- Budge, E.A.W. (1895). *The Book of the Dead: The Papyrus of Ani*. British Museum.
- Faulkner, R.O. (1972). *The Ancient Egyptian Book of the Dead*. British Museum Press.
- Brinsmead, E. and Claude (Anthropic) (2026). *The Geometry of Meaning*. github.com/ebrinz/heiroglyphy
