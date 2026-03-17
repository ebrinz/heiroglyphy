# The Geometry of the Decree: An Alternate Translation Through Semantic Alignment

**Date:** 2026-03-17
**Status:** Draft
**Companion to:** "The Geometry of Meaning" (docs/paper/heiroglyphy.tex)

---

## Overview

A LaTeX document presenting the Decree of Memphis (the Rosetta Stone text) translated through vector-space alignment, shown side-by-side with the standard scholarly translation. Wide margin notes annotate individual glyphs with contextual swapping data — how the embedding space reads each glyph differently depending on its surrounding context — along with Demotic and Greek cross-references.

This is a companion piece to the main paper, aimed at a scholarly audience familiar with the methodology.

---

## Page Layout

- **Page size:** A4
- **Margins:** Asymmetric Tufte-style
  - Inner margin: 1in
  - Outer margin: 3.5in
  - Margin note column: ~2.8in wide
  - Text block: ~4in wide
- **Font:** XeLaTeX with EgyptianHiero.ttf (same setup as existing paper)
- **Body text:** Latin Modern or similar serif
- **Margin notes:** `\small` or `\footnotesize`

---

## Interlinear Row Structure

Each line of the decree is rendered as a four-layer stack using a `\decreerow` command:

```
Layer 1: Hieroglyphs          (EgyptianHiero font, large)
Layer 2: Transliteration       (italic, standard Egyptological)
Layer 3: Standard translation  (regular weight, cited source)
Layer 4: Geometric translation (bold, terracotta accent color)
```

- Rows separated by subtle horizontal rules or vertical spacing.
- A `\reconstructedrow` variant renders layers in gray with `⟦ ⟧` brackets around hieroglyphs, following standard Egyptological convention for lacunae.

---

## Margin Note Structure

Margin notes are anchored to specific glyphs or phrases within a row using a `\glyphnote` command. Each note contains:

### Header
- The glyph (rendered in EgyptianHiero), Gardiner code, transliteration

### Contextual Swap
- **standard:** the conventional translation in this context
- **geometric:** the vector-space translation in this context
- **cos_sim:** cosine similarity score
- **context window:** surrounding words that influence the reading
- **nearest neighbors:** top 3-5 neighbors with similarity scores

### Cross-Script References
- **Demotic:** what the Demotic version says for this passage
- **Greek:** what the Greek version says (in Greek script)

### Selection Criteria

Not every glyph receives a margin note. Notes appear when:

1. The geometric translation diverges meaningfully from the standard
2. The contextual neighbors reveal something unexpected
3. The Demotic/Greek cross-reference adds insight
4. A glyph appears multiple times in the decree with different contextual readings (the core "contextual swapping" showcase)

---

## Document Structure

### 1. Title Page
- Title: "The Geometry of the Decree: An Alternate Translation Through Semantic Alignment"
- Authors: Erik Brinsmead (Independent Researcher), Claude (Anthropic)
- Subtitle referencing companion status to "The Geometry of Meaning"

### 2. Introduction (1–2 pages)
- What this document is and how it relates to the main paper
- Brief methodology recap (pointing to main paper for detail)
- How to read the interlinear format and margin notes
- Note on fragmentary hieroglyphic section and reconstruction conventions
- Source citations for the transcription used

### 3. The Decree — Surviving Lines (~14 lines)
- Interlinear rows with full margin annotation
- Lines numbered to match the physical stone

### 4. The Decree — Reconstructed Lines (~15 lines)
- Same interlinear format, visually marked as reconstructions (gray + `⟦ ⟧`)
- Lighter margin annotation where less certainty exists

### 5. Commentary (2–3 pages)
- Thematic discussion of the most significant divergences
- Patterns in contextual swapping — glyphs that shift most across the decree
- Connections to discoveries from the main paper (gold/divinity cluster, priest-as-administrator, etc.)

### 6. Appendix: Glyph Index
- Every glyph appearing in the decree, sorted by Gardiner code
- Each entry lists all contextual readings across the text, showing the full "swap" range

---

## Technical Implementation

### File Location
`docs/paper/rosetta_decree.tex` — alongside existing paper files, sharing the relative path to `EgyptianHiero.ttf`.

### Key Packages
| Package | Purpose |
|---------|---------|
| `fontspec` | XeLaTeX font handling, EgyptianHiero |
| `geometry` | Asymmetric margins |
| `marginnote` | Reliable margin notes (no float conflicts) |
| `xcolor` | Terracotta accent for geometric translations |
| `enumitem` | List formatting in commentary |
| `booktabs` | Tables in glyph index |
| `hyperref` | Cross-references between margin notes and glyph index |

### Custom Commands
| Command | Purpose |
|---------|---------|
| `\decreerow{hieroglyphs}{translit}{standard}{geometric}` | Four-layer interlinear stack |
| `\reconstructedrow{hieroglyphs}{translit}{standard}{geometric}` | Same, with gray styling + `⟦ ⟧` |
| `\glyphnote{glyph}{gardiner}{translit}{body}` | Margin annotation block |
| `\rosettacolor` | Terracotta accent color definition |
| `\hiero{text}` | Hieroglyphic font wrapper (reused from existing paper) |

### Compilation
XeLaTeX. No special build pipeline beyond what the existing paper requires.

### Source Text
Modern transcription of the Rosetta Stone hieroglyphic section, based on scholarly references (Quirke & Andrews, with Budge as secondary). Reconstructed lines sourced from Demotic/Greek back-translation per standard Egyptological practice.

---

## Scope Boundaries

**In scope:**
- The LaTeX document with all sections described above
- Custom commands for interlinear rows and margin notes
- Placeholder/example content for a representative subset of lines (the full decree transcription is a content task, not a code task)

**Out of scope:**
- Automated embedding lookups at build time (Approach C — deferred)
- Demotic or Greek font rendering (referenced in plain text/transliteration for now)
- Interactive or digital-only features
