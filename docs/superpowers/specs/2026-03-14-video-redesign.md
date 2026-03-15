# Heiroglyphy Video Redesign — Full & 3-Minute Cut

## Overview

Two video products from one Manim codebase:

1. **Full version** (~5:30–6:30): Complete project presentation with all 8 discoveries
2. **3-minute cut**: Condensed version for the encore.pillar.vc application

Both share S1–S4. The full version expands S5 to 8 bespoke discovery scenes, adds a discussion/conclusion section. The 3-minute cut cherry-picks 3–4 discoveries.

## Target audience

Lay audience. Insights should land without ML or Egyptology background. The discoveries section should feel like revelations, not academic facts.

## Structure

### S1 Hook (~25s) — Unchanged, tightened waits

Hieroglyphic strip fades in. Three text beats:
- "These symbols are four thousand years old."
- "Scholars have been translating them for two centuries."
- "But translation compresses meaning. The relationships between words are lost."
- "What if we could recover them?"

**Changes**: Reduce `self.wait()` pauses from 14s total to ~8s. Narration unchanged.

### S2 Idea (~30s) — Unchanged, tightened waits

Word embedding concept: labeled word dots cluster by meaning (water/river/flood, king/throne/crown).

**Changes**: Reduce waits from 10s to ~6s. Narration unchanged.

### S3 Alignment (~40s) — Rebuilt with real data

**Data source**: `docs/viz_data.json` (247 Egyptian points, 279 English points, semantic axes, anchor connections, 3 highlighted pairs).

**Phase 1 — Two clouds appear** (~8s)
- Load viz_data.json
- Plot Egyptian points (gold dots) on left half, English points (teal dots) on right half
- Each cloud independently normalized to fit its screen half
- Real coordinates preserve relative structure within each cloud
- Labels: hieroglyph + "Egyptian" (left), "English" (right)
- Subtle muted axis labels at screen edges: "mortal ↔ divine" (x-axis), "death ↔ life" (y-axis)

**Phase 2 — "Find the rotation"** (~7s)
- Text: "Both languages form a shape. The shapes are similar — but rotated."
- Animate Egyptian cloud: scale + translate + rotate to overlap English cloud
- Transform parameters computed from actual data ranges (Egyptian space is compressed relative to English)

**Phase 3 — Anchor lines** (~4s)
- Draw ~10 thin connection lines between real anchor pairs from the data
- Pick pairs where points ended up close after merge (visually clean)

**Phase 4 — Golden hits** (~5s)
- Highlight 3 real hits with bright labeled connections:
  - mw → water
  - nṯr → god
  - nswt → king
- Each pair gets Egyptian word label (gold) + English word label (teal) + connecting line (lavender)

**Phase 5 — Accuracy beat** (~4s)
- "32.35% accuracy — no dictionary needed."

**Narration**: Same 5 segments as current, timing adjusted to match tighter animation.

### S4 Journey (~20s) — Unchanged, tightened waits

Bar chart showing V3→V15 accuracy progression. BERT failure highlighted.

**Changes**: Reduce waits from 6.5s to ~4s. Narration unchanged.

### S5 Discoveries — 8 independent mini-scenes

Each discovery is its own visual moment: ~25–35s, filling a large portion of the screen. Structure per discovery:
- Glyph strip + title at top
- Bespoke visualization filling center
- Punchline insight text at bottom

Narration is simplified from paper prose — accessible but preserving the "reveal" moments. Structured as: context → finding → punchline.

---

#### D1: Gold Is Divine Flesh (~30s)

**Glyphs**: 𓋴𓈖𓃀𓅱

**Narration** (simplified):
"What happens when you find the midpoint of 'gold' and 'divine' in English, then look for the nearest Egyptian words? You find nṭri — divine — and nbw — gold. They're in the same place. Modern readers call 'the flesh of the gods is gold' a metaphor. The embedding space says it's not. Gold and divinity aren't compared in the texts. They're the same concept. This is not metaphor. It is ontology."

**Visualization**:
- Two labeled dots ("gold", "divine") on opposite sides of screen
- A midpoint marker pulses between them
- Arrow projects into Egyptian space
- nṭri and nbw dots glow, nearly overlapping
- Dots merge into one radiant point
- Punchline: "Not metaphor. Ontology."

---

#### D2: Silence Is the Condition of the Dead (~30s)

**Glyphs**: 𓇯𓂋𓊽

**Narration** (simplified):
"Find the midpoint of 'silence' and 'death.' Every single nearest neighbor is a variant of the word 'to die.' There is no Egyptian word between silence and death — they are the same point in space. The Egyptians called the necropolis 'the silent land.' The dead were 'the silent ones.' What the dead lost was not life. It was voice."

**Visualization**:
- A stylized sound wave on the left, gently oscillating
- Wave gradually flattens to a line (silence)
- Word dots for "silence" and "death" drift together, converge to same point
- m(w)t variants appear clustered at that point
- Punchline: "What the dead lost was not life. It was voice."

---

#### D3: Seeing Was an Act of Magical Power (~25s)

**Glyphs**: 𓂀𓎛𓋴

**Narration** (simplified):
"The midpoint of 'eye' and 'knowledge' finds the Egyptian word for eyes at the top — and heka, magic, as the third result. The Eye of Horus was an organ, an amulet, and a unit of measurement all at once. Seeing was not passive observation. It was an act of power."

**Visualization**:
- Eye of Horus glyph at center, large
- Three vectors radiate outward to labeled points: "knowledge", "spellcasting", "protection"
- Triangle of meaning forms, connecting the three
- Glyph pulses with energy along the connections
- Punchline: "Sight was not observation. It was power."

---

#### D4: The Snake Is Divine, Not Wise (~25s)

**Glyphs**: 𓆙𓊽𓋴

**Narration** (simplified):
"Find the midpoint of 'snake' and 'wisdom.' In the Greek tradition, you'd expect to find knowledge and cunning. In the Egyptian space, every result is a variant of 'god.' The uraeus cobra on the pharaoh's brow was divine power, not wisdom. Two cultures, separated by geometry."

**Visualization**:
- Split screen, divided by a vertical line
- Left: Greek column icon, snake → "wisdom" path (expected)
- Right: Egyptian temple icon, snake → "gods" path (actual)
- The expected path fades, the actual path glows
- Punchline: "Two cultures, separated by geometry."

---

#### D5: Temple Is to House as God Is to Man (~30s)

**Glyphs**: 𓉐𓊽𓀀

**Narration** (simplified):
"Here's where it gets remarkable. Take the vector from 'house' to 'temple' and apply it to 'man.' The result? 'God.' A temple is a god's house, in the same geometric sense that a house is a man's dwelling. This is the 'king minus man plus woman equals queen' trick — working across a four-thousand-year language boundary."

**Visualization**:
- Four points arranged in a parallelogram: house (bottom-left), temple (top-left), man (bottom-right), ? (top-right)
- Animated arrow from house→temple (labeled "sacred")
- Same arrow slides to man→? position
- ? resolves to "god" with a satisfying snap
- Parallel arrows highlighted to show the proportional relationship
- Punchline: "Vector arithmetic across 4,000 years."

---

#### D6: Mother Is Royalty, Not Earth (~25s)

**Glyphs**: 𓅭𓏏𓀀

**Narration** (simplified):
"The midpoint of 'mother' and 'earth' — you'd expect soil, land, fertility. The earth-mother archetype. Instead, every result is royal: 'king's wife,' 'king's daughter.' The earth mother is an Indo-European idea. The Egyptian mother-goddess is not earthy. She is regal."

**Visualization**:
- Expected cluster ("earth", "soil", "fertility") faded/ghosted in background
- Actual cluster ("royal wife", "king's daughter") lights up brightly in foreground
- A subtle crown or royal icon appears over the "mother" label
- The expectation vs reality contrast is the visual story
- Punchline: "Motherhood is a crown, not the earth."

---

#### D7: Truth and Power Are the Same Force (~25s)

**Glyphs**: 𓁹𓋴𓊽

**Narration** (simplified):
"The midpoint of 'truth' and 'power' finds authority at number one and enemies at number four. Truth, power, and the defeat of enemies — they all occupy the same region. This is māʿat. Usually translated as 'truth' or 'justice,' but it's really cosmic order: the active force that holds the universe together against chaos."

**Visualization**:
- Constellation-style diagram: "truth", "power", "authority", "enemies" as stars
- Lines connect them into a tight cluster
- The cluster pulses as a unified force
- A feather (māʿat symbol) appears at center
- Punchline: "Truth is not correctness. It is force."

---

#### D8: Love and Fear Meet at Eternity (~30s)

**Glyphs**: 𓆣𓇯𓋴

**Narration** (simplified):
"The midpoint of 'love' and 'fear' — what do you find? Eternity. Between love and fear, the Egyptians placed forever. The gods' love is awe-inspiring. Their wrath is terrifying. Both extend beyond time. The offering formula asks that the dead be 'loved and feared for eternity.' These aren't opposites. They're the same prayer."

**Visualization**:
- "Love" and "fear" as two poles on opposite sides
- A point appears at the midpoint, labeled "eternity" (r-nḥḥ)
- Radiating rings expand from the midpoint outward — timelessness
- The two poles are connected through the eternal center
- Punchline: "Between love and fear: forever."

---

### S6 Discussion (~30s) — New

Clean typography, no flashy visualization. Honest and grounding.

**Narration** (simplified):
"A few honest caveats. The surviving Egyptian texts are mostly funerary and religious. This is the language of temples and tombs — not markets and homes. And thirty-two percent accuracy means two-thirds of words don't find their match. These are statistical tendencies, not certainties. But what this approach captures is something translation destroys: the distances between words. A dictionary tells you nṯr means 'god' and nbw means 'gold.' Only the embedding space tells you they're essentially the same word."

**Visual**: Simple text cards with key stats, minimal animation.

### S7 Conclusion (~15s) — Expanded from current S6

**Narration**:
"Translation gave us the words. The vectors gave us the world between them."

**Visual**: Final text with repo link. Glyph strip fades across top.

## Timing Summary

### Full version

| Section | Duration |
|---------|----------|
| S1 Hook | ~25s |
| S2 Idea | ~30s |
| S3 Alignment | ~40s |
| S4 Journey | ~20s |
| D1–D8 (8 discoveries) | ~3:40 |
| S6 Discussion | ~30s |
| S7 Conclusion | ~15s |
| **Total** | **~6:20** |

### 3-minute cut

| Section | Duration |
|---------|----------|
| S1 Hook | ~25s |
| S2 Idea | ~30s |
| S3 Alignment | ~35s |
| S4 Journey | ~18s |
| 3 discoveries (Gold, Silence, Temple) | ~1:15 |
| Conclusion | ~10s |
| **Total** | **~2:53** |

## Technical details

### Data
- S3 uses `docs/viz_data.json` (already generated)
- Discovery scenes use illustrative/diagrammatic visualizations, not raw data plots
- May add 2 more highlights to viz_data.json for S3

### Audio
- OpenAI TTS: tts-1-hd model, "echo" voice
- Updated `audio_timing.json` for both versions
- Narration re-recorded for all changed/new segments
- Existing cached WAVs reused where narration text is unchanged

### Manim
- Single `heiroglyphy_video.py` file with all scenes
- `HeiroglyphyVideo` class composes full version
- `HeiroglyphyVideo3Min` class composes 3-minute cut
- Each discovery is its own Scene class (D1_Gold, D2_Silence, etc.)

### Palette
Unchanged: BG #0f0f1a, GOLD #f5c518, TEAL #3ec9a7, LAVENDER #c4b5fd, MUTED #888899, WHITE #f0f0f0, SOFT_RED #e74c3c
