# Video Narrative Redesign — Bridge Scores, Real Text, Full Circle

## Overview

Rework the video narrative to add analytical depth (bridge scores, midpoint scores), replace decorative glyphs with a real Pyramid Text passage (Utterance 213), and create a full-circle conclusion where the audience re-reads the opening text through the lens of the discoveries.

## Two deliverables
- **Full version** (~6 min): all 8 discoveries with metrics
- **3-minute cut**: S1, S2, S3, S_Bridge, 3 discoveries (Gold, Silence, Temple), Conclusion

## Audience
Layman-lite. Can handle two metrics explained simply. No equations, but comfortable with concepts like "similarity score" and "accuracy rate."

---

## Structure

### S1_Hook (~23s) — Updated opening text

Replace the decorative GLYPH_STRIP with the real hieroglyphs from Pyramid Text Utterance 213:

> *ꜥnḫ ꜥnḫ m rn =k pw ḫr nṯr.w ḫꜥi̯ m wpꞽ.w bꜣ ꞽs ḫnt ꜥnḫ.w sḫm ꞽs ḫnt ꜣḫ.w*

The glyphs appear large, beautiful, untranslated. The audience doesn't know what they say yet. Narration unchanged: "These symbols are four thousand years old..."

Need: Unicode hieroglyphic rendering of Utt. 213 to replace GLYPH_STRIP.

### S2_Idea (~20s) — Unchanged

Words as points in space. Clustering concept.

### S3_Alignment (~30s) — Remove accuracy beat

Keep the real data cloud visualization (viz_data.json). Remove the "32.35% accuracy" closing beat — that moves to S_Bridge.

The scene ends after the golden hits (mw→water, nṯr→god, nswt→king) with a brief pause, then fades.

### S_Bridge (~25s) — NEW: What the numbers mean

**Purpose**: Explain 32% accuracy and introduce the bridge score, so the discovery scenes have a framework.

**Narration** (simplified for layman-lite):
> "So how well does this work? For one in three Egyptian words, the nearest English word in the aligned space is the exact correct translation. One in three — with no dictionary, no bilingual text, just the shape of meaning.
>
> For each word, we get a bridge score — how closely the Egyptian word's neighborhood matches its English counterpart. A high bridge score means the word works the same way in both languages. And when we probe the space between words, we get a midpoint score — how strongly two concepts converge in the Egyptian worldview.
>
> Here's what those numbers revealed."

**Visualization**:
- "1 in 3" appears large, centered
- Brief visual: a single Egyptian dot finding its English match (animated from S3's data)
- Bridge score concept: two dots connected by a line, score label
- Midpoint score concept: two dots with a midpoint marker, similarity label
- Transition: "Here's what those numbers revealed" → fade to first discovery

### D1–D8 Discovery Scenes — Updated format

Each discovery scene adds an **info overlay** in the upper-right corner that appears mid-scene (after the visual context is established):

**Upper-right overlay format:**
```
┌─────────────────────┐
│ nṯr → "god"         │  ← literal dictionary translation
│ bridge: 0.642       │  ← bridge score (cosine sim to correct translation)
│ midpoint: 0.617     │  ← midpoint score (query similarity)
└─────────────────────┘
```

- Small, muted styling so it doesn't dominate
- Appears ~8-10s into the scene, after the visual has established context
- Stays visible through the punchline

**Per-discovery data needed:**
Pull from aligned embeddings:
1. D1_Gold: nṯr bridge score, midpoint("gold","divine") → nṯr similarity
2. D2_Silence: mwt bridge score, midpoint("silence","death") → mwt similarity
3. D3_Seeing: jr.t bridge score, midpoint("eye","knowledge") → ḥkꜣ similarity
4. D4_Snake: snake term bridge score, midpoint("snake","wisdom") → nṯr similarity
5. D5_Temple: house→temple::man→? analogy score
6. D6_Mother: mother term bridge score, midpoint("mother","earth") → royal results similarity
7. D7_Truth: māʿat bridge score, midpoint("truth","power") → sḫm similarity
8. D8_Eternity: love/fear terms, midpoint("love","fear") → r-nḥḥ similarity

**Literal translations** (hand-curated, one-liner per term):
1. D1: nṯr → "god, divine being" / nbw → "gold"
2. D2: mwt → "to die, dead"
3. D3: jr.t → "eye" / ḥkꜣ → "magic, magical power"
4. D4: (snake glyph) → "serpent, snake"
5. D5: pr → "house" / ḥw.t-nṯr → "temple" / nṯr → "god"
6. D6: mw.t → "mother" / ḥm.t-nswt → "king's wife"
7. D7: māʿat → "truth, justice" / sḫm → "power, authority"
8. D8: r-nḥḥ → "to eternity, forever"

### S6_Discussion (~30s) — Unchanged

Honest caveats. Corpus bias, 32% accuracy, what translation misses.

### S7_Conclusion (~30s) — Rebuilt: Full circle

**Phase 1** (~5s): The Utt. 213 glyphs from S1 reappear — the audience recognizes them.

**Phase 2** (~8s): The literal translation appears below:
> "Live! Live! — for this is your name among the gods. You appear as the opener of ways. A soul indeed, foremost of the living. Powerful indeed, foremost of the spirits."

**Phase 3** (~12s): Word-by-word reframing. Key terms highlight and their embedding-informed meaning appears:
- "Live" (ꜥnḫ) → highlights → "not biological life — divine permanence"
- "gods" (nṯr) → highlights → "indistinguishable from gold — ontological, not metaphorical"
- "soul" (bꜣ) → highlights → "the animating force — not ethereal, but powerful"
- "powerful" (sḫm) → highlights → "the same force as truth — māʿat"
- "spirits" (ꜣḫ.w) → highlights → "the transfigured — those who achieved power through knowledge"

Each reframing is quick (2-3s per term), stacking to show how the entire passage transforms.

**Phase 4** (~5s):
> "Translation gave us the words. The vectors gave us the world between them."
>
> github.com/ebrinz/heiroglyphy

---

## Data Pipeline

### Bridge scores extraction

Create a script `docs/extract_bridge_scores.py` that:
1. Loads `final_output/egyptian_aligned_vectors.npz` and `final_output/concept_vectors.npz`
2. For each discovery term, computes:
   - Bridge score: cosine similarity between the aligned Egyptian word and its English translation
   - Midpoint score: cosine similarity between the English midpoint query and the Egyptian result
3. Outputs `docs/bridge_scores.json`

### Utterance 213 glyphs

Find or compose the Unicode hieroglyphic rendering of Utt. 213 key phrase. Update GLYPH_STRIP to use these real glyphs.

---

## Technical changes

### Files modified
- `docs/heiroglyphy_video.py` — GLYPH_STRIP update, remove S4_Journey, remove accuracy beat from S3, add S_Bridge scene, update D1-D8 with overlay, rebuild S7_Conclusion
- `docs/audio/audio_timing.json` — remove S4, add S_Bridge narration, update S7 narration
- `docs/audio/audio_timing_3min.json` — matching changes for short cut

### Files created
- `docs/extract_bridge_scores.py` — compute bridge/midpoint scores
- `docs/bridge_scores.json` — extracted scores for video use

### Palette / styling
Unchanged. Info overlay uses MUTED color, small scale (~0.2-0.25), positioned upper-right.
