# Video Narrative Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the Manim video with bridge scores, real Pyramid Text passage (Utt. 213), and a full-circle conclusion that reframes the opening text.

**Architecture:** Extract bridge/midpoint scores from aligned embeddings into a JSON file. Update GLYPH_STRIP to real Utt. 213 hieroglyphs. Remove S4_Journey, trim S3 accuracy beat, add S_Bridge scene, add score overlays to D1-D8, rebuild S7 conclusion. Render scene-by-scene for iterative review.

**Tech Stack:** Manim Community Edition, numpy (embeddings), OpenAI TTS (tts-1-hd, echo), scipy (audio), ffmpeg

**Spec:** `docs/superpowers/specs/2026-03-15-video-narrative-redesign.md`

**Note:** The project uses pickle to load its own serialized vocabulary file (`egyptian_aligned_vocab.pkl`). This is the project's own data, not untrusted external content.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `docs/extract_bridge_scores.py` | Create | Compute bridge + midpoint scores for all discovery terms |
| `docs/bridge_scores.json` | Create (generated) | Score data consumed by video scenes |
| `docs/heiroglyphy_video.py` | Modify | GLYPH_STRIP, remove S4, trim S3, add S_Bridge, update D1-D8 overlays, rebuild S7 |
| `docs/audio/audio_timing.json` | Rewrite | Remove S4, add S_Bridge, update S7 narration |
| `docs/audio/audio_timing_3min.json` | Rewrite | Matching changes for 3-min cut |

---

## Chunk 1: Extract Bridge Scores

### Task 1: Create bridge score extraction script

**Files:**
- Create: `docs/extract_bridge_scores.py`
- Create: `docs/bridge_scores.json` (output)

- [ ] **Step 1: Write extract_bridge_scores.py**

This script loads the aligned Egyptian and English embeddings and computes cosine similarities for each discovery. It uses pickle to load the project's own vocabulary file (egyptian_aligned_vocab.pkl).

The script should:
1. Load `final_output/egyptian_aligned_vectors.npz` and vocabulary
2. Load `final_output/concept_vectors.npz` with English concepts
3. For each of the 8 discoveries, compute:
   - Bridge score: cosine similarity between aligned Egyptian word and English translation
   - Midpoint score: cosine similarity of midpoint(English_A, English_B) to nearest Egyptian result
4. Output results to `docs/bridge_scores.json`

Discovery computations:
- D1_Gold: bridge(nṯr, "god"), midpoint("gold","divine") → nearest Egyptian
- D2_Silence: bridge(mwt, "death"), midpoint("silence","death") → nearest Egyptian
- D3_Seeing: bridge(jr.t, "eye"), midpoint("eye","knowledge") → nearest Egyptian
- D4_Snake: midpoint("snake","wisdom") → nearest Egyptian (no direct bridge for snake)
- D5_Temple: analogy vector (temple - house + man) → nearest Egyptian
- D6_Mother: bridge(mw.t, "mother"), midpoint("mother","earth") → nearest Egyptian
- D7_Truth: bridge(mꜣꜥ.t, "truth"), midpoint("truth","power") → nearest Egyptian
- D8_Eternity: midpoint("love","fear") → nearest Egyptian, check for r-nḥḥ

Include literal translations (hand-curated) in the output:
- nṯr: "god, divine being"
- nbw: "gold"
- mwt: "to die, dead"
- jr.t: "eye"
- ḥkꜣ: "magic, magical power"
- mw.t: "mother"
- mꜣꜥ.t: "truth, justice, cosmic order"
- sḫm: "power, authority"
- r-nḥḥ: "to eternity, forever"

- [ ] **Step 2: Run extraction**

```bash
cd /Users/crashy/Development/heiroglyphy && python docs/extract_bridge_scores.py
```

Expected: Creates `docs/bridge_scores.json` with scores for all 8 discoveries.

- [ ] **Step 3: Commit**

```bash
git add docs/extract_bridge_scores.py docs/bridge_scores.json
git commit -m "feat: extract bridge and midpoint scores for discovery scenes"
```

---

## Chunk 2: Update GLYPH_STRIP and Remove S4

### Task 2: Replace GLYPH_STRIP with Utterance 213

**Files:**
- Modify: `docs/heiroglyphy_video.py` (lines 50-70, HIERO dict and GLYPH_STRIP)

- [ ] **Step 1: Research Utt. 213 Unicode hieroglyphs**

Search `final_output/hieroglyph_dictionary.json` for the component words of Utterance 213:
- ꜥnḫ (life/ankh) — transliteration "anx"
- nṯr (god) — transliteration "nTr"
- bꜣ (soul) — transliteration "bA"
- sḫm (power) — transliteration "sxm"
- ꜣḫ (spirit) — transliteration "Ax"

Extract the Unicode hieroglyphic characters for each.

```bash
cd /Users/crashy/Development/heiroglyphy && python3 -c "
import json
with open('final_output/hieroglyph_dictionary.json') as f:
    d = json.load(f)
targets = ['anx', 'nTr', 'bA', 'sxm', 'Ax', 'rn', 'xnt']
for entry in d:
    t = entry.get('transliteration', '')
    for target in targets:
        if t == target or t.replace(' ', '') == target:
            print(f'{t}: {entry[\"hieroglyph\"]} (gardiner: {entry.get(\"gardiner_codes\", \"\")})')
            break
"
```

- [ ] **Step 2: Update GLYPH_STRIP and add UTT_213 data**

Replace the decorative GLYPH_STRIP with real Utt. 213 hieroglyphs. Add a UTT_213 dict containing:
- glyphs: Unicode codepoints for each key word
- literal: the full literal translation string
- reframed: list of (english_word, egyptian_term, insight) tuples for the conclusion

The exact Unicode codepoints come from Step 1. Example structure:

```python
UTT_213 = {
    "glyphs": { "ankh": "...", "god": "...", "soul": "...", "power": "...", "spirit": "..." },
    "literal": "Live! Live! — for this is your name among the gods.\nA soul indeed, foremost of the living.\nPowerful indeed, foremost of the spirits.",
    "reframed": [
        ("Live", "ꜥnḫ", "not biological life — divine permanence"),
        ("gods", "nṯr", "indistinguishable from gold — ontological, not metaphorical"),
        ("soul", "bꜣ", "the animating force — not ethereal, but powerful"),
        ("powerful", "sḫm", "the same force as truth — māʿat"),
        ("spirits", "ꜣḫ.w", "the transfigured — those who achieved power through knowledge"),
    ],
}
GLYPH_STRIP = " ".join([UTT_213["glyphs"][k] for k in ["ankh", "ankh", "god", "god", "soul", "ankh", "power", "spirit"]])
```

- [ ] **Step 3: Test render S1_Hook with new glyphs**

```bash
cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S1_Hook
```

Expected: Real Utt. 213 hieroglyphs render correctly.

- [ ] **Step 4: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: replace decorative glyphs with Pyramid Text Utt. 213"
```

### Task 3: Remove S4_Journey and trim S3 accuracy beat

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Remove S4_Journey class entirely**

Delete the entire S4_Journey class and its comment block.

- [ ] **Step 2: Remove accuracy beat from end of S3_Alignment**

Remove the "Phase 6: Accuracy" section at the end of S3_Alignment (the "32.35% accuracy" text). Replace with a simple `self.wait(3)` so the golden hits linger before fading.

- [ ] **Step 3: Update compositor classes**

In HeiroglyphyVideo, replace the scene list with:
```python
S1_Hook, S2_Idea, S3_Alignment, S_Bridge,
D1_Gold, D2_Silence, D3_Seeing, D4_Snake,
D5_Temple, D6_Mother, D7_Truth, D8_Eternity,
S6_Discussion, S7_Conclusion,
```

In HeiroglyphyVideo3Min, replace with:
```python
S1_Hook, S2_Idea, S3_Alignment, S_Bridge,
D1_Gold, D2_Silence, D5_Temple,
S7_Conclusion,
```

Note: S_Bridge will be created in Task 4. Add it to the list now — Python will fail until Task 4 is done, but that's fine since we render scene-by-scene.

- [ ] **Step 4: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: remove S4_Journey, trim S3 accuracy beat, update compositors"
```

---

## Chunk 3: Add S_Bridge Scene

### Task 4: Create S_Bridge scene

**Files:**
- Modify: `docs/heiroglyphy_video.py` (add new class after S3_Alignment)

- [ ] **Step 1: Write S_Bridge scene class**

The scene has three phases (~25s total):

Phase 1 — "1 in 3" stat (~8s):
- Large "1 in 3" text in GOLD
- Subtitle: "Egyptian words land on their correct English meaning"
- Qualifier: "No dictionary. No bilingual text. Just the shape of meaning."

Phase 2 — Bridge score concept (~8s, left side):
- Title: "Bridge score" in TEAL
- Two dots (Egyptian GOLD, English TEAL) connected by a line
- Labels: "nṯr" and "god" with score "0.642"
- Description: "How closely an Egyptian word's neighborhood matches its English counterpart"

Phase 3 — Midpoint score concept (~8s, right side):
- Title: "Midpoint score" in TEAL
- Two English dots with a midpoint marker between them
- Labels: "gold" and "divine" with score
- Description: "How strongly two concepts converge in the Egyptian worldview"

Closing: "Here's what those numbers revealed." in LAVENDER

Use actual scores from bridge_scores.json if available, otherwise hardcode representative values.

- [ ] **Step 2: Test render S_Bridge**

```bash
cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S_Bridge
```

Expected: Renders ~25s with stat, bridge concept, midpoint concept.

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add S_Bridge scene explaining accuracy and score concepts"
```

---

## Chunk 4: Add Score Overlays to Discovery Scenes

### Task 5: Create score overlay helper

**Files:**
- Modify: `docs/heiroglyphy_video.py` (add helper function)

- [ ] **Step 1: Add BRIDGE_SCORES path constant and load function**

Near the other path constants:
```python
BRIDGE_SCORES = DOCS_DIR / "bridge_scores.json"
```

Add a load function near other helpers:
```python
def load_bridge_scores():
    with open(BRIDGE_SCORES) as f:
        return json.load(f)
```

- [ ] **Step 2: Add score_overlay helper function**

```python
def score_overlay(scene, term, literal, bridge, midpoint):
    """Show bridge + midpoint score overlay in upper-right. Returns VGroup."""
    lines = []
    lines.append(Text(f"{term} → \"{literal}\"", color=MUTED).scale(0.22))
    if bridge is not None:
        lines.append(Text(f"bridge: {bridge:.3f}", color=TEAL).scale(0.2))
    if midpoint is not None:
        lines.append(Text(f"midpoint: {midpoint:.3f}", color=LAVENDER).scale(0.2))

    overlay = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
    overlay.move_to(RIGHT * 5.5 + UP * 2.8)
    bg = SurroundingRectangle(overlay, color=MUTED, fill_color=BG,
                               fill_opacity=0.85, stroke_width=0.5, buff=0.1)
    overlay_group = VGroup(bg, overlay)
    scene.play(FadeIn(overlay_group), run_time=1)
    return overlay_group
```

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add score overlay helper for discovery scenes"
```

### Task 6: Add overlays to D1–D8

**Files:**
- Modify: `docs/heiroglyphy_video.py` (all 8 discovery scene classes)

- [ ] **Step 1: Add overlay to each discovery scene**

In each discovery scene's construct(), after the key visual reveal (and before the punchline), add:

```python
scores = load_bridge_scores()["discoveries"]["D1_Gold"]  # use appropriate key
overlay = score_overlay(
    self,
    term=scores["primary_term"],
    literal=scores["literal"],
    bridge=scores.get("bridge_score"),
    midpoint=scores.get("midpoint_score") or scores.get("analogy_score"),
)
```

Place this call at these points in each scene:
- D1_Gold: after Egyptian results (nṯr/nbw dots) appear
- D2_Silence: after mwt variants cluster appears
- D3_Seeing: after triangle of meaning forms
- D4_Snake: after Egyptian side is revealed
- D5_Temple: after "?" resolves to "god"
- D6_Mother: after actual (royal) results light up
- D7_Truth: after māʿat appears at center
- D8_Eternity: after eternity label appears

- [ ] **Step 2: Test render one discovery scene**

```bash
cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D1_Gold
```

Expected: Score overlay appears in upper-right with bridge and midpoint values.

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add score overlays to all discovery scenes"
```

---

## Chunk 5: Rebuild S7 Conclusion

### Task 7: Rebuild S7_Conclusion with full-circle reframing

**Files:**
- Modify: `docs/heiroglyphy_video.py` (S7_Conclusion class)

- [ ] **Step 1: Replace S7_Conclusion with full-circle version**

Four phases:

Phase 1 (~5s): The same GLYPH_STRIP from S1 reappears, large, golden.

Phase 2 (~8s): The literal translation appears below the glyphs (from UTT_213["literal"]).

Phase 3 (~12s): Word-by-word reframing. For each entry in UTT_213["reframed"]:
- Show "english_word (eg_term)" in GOLD
- Show "→ insight" in LAVENDER
- Each row fades in with a slight right-shift, ~1.5s per term + 0.5s pause

Phase 4 (~5s):
- Everything fades
- "Translation gave us the words." (WHITE)
- "The vectors gave us the world between them." (LAVENDER)
- Repo link

- [ ] **Step 2: Test render S7**

```bash
cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S7_Conclusion
```

Expected: Glyphs return → literal → reframing → closing. ~30s.

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: rebuild S7 conclusion with full-circle Utt. 213 reframing"
```

---

## Chunk 6: Audio and Final Assembly

### Task 8: Update audio timing files

**Files:**
- Rewrite: `docs/audio/audio_timing.json`
- Rewrite: `docs/audio/audio_timing_3min.json`

- [ ] **Step 1: Write updated audio_timing.json**

Key changes:
- Remove S4_Journey scene entirely
- Remove s3_05 segment ("We got it right thirty-two percent...")
- Add S_Bridge scene with segments:
  - sb_01: "So how well does this work? For one in three Egyptian words, the nearest English word in the aligned space is the exact correct translation. One in three — with no dictionary, no bilingual text, just the shape of meaning."
  - sb_02: "For each word, we get a bridge score — how closely the Egyptian word's neighborhood matches its English counterpart. And when we probe the space between words, we get a midpoint score — how strongly two concepts converge."
  - sb_03: "Here's what those numbers revealed."
- Update S7_Conclusion with new segments:
  - s7_01: "These are the same symbols we started with. From the Pyramid Texts, carved five thousand years ago."
  - s7_02: "Live, live — for this is your name among the gods. A soul indeed, foremost of the living. Powerful indeed, foremost of the spirits."
  - s7_03: "But now we can read between the words. Live doesn't mean breathe. It means endure as the gods endure. Soul isn't ethereal. It's the force that animates. And powerful — that's the same force as truth."
  - s7_04: "Translation gave us the words. The vectors gave us the world between them."
- Recalculate all start/end times to account for removed S4 and added S_Bridge

- [ ] **Step 2: Write updated audio_timing_3min.json**

Same structure, scenes: S1, S2, S3, S_Bridge, D1, D2, D5, S7.

- [ ] **Step 3: Commit**

```bash
git add docs/audio/audio_timing.json docs/audio/audio_timing_3min.json
git commit -m "feat: update audio timing — remove S4, add S_Bridge, update S7"
```

### Task 9: Generate TTS, render, and stitch

- [ ] **Step 1: Clear changed voice segments and regenerate**

```bash
rm -f docs/audio/voice/voice_s3_05.wav docs/audio/voice/voice_s4_*.wav
rm -f docs/audio/voice/voice_sb_*.wav docs/audio/voice/voice_s7_*.wav
rm -f docs/audio/voice_full.wav docs/audio/mixed_audio.wav
cd /Users/crashy/Development/heiroglyphy/docs/audio && python generate_voice.py
```

- [ ] **Step 2: Render each scene individually**

```bash
cd /Users/crashy/Development/heiroglyphy/docs
for scene in S1_Hook S2_Idea S3_Alignment S_Bridge D1_Gold D2_Silence D3_Seeing D4_Snake D5_Temple D6_Mother D7_Truth D8_Eternity S6_Discussion S7_Conclusion; do
    manim -qh heiroglyphy_video.py $scene
done
```

For each scene, create a preview with its narration and review timing. Iterate as needed (adjusting self.wait() values).

- [ ] **Step 3: Stitch scenes with ffmpeg**

Create a concat file listing all scene mp4s in order, then concatenate:

```bash
cd /Users/crashy/Development/heiroglyphy/docs

# Full version
ls -1 media/videos/heiroglyphy_video/1080p60/{S1_Hook,S2_Idea,S3_Alignment,S_Bridge,D1_Gold,D2_Silence,D3_Seeing,D4_Snake,D5_Temple,D6_Mother,D7_Truth,D8_Eternity,S6_Discussion,S7_Conclusion}.mp4 | sed "s/^/file '/" | sed "s/$/'/" > concat_full.txt
ffmpeg -f concat -safe 0 -i concat_full.txt -c copy media/HeiroglyphyVideo_stitched.mp4

# 3-minute cut
ls -1 media/videos/heiroglyphy_video/1080p60/{S1_Hook,S2_Idea,S3_Alignment,S_Bridge,D1_Gold,D2_Silence,D5_Temple,S7_Conclusion}.mp4 | sed "s/^/file '/" | sed "s/$/'/" > concat_3min.txt
ffmpeg -f concat -safe 0 -i concat_3min.txt -c copy media/HeiroglyphyVideo3Min_stitched.mp4
```

- [ ] **Step 4: Generate drone and mix audio**

```bash
cd /Users/crashy/Development/heiroglyphy/docs/audio
python generate_drone.py
python mix_audio.py
```

May need to update mix_audio.py to point at the stitched video path.

- [ ] **Step 5: Final commit**

```bash
git add docs/heiroglyphy_video.py docs/audio/audio_timing.json docs/audio/audio_timing_3min.json
git commit -m "feat: complete narrative redesign — bridge scores, Utt. 213, full circle"
```
