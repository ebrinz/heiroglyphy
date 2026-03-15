# Video Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Manim explainer video with real embedding data in S3, 8 bespoke discovery scenes, and produce both a full (~6:20) and 3-minute cut.

**Architecture:** Single `heiroglyphy_video.py` with modular Scene classes. Each discovery is its own class. Two compositor classes (`HeiroglyphyVideo` for full, `HeiroglyphyVideo3Min` for short). Audio pipeline (TTS → drone → mix → ffmpeg) runs after Manim renders.

**Tech Stack:** Manim Community Edition, OpenAI TTS (tts-1-hd, echo), scipy (audio), ffmpeg

**Spec:** `docs/superpowers/specs/2026-03-14-video-redesign.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `docs/heiroglyphy_video.py` | Modify | All Manim scenes — S1-S4 wait tightening, S3 real data, 8 discovery scenes, discussion, conclusion, two compositor classes |
| `docs/audio/audio_timing.json` | Rewrite | New narration text + timings for full version |
| `docs/audio/audio_timing_3min.json` | Create | Narration text + timings for 3-minute cut |
| `docs/generate_viz_data.py` | Modify | Add 2 more highlight pairs if needed |

Pipeline scripts unchanged: `generate_voice.py`, `generate_drone.py`, `mix_audio.py`

---

## Chunk 1: Tighten S1–S4 Waits + S3 Real Data

### Task 1: Tighten S1 Hook waits

**Files:**
- Modify: `docs/heiroglyphy_video.py` (S1_Hook class, lines 109–143)

- [ ] **Step 1: Reduce S1 wait times**

Replace the wait times in S1_Hook. Current waits total ~14s, target ~8s:

```python
class S1_Hook(Scene):
    def construct(self):
        glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=0.7)
        glyphs.move_to(UP * 1.5)

        # [Glyphs appear] "These symbols are four thousand years old."
        self.play(FadeIn(glyphs, shift=UP * 0.2), run_time=3)
        self.wait(1.5)

        # "Scholars have been translating them for two centuries."
        line1 = body_text(
            "For 200 years, scholars have translated these symbols.",
            color=WHITE
        ).scale(1.1).next_to(glyphs, DOWN, buff=0.8)
        self.play(Write(line1), run_time=3)
        self.wait(1.5)

        # "But translation is lossy..."
        line2 = body_text(
            "But translation compresses meaning.\n"
            "The relationships between words are lost.",
            color=MUTED
        ).scale(1.0).next_to(line1, DOWN, buff=0.5)
        self.play(Write(line2), run_time=3.5)
        self.wait(2)

        # "What if we could get it back?"
        line3 = body_text(
            "What if we could recover them?",
            color=LAVENDER
        ).scale(1.1).next_to(line2, DOWN, buff=0.6)
        self.play(FadeIn(line3, shift=UP * 0.1), run_time=2)
        self.wait(2)
```

- [ ] **Step 2: Test render S1**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S1_Hook`
Expected: Renders without error, visually tighter pacing (~25s)

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "perf: tighten S1 Hook wait times"
```

---

### Task 2: Tighten S2 Idea waits

**Files:**
- Modify: `docs/heiroglyphy_video.py` (S2_Idea class, lines 156–214)

- [ ] **Step 1: Reduce S2 wait times**

Current waits total ~10s, target ~6s:

```python
class S2_Idea(Scene):
    def construct(self):
        # "Here's the key insight."
        header = title_text("Words live in space", color=WHITE, scale=0.75)
        header.move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)
        self.wait(1)

        cluster_words = [
            ("water", -1.5, 0.8),
            ("river", -0.7, 1.2),
            ("flood", -1.8, 1.5),
            ("fish", -0.5, 0.5),
            ("boat", -1.2, 0.3),
        ]

        dots_and_labels = VGroup()
        for word, x, y in cluster_words:
            dot = Dot(point=[x, y, 0], radius=0.08, color=TEAL).set_opacity(0.8)
            label = Text(word, color=TEAL).scale(0.3).next_to(dot, RIGHT, buff=0.1)
            dots_and_labels.add(dot, label)

        cluster_words2 = [
            ("king", 1.5, 0.6),
            ("throne", 2.0, 1.0),
            ("crown", 1.2, 1.1),
            ("queen", 1.8, 0.2),
            ("palace", 1.0, 0.5),
        ]

        for word, x, y in cluster_words2:
            dot = Dot(point=[x, y, 0], radius=0.08, color=TEAL).set_opacity(0.8)
            label = Text(word, color=TEAL).scale(0.3).next_to(dot, RIGHT, buff=0.1)
            dots_and_labels.add(dot, label)

        self.play(
            LaggedStart(*[FadeIn(m) for m in dots_and_labels], lag_ratio=0.08),
            run_time=4
        )
        self.wait(1.5)

        explain = body_text(
            "Words that appear in similar contexts\n"
            "end up close together in this space.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 1.5)
        self.play(Write(explain), run_time=3)
        self.wait(1.5)

        explain2 = body_text(
            "This works for every language — including Ancient Egyptian.",
            color=LAVENDER
        ).scale(1.0).move_to(DOWN * 2.8)
        self.play(FadeIn(explain2, shift=UP * 0.1), run_time=2)
        self.wait(2)
```

- [ ] **Step 2: Test render S2**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S2_Idea`
Expected: Renders without error (~30s)

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "perf: tighten S2 Idea wait times"
```

---

### Task 3: Rebuild S3 Alignment with real data

**Files:**
- Modify: `docs/heiroglyphy_video.py` (S3_Alignment class, lines 230–303)

This is the biggest single change. Replace the fake random clouds with real viz_data.json points.

- [ ] **Step 1: Add data loading helper**

Add after the existing helpers (after `make_cloud` function around line 96):

```python
def load_viz_data():
    """Load real embedding projections from viz_data.json."""
    with open(VIZ_DATA) as f:
        return json.load(f)


def normalize_points(points, target_range=2.5, center=(0, 0)):
    """Normalize (x, y) points to fit within target_range of center."""
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min or 1
    y_span = y_max - y_min or 1
    scale = target_range / max(x_span, y_span)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    result = []
    for p in points:
        nx = (p["x"] - x_mid) * scale + center[0]
        ny = (p["y"] - y_mid) * scale + center[1]
        result.append((nx, ny, p))
    return result
```

- [ ] **Step 2: Rewrite S3_Alignment**

Replace the entire S3_Alignment class:

```python
class S3_Alignment(Scene):
    def construct(self):
        data = load_viz_data()

        # Normalize each cloud to fit its half of the screen
        eg_norm = normalize_points(data["egyptian"], target_range=2.0, center=(-3, 0))
        en_norm = normalize_points(data["english"], target_range=2.0, center=(3, 0))

        # Phase 1: Two clouds appear
        eg_cloud = VGroup()
        for nx, ny, pt in eg_norm:
            dot = Dot(point=[nx, ny, 0], radius=0.025, color=GOLD)
            dot.set_opacity(0.6)
            eg_cloud.add(dot)

        en_cloud = VGroup()
        for nx, ny, pt in en_norm:
            dot = Dot(point=[nx, ny, 0], radius=0.025, color=TEAL)
            dot.set_opacity(0.6)
            en_cloud.add(dot)

        label_eg = VGroup(
            hiero_text(HIERO["eye"], color=GOLD, scale=0.35),
            Text("Egyptian", color=GOLD).scale(0.45),
        ).arrange(RIGHT, buff=0.15).move_to(LEFT * 3 + UP * 2.8)

        label_en = Text("English", color=TEAL).scale(0.45)
        label_en.move_to(RIGHT * 3 + UP * 2.8)

        # Subtle axis labels
        ax_divine = Text("divine →", color=MUTED).scale(0.2).move_to(RIGHT * 6.5 + UP * 0)
        ax_mortal = Text("← mortal", color=MUTED).scale(0.2).move_to(LEFT * 6.5 + UP * 0)
        ax_life = Text("life ↑", color=MUTED).scale(0.2).move_to(UP * 3.5 + RIGHT * 0)
        ax_death = Text("↓ death", color=MUTED).scale(0.2).move_to(DOWN * 3.5 + RIGHT * 0)
        axes_labels = VGroup(ax_divine, ax_mortal, ax_life, ax_death)

        self.play(FadeIn(axes_labels, run_time=1))
        self.play(FadeIn(label_eg), Create(eg_cloud), run_time=3)
        self.wait(1)
        self.play(FadeIn(label_en), Create(en_cloud), run_time=3)
        self.wait(1)

        # Phase 2: "The shapes are similar — but rotated."
        explain = body_text(
            "Both languages form a shape.\n"
            "The shapes are similar — but rotated.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 3.0)
        self.play(Write(explain), run_time=3)
        self.wait(1.5)

        # Phase 3: Clouds merge — Egyptian scales/translates/rotates to overlap English
        self.play(FadeOut(explain), run_time=0.5)
        finding = body_text("Find the rotation...", color=LAVENDER).scale(1.1)
        finding.move_to(DOWN * 3.0)
        self.play(FadeIn(finding), run_time=1)

        # Compute transform: move Egyptian cloud center to English cloud center
        # The Egyptian space is compressed — this scaling is real
        eg_center = eg_cloud.get_center()
        en_center = en_cloud.get_center()
        shift_vec = en_center - eg_center

        self.play(
            eg_cloud.animate.shift(shift_vec).scale(1.3).rotate(0.25),
            label_eg.animate.shift(RIGHT * 2),
            run_time=5, rate_func=smooth
        )
        self.wait(0.5)

        # Phase 4: Anchor lines
        self.play(FadeOut(finding), run_time=0.5)
        overlap = body_text(
            "...and the words align across 4,000 years.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 3.0)
        self.play(FadeIn(overlap), run_time=1.5)

        # Draw real anchor connections (pick ~10 with shortest distances after merge)
        anchor_lines = VGroup()
        rng = np.random.default_rng(42)
        anchor_indices = rng.choice(len(eg_cloud), size=min(10, len(eg_cloud)), replace=False)
        for idx in anchor_indices:
            eg_dot = eg_cloud[int(idx)]
            # Find nearest English dot
            eg_pos = eg_dot.get_center()
            dists = [np.linalg.norm(eg_pos - en_cloud[j].get_center()) for j in range(len(en_cloud))]
            nearest = int(np.argmin(dists))
            line = Line(
                eg_pos, en_cloud[nearest].get_center(),
                color=LAVENDER, stroke_width=1
            ).set_opacity(0.4)
            anchor_lines.add(line)

        self.play(Create(anchor_lines), run_time=2)
        self.wait(1)

        # Phase 5: Golden hits — highlight real matched pairs
        self.play(FadeOut(anchor_lines), FadeOut(overlap), run_time=0.8)

        highlights = data.get("highlights", [])
        highlight_lines = VGroup()
        highlight_labels = VGroup()

        for hl in highlights:
            # Find the Egyptian and English dots closest to the highlight coordinates
            eg_target = np.array([hl["eg_x"], hl["eg_y"]])
            en_target = np.array([hl["en_x"], hl["en_y"]])

            # Find nearest dot in each cloud (after normalization and transform)
            eg_dists = [np.linalg.norm(eg_cloud[i].get_center()[:2] - eg_cloud.get_center()[:2])
                        for i in range(len(eg_cloud))]
            # Use normalized positions stored in the cloud
            best_eg = eg_cloud[0]
            best_en = en_cloud[0]
            best_eg_d = float('inf')
            best_en_d = float('inf')
            for i, (nx, ny, pt) in enumerate(eg_norm):
                if pt["word"] == hl["egyptian"]:
                    best_eg = eg_cloud[i]
                    break
            for i, (nx, ny, pt) in enumerate(en_norm):
                if pt.get("word") == hl["english"]:
                    best_en = en_cloud[i]
                    break

            line = Line(
                best_eg.get_center(), best_en.get_center(),
                color=LAVENDER, stroke_width=2
            ).set_opacity(0.8)
            highlight_lines.add(line)

            eg_label = Text(hl["egyptian"], color=GOLD).scale(0.25)
            eg_label.next_to(best_eg, LEFT, buff=0.1)
            en_label = Text(hl["english"], color=TEAL).scale(0.25)
            en_label.next_to(best_en, RIGHT, buff=0.1)
            highlight_labels.add(eg_label, en_label)

        self.play(Create(highlight_lines), FadeIn(highlight_labels), run_time=2)
        self.wait(2)

        # Phase 6: Accuracy
        self.play(FadeOut(highlight_lines), FadeOut(highlight_labels), run_time=0.5)
        acc = body_text("32.35% accuracy — no dictionary needed.",
                        color=GOLD).scale(1.1).move_to(DOWN * 3.0)
        self.play(FadeIn(acc), run_time=2)
        self.wait(2)
```

- [ ] **Step 3: Test render S3**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S3_Alignment`
Expected: Renders with real data points, clouds merge, highlights labeled (~40s)

- [ ] **Step 4: Visual review and iterate**

Watch the preview. Check:
- Are both clouds clearly visible and distinct before merge?
- Does the merge animation look natural?
- Are the highlight labels readable?
- Do axis labels show without being distracting?

Fix any visual issues.

- [ ] **Step 5: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: rebuild S3 alignment with real viz_data.json"
```

---

### Task 4: Tighten S4 Journey waits

**Files:**
- Modify: `docs/heiroglyphy_video.py` (S4_Journey class)

- [ ] **Step 1: Reduce S4 wait times**

Current waits ~6.5s, target ~4s. Change these wait values:

```python
# After header FadeIn
self.wait(0.5)  # was 1

# After BERT bar (i == 2)
self.wait(1.0)  # was 1.5

# After all bars
self.wait(0.5)  # was 1

# After lesson text
self.wait(2)  # was 4
```

- [ ] **Step 2: Test render S4**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S4_Journey`
Expected: Renders, tighter pacing (~20s)

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "perf: tighten S4 Journey wait times"
```

---

## Chunk 2: Discovery Scenes D1–D4

### Task 5: D1 — Gold Is Divine Flesh

**Files:**
- Modify: `docs/heiroglyphy_video.py` (add new class after S4_Journey)

- [ ] **Step 1: Write D1_Gold scene**

```python
class D1_Gold(Scene):
    def construct(self):
        # Title bar
        glyph = hiero_text("\U000131B4\U00013208\U000130C3\U000130F1", color=GOLD, scale=0.4)
        title = Text("Gold Is Divine Flesh", color=GOLD).scale(0.55)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # Two concept dots on opposite sides
        dot_gold = Dot(point=[-3, 0, 0], radius=0.12, color="#f1c40f").set_opacity(0.9)
        dot_divine = Dot(point=[3, 0, 0], radius=0.12, color=LAVENDER).set_opacity(0.9)
        lbl_gold = Text("gold", color="#f1c40f").scale(0.4).next_to(dot_gold, DOWN, buff=0.15)
        lbl_divine = Text("divine", color=LAVENDER).scale(0.4).next_to(dot_divine, DOWN, buff=0.15)

        self.play(FadeIn(dot_gold, lbl_gold), FadeIn(dot_divine, lbl_divine), run_time=2)
        self.wait(1)

        # Midpoint marker
        midpoint = Dot(point=[0, 0, 0], radius=0.08, color=WHITE).set_opacity(0.6)
        mid_label = Text("midpoint", color=MUTED).scale(0.25).next_to(midpoint, UP, buff=0.1)
        dashed_left = DashedLine(dot_gold.get_center(), midpoint.get_center(), color=MUTED, stroke_width=1)
        dashed_right = DashedLine(midpoint.get_center(), dot_divine.get_center(), color=MUTED, stroke_width=1)

        self.play(Create(dashed_left), Create(dashed_right), FadeIn(midpoint, mid_label), run_time=2)
        self.wait(1)

        # Arrow projects down into "Egyptian space"
        eg_label = Text("Egyptian space", color=GOLD).scale(0.3).move_to(DOWN * 1.2)
        arrow = Arrow(midpoint.get_center(), DOWN * 1.5, color=MUTED, stroke_width=2)
        self.play(Create(arrow), FadeIn(eg_label), run_time=1.5)

        # Egyptian results appear
        dot_ntri = Dot(point=[-0.5, -2.2, 0], radius=0.1, color=GOLD)
        dot_nbw = Dot(point=[0.5, -2.2, 0], radius=0.1, color=GOLD)
        lbl_ntri = Text("nṭri (divine)", color=GOLD).scale(0.3).next_to(dot_ntri, DOWN, buff=0.1)
        lbl_nbw = Text("nbw (gold)", color=GOLD).scale(0.3).next_to(dot_nbw, DOWN, buff=0.1)

        self.play(FadeIn(dot_ntri, lbl_ntri), FadeIn(dot_nbw, lbl_nbw), run_time=2)
        self.wait(1.5)

        # Dots merge — they're the same point
        self.play(
            dot_ntri.animate.move_to([0, -2.2, 0]),
            dot_nbw.animate.move_to([0, -2.2, 0]),
            lbl_ntri.animate.move_to([-1.2, -2.7, 0]),
            lbl_nbw.animate.move_to([1.2, -2.7, 0]),
            run_time=2
        )

        # Glow effect
        glow = Dot(point=[0, -2.2, 0], radius=0.3, color=GOLD).set_opacity(0.3)
        self.play(FadeIn(glow), run_time=1)
        self.wait(1)

        # Punchline
        punchline = body_text(
            "Not metaphor. Ontology.",
            color=WHITE
        ).scale(1.2).move_to(DOWN * 3.5)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)
```

- [ ] **Step 2: Test render D1**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D1_Gold`
Expected: Renders ~30s, two dots merge into glowing point

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add D1 Gold Is Divine Flesh discovery scene"
```

---

### Task 6: D2 — Silence Is the Condition of the Dead

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write D2_Silence scene**

```python
class D2_Silence(Scene):
    def construct(self):
        glyph = hiero_text("\U000131EF\U0001337F\U000132BD", color=GOLD, scale=0.4)
        title = Text("Silence Is the Condition of the Dead", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # Sound wave — series of sine-wave dots
        wave_dots = VGroup()
        n_pts = 80
        for i in range(n_pts):
            x = (i / n_pts) * 10 - 5
            y = 0.6 * np.sin(i * 0.3) * np.exp(-abs(x) * 0.05)
            wave_dots.add(
                Dot(point=[x, y + 0.5, 0], radius=0.03, color=TEAL).set_opacity(0.7)
            )

        self.play(Create(wave_dots), run_time=2)
        self.wait(1)

        # Wave flattens to silence
        flat_targets = []
        for i, dot in enumerate(wave_dots):
            x = dot.get_center()[0]
            flat_targets.append(dot.animate.move_to([x, 0.5, 0]).set_opacity(0.2))

        self.play(*flat_targets, run_time=3, rate_func=smooth)
        self.wait(0.5)

        # "silence" and "death" dots converge
        dot_silence = Dot(point=[-2, -1, 0], radius=0.1, color=LAVENDER)
        dot_death = Dot(point=[2, -1, 0], radius=0.1, color=SOFT_RED)
        lbl_silence = Text("silence", color=LAVENDER).scale(0.35).next_to(dot_silence, DOWN, buff=0.1)
        lbl_death = Text("death", color=SOFT_RED).scale(0.35).next_to(dot_death, DOWN, buff=0.1)

        self.play(FadeIn(dot_silence, lbl_silence), FadeIn(dot_death, lbl_death), run_time=1.5)
        self.wait(1)

        # Converge to same point
        converge_pt = [0, -1.5, 0]
        self.play(
            dot_silence.animate.move_to(converge_pt),
            dot_death.animate.move_to(converge_pt),
            lbl_silence.animate.next_to(converge_pt, LEFT, buff=0.3),
            lbl_death.animate.next_to(converge_pt, RIGHT, buff=0.3),
            run_time=2.5
        )

        # m(w)t variants cluster at that point
        mwt_words = ["m.wt", "mt", "mwt", "mwt.w", "mt.t"]
        mwt_group = VGroup()
        rng = np.random.default_rng(7)
        for w in mwt_words:
            offset = rng.uniform(-0.3, 0.3, 2)
            lbl = Text(w, color=GOLD).scale(0.22)
            lbl.move_to([converge_pt[0] + offset[0], converge_pt[1] - 0.6 + offset[1], 0])
            mwt_group.add(lbl)

        self.play(FadeIn(mwt_group), run_time=1.5)
        self.wait(1.5)

        # Punchline
        punchline = body_text(
            "What the dead lost was not life. It was voice.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)
```

- [ ] **Step 2: Test render D2**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D2_Silence`
Expected: Renders ~30s, wave flattens, dots converge

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add D2 Silence Is Death discovery scene"
```

---

### Task 7: D3 — Seeing Was an Act of Magical Power

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write D3_Seeing scene**

```python
class D3_Seeing(Scene):
    def construct(self):
        glyph = hiero_text("\U00013080\U000133DB\U000131B4", color=GOLD, scale=0.4)
        title = Text("Seeing Was an Act of Magical Power", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # Eye of Horus glyph — large, center
        eye = hiero_text("\U00013080", color=GOLD, scale=2.0)
        eye.move_to(ORIGIN + UP * 0.3)
        self.play(FadeIn(eye, scale=0.8), run_time=2)
        self.wait(0.5)

        # Three concept points radiating outward
        concepts = [
            ("knowledge", UP * 2 + LEFT * 2.5, TEAL),
            ("spellcasting", UP * 2 + RIGHT * 2.5, LAVENDER),
            ("protection", DOWN * 1.8, "#e74c3c"),
        ]

        concept_dots = VGroup()
        concept_labels = VGroup()
        for word, pos, color in concepts:
            dot = Dot(point=pos, radius=0.1, color=color).set_opacity(0.8)
            label = Text(word, color=color).scale(0.35).next_to(dot, DOWN, buff=0.1)
            concept_dots.add(dot)
            concept_labels.add(label)

        self.play(
            LaggedStart(*[FadeIn(d) for d in concept_dots], lag_ratio=0.3),
            LaggedStart(*[FadeIn(l) for l in concept_labels], lag_ratio=0.3),
            run_time=2.5
        )

        # Vectors radiate from eye to each concept
        vectors = VGroup()
        for dot in concept_dots:
            vec = Arrow(
                eye.get_center(), dot.get_center(),
                color=GOLD, stroke_width=2, buff=0.3
            ).set_opacity(0.5)
            vectors.add(vec)

        self.play(Create(vectors), run_time=2)
        self.wait(1)

        # Triangle of meaning connects the three
        triangle = Polygon(
            *[d.get_center() for d in concept_dots],
            color=LAVENDER, stroke_width=1
        ).set_opacity(0.3).set_fill(LAVENDER, opacity=0.05)
        self.play(Create(triangle), run_time=1.5)

        # Eye pulses
        self.play(
            eye.animate.scale(1.15).set_opacity(1),
            run_time=0.5, rate_func=there_and_back
        )
        self.wait(1)

        # Punchline
        punchline = body_text(
            "Sight was not observation. It was power.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)
```

- [ ] **Step 2: Test render D3**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D3_Seeing`
Expected: Renders ~25s, eye with radiating vectors and triangle

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add D3 Seeing Is Magic discovery scene"
```

---

### Task 8: D4 — The Snake Is Divine, Not Wise

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write D4_Snake scene**

```python
class D4_Snake(Scene):
    def construct(self):
        glyph = hiero_text("\U00013196\U000132BD\U000131B4", color=GOLD, scale=0.4)
        title = Text("The Snake Is Divine, Not Wise", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # Dividing line
        divider = Line(UP * 2.5, DOWN * 2.5, color=MUTED, stroke_width=1).set_opacity(0.3)
        self.play(Create(divider), run_time=0.5)

        # Left side: Greek expectation
        greek_title = Text("Greek tradition", color=MUTED).scale(0.3).move_to(LEFT * 3 + UP * 2.2)
        snake_left = Text("🐍", color=MUTED).scale(0.8).move_to(LEFT * 3 + UP * 0.5)
        arrow_left = Arrow(LEFT * 3 + DOWN * 0, LEFT * 3 + DOWN * 1.2, color=MUTED, stroke_width=2)
        wisdom = Text("wisdom", color=MUTED).scale(0.4).move_to(LEFT * 3 + DOWN * 1.6)

        self.play(FadeIn(greek_title), FadeIn(snake_left), run_time=1.5)
        self.play(Create(arrow_left), FadeIn(wisdom), run_time=1.5)
        self.wait(0.5)

        # Right side: Egyptian reality
        egyptian_title = Text("Egyptian vectors", color=GOLD).scale(0.3).move_to(RIGHT * 3 + UP * 2.2)
        snake_right = hiero_text("\U00013196", color=GOLD, scale=0.7).move_to(RIGHT * 3 + UP * 0.5)
        arrow_right = Arrow(RIGHT * 3 + DOWN * 0, RIGHT * 3 + DOWN * 1.2, color=GOLD, stroke_width=2)
        gods = Text("the gods", color=GOLD).scale(0.4).move_to(RIGHT * 3 + DOWN * 1.6)

        self.play(FadeIn(egyptian_title), FadeIn(snake_right), run_time=1.5)
        self.play(Create(arrow_right), FadeIn(gods), run_time=1.5)
        self.wait(1)

        # Greek side fades, Egyptian side glows
        self.play(
            snake_left.animate.set_opacity(0.2),
            arrow_left.animate.set_opacity(0.2),
            wisdom.animate.set_opacity(0.2),
            greek_title.animate.set_opacity(0.2),
            gods.animate.scale(1.3),
            snake_right.animate.scale(1.2),
            run_time=2
        )
        self.wait(1)

        # Punchline
        punchline = body_text(
            "Two cultures, separated by geometry.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)
```

- [ ] **Step 2: Test render D4**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D4_Snake`
Expected: Renders ~25s, split screen with Greek fading

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add D4 Snake Is Divine discovery scene"
```

---

## Chunk 3: Discovery Scenes D5–D8

### Task 9: D5 — Temple Is to House as God Is to Man

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write D5_Temple scene**

```python
class D5_Temple(Scene):
    def construct(self):
        glyph = hiero_text("\U00013250\U000132BD\U00013000", color=GOLD, scale=0.4)
        title = Text("Temple : House :: God : Man", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # Four points of the parallelogram
        pts = {
            "house":  [-2.5, -1, 0],
            "temple": [-2.5, 1.5, 0],
            "man":    [2.5, -1, 0],
            "?":      [2.5, 1.5, 0],
        }

        dots = {}
        labels = {}
        for word, pos in pts.items():
            color = TEAL if word != "?" else MUTED
            dot = Dot(point=pos, radius=0.12, color=color)
            label = Text(word, color=color).scale(0.4).next_to(dot, DOWN, buff=0.15)
            dots[word] = dot
            labels[word] = label

        # Show house, temple, man first
        self.play(
            *[FadeIn(dots[w], labels[w]) for w in ["house", "temple", "man"]],
            FadeIn(dots["?"], labels["?"]),
            run_time=2
        )
        self.wait(0.5)

        # Arrow from house → temple (labeled "sacred")
        arrow_left = Arrow(
            dots["house"].get_center(), dots["temple"].get_center(),
            color=LAVENDER, stroke_width=3, buff=0.2
        )
        sacred_label = Text("sacred", color=LAVENDER).scale(0.25)
        sacred_label.next_to(arrow_left, LEFT, buff=0.1)

        self.play(Create(arrow_left), FadeIn(sacred_label), run_time=2)
        self.wait(1.5)

        # Same arrow from man → ?
        arrow_right = Arrow(
            dots["man"].get_center(), dots["?"].get_center(),
            color=LAVENDER, stroke_width=3, buff=0.2
        )
        sacred_label2 = Text("sacred", color=LAVENDER).scale(0.25)
        sacred_label2.next_to(arrow_right, RIGHT, buff=0.1)

        self.play(Create(arrow_right), FadeIn(sacred_label2), run_time=2)
        self.wait(1)

        # ? resolves to "god" with a snap
        god_label = Text("god", color=GOLD).scale(0.5).next_to(dots["?"], DOWN, buff=0.15)
        self.play(
            dots["?"].animate.set_color(GOLD),
            Transform(labels["?"], god_label),
            Flash(dots["?"].get_center(), color=GOLD, line_length=0.3, num_lines=8),
            run_time=1.5
        )
        self.wait(1)

        # Show parallel lines connecting the analogy
        parallel_top = DashedLine(
            dots["temple"].get_center(), dots["?"].get_center(),
            color=MUTED, stroke_width=1
        ).set_opacity(0.4)
        parallel_bot = DashedLine(
            dots["house"].get_center(), dots["man"].get_center(),
            color=MUTED, stroke_width=1
        ).set_opacity(0.4)
        self.play(Create(parallel_top), Create(parallel_bot), run_time=1)
        self.wait(0.5)

        # Punchline
        punchline = body_text(
            "Vector arithmetic across 4,000 years.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)
```

- [ ] **Step 2: Test render D5**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D5_Temple`
Expected: Renders ~30s, parallelogram with "?" resolving to "god"

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add D5 Temple analogy discovery scene"
```

---

### Task 10: D6 — Mother Is Royalty, Not Earth

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write D6_Mother scene**

```python
class D6_Mother(Scene):
    def construct(self):
        glyph = hiero_text("\U000130AD\U00013300\U00013000", color=GOLD, scale=0.4)
        title = Text("Mother Is Royalty, Not Earth", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # "mother" + "earth" query
        dot_mother = Dot(point=[-2, 1, 0], radius=0.1, color=LAVENDER)
        dot_earth = Dot(point=[2, 1, 0], radius=0.1, color=TEAL)
        lbl_mother = Text("mother", color=LAVENDER).scale(0.35).next_to(dot_mother, UP, buff=0.1)
        lbl_earth = Text("earth", color=TEAL).scale(0.35).next_to(dot_earth, UP, buff=0.1)

        self.play(FadeIn(dot_mother, lbl_mother), FadeIn(dot_earth, lbl_earth), run_time=1.5)
        self.wait(0.5)

        # Expected results (ghosted)
        expected = ["soil", "fertility", "land", "harvest"]
        expected_group = VGroup()
        for i, w in enumerate(expected):
            lbl = Text(w, color=MUTED).scale(0.3).set_opacity(0.3)
            lbl.move_to([i * 1.2 - 1.8, -0.8, 0])
            expected_group.add(lbl)

        expect_header = Text("expected:", color=MUTED).scale(0.25).set_opacity(0.4)
        expect_header.move_to(LEFT * 4 + DOWN * 0.8)

        self.play(FadeIn(expected_group), FadeIn(expect_header), run_time=1.5)
        self.wait(1.5)

        # Expected fades out further, actual lights up
        actual = ["royal wife", "king's daughter", "queen", "princess"]
        actual_group = VGroup()
        for i, w in enumerate(actual):
            lbl = Text(w, color=GOLD).scale(0.35)
            lbl.move_to([i * 1.5 - 2.2, -2.0, 0])
            actual_group.add(lbl)

        actual_header = Text("actual:", color=GOLD).scale(0.25)
        actual_header.move_to(LEFT * 4 + DOWN * 2.0)

        self.play(
            expected_group.animate.set_opacity(0.1),
            expect_header.animate.set_opacity(0.15),
            FadeIn(actual_group), FadeIn(actual_header),
            run_time=2.5
        )
        self.wait(2)

        # Punchline
        punchline = body_text(
            "Motherhood is a crown, not the earth.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)
```

- [ ] **Step 2: Test render D6**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D6_Mother`
Expected: Renders ~25s, expected cluster fades as royal cluster lights up

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add D6 Mother Is Royalty discovery scene"
```

---

### Task 11: D7 — Truth and Power Are the Same Force

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write D7_Truth scene**

```python
class D7_Truth(Scene):
    def construct(self):
        glyph = hiero_text("\U00013080\U000131B4\U000132BD", color=GOLD, scale=0.4)
        title = Text("Truth and Power Are the Same Force", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # Constellation: truth, power, authority, enemies as stars
        stars = [
            ("truth", -1.5, 1.0, TEAL),
            ("power", 1.5, 1.0, LAVENDER),
            ("authority", 0, -0.5, GOLD),
            ("enemies", 1.8, -1.5, SOFT_RED),
        ]

        star_dots = {}
        star_labels = {}
        star_group = VGroup()
        for word, x, y, color in stars:
            dot = Dot(point=[x, y, 0], radius=0.1, color=color).set_opacity(0.8)
            label = Text(word, color=color).scale(0.3).next_to(dot, DOWN, buff=0.1)
            star_dots[word] = dot
            star_labels[word] = label
            star_group.add(dot, label)

        self.play(
            LaggedStart(*[FadeIn(m) for m in star_group], lag_ratio=0.1),
            run_time=3
        )
        self.wait(1)

        # Lines connect them into a tight cluster
        connections = [
            ("truth", "power"), ("truth", "authority"),
            ("power", "authority"), ("power", "enemies"),
            ("authority", "enemies"),
        ]
        conn_lines = VGroup()
        for a, b in connections:
            line = Line(
                star_dots[a].get_center(), star_dots[b].get_center(),
                color=MUTED, stroke_width=1
            ).set_opacity(0.3)
            conn_lines.add(line)

        self.play(Create(conn_lines), run_time=2)
        self.wait(0.5)

        # Cluster pulses together
        cluster_center = np.mean([star_dots[w].get_center() for w in star_dots], axis=0)

        # Contract toward center slightly
        self.play(
            *[star_dots[w].animate.move_to(
                star_dots[w].get_center() * 0.7 + cluster_center * 0.3
            ) for w in star_dots],
            run_time=2, rate_func=there_and_back_with_pause
        )
        self.wait(0.5)

        # māʿat feather at center
        maat_label = Text("māʿat", color=GOLD).scale(0.45)
        maat_sub = Text("cosmic order", color=MUTED).scale(0.25)
        maat = VGroup(maat_label, maat_sub).arrange(DOWN, buff=0.05).move_to(cluster_center)
        self.play(FadeIn(maat, scale=0.8), run_time=1.5)
        self.wait(1.5)

        # Punchline
        punchline = body_text(
            "Truth is not correctness. It is force.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)
```

- [ ] **Step 2: Test render D7**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D7_Truth`
Expected: Renders ~25s, constellation contracts with māʿat at center

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add D7 Truth Is Power discovery scene"
```

---

### Task 12: D8 — Love and Fear Meet at Eternity

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write D8_Eternity scene**

```python
class D8_Eternity(Scene):
    def construct(self):
        glyph = hiero_text("\U000131A3\U000131EF\U000131B4", color=GOLD, scale=0.4)
        title = Text("Love and Fear Meet at Eternity", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # Love and fear as two poles
        dot_love = Dot(point=[-3.5, 0, 0], radius=0.12, color=TEAL)
        dot_fear = Dot(point=[3.5, 0, 0], radius=0.12, color=SOFT_RED)
        lbl_love = Text("love", color=TEAL).scale(0.4).next_to(dot_love, DOWN, buff=0.15)
        lbl_fear = Text("fear", color=SOFT_RED).scale(0.4).next_to(dot_fear, DOWN, buff=0.15)

        axis_line = Line([-4, 0, 0], [4, 0, 0], color=MUTED, stroke_width=1).set_opacity(0.3)

        self.play(Create(axis_line), run_time=0.5)
        self.play(FadeIn(dot_love, lbl_love), FadeIn(dot_fear, lbl_fear), run_time=2)
        self.wait(1)

        # Midpoint appears
        dot_mid = Dot(point=[0, 0, 0], radius=0.15, color=GOLD).set_opacity(0)
        self.play(
            dot_mid.animate.set_opacity(0.9),
            run_time=2, rate_func=smooth
        )

        eternity_label = Text("r-nḥḥ", color=GOLD).scale(0.35)
        eternity_eng = Text("eternity", color=WHITE).scale(0.45)
        eternity_group = VGroup(eternity_label, eternity_eng).arrange(DOWN, buff=0.08)
        eternity_group.next_to(dot_mid, UP, buff=0.2)

        self.play(FadeIn(eternity_group, shift=DOWN * 0.1), run_time=2)
        self.wait(1)

        # Radiating rings — timelessness
        rings = VGroup()
        for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
            ring = Circle(radius=r, color=GOLD, stroke_width=1).set_opacity(0)
            ring.move_to(dot_mid.get_center())
            rings.add(ring)

        self.play(
            LaggedStart(
                *[ring.animate.set_opacity(0.15) for ring in rings],
                lag_ratio=0.3
            ),
            run_time=3
        )
        self.wait(1)

        # Fade rings outward
        self.play(
            *[ring.animate.set_opacity(0).scale(1.2) for ring in rings],
            run_time=2
        )

        # Punchline
        punchline = body_text(
            "Between love and fear: forever.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)
```

- [ ] **Step 2: Test render D8**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py D8_Eternity`
Expected: Renders ~30s, poles with radiating rings from midpoint

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add D8 Love Fear Eternity discovery scene"
```

---

## Chunk 4: Discussion, Conclusion, Compositors, Audio

### Task 13: S6 Discussion scene

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write S6_Discussion scene**

```python
class S6_Discussion(Scene):
    def construct(self):
        header = title_text("Honest caveats", color=WHITE, scale=0.7)
        header.move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        caveats = [
            ("Corpus bias", "The surviving texts are funerary and religious —\ntemples and tombs, not markets and homes."),
            ("32% accuracy", "Two-thirds of words don't find their match.\nThese are statistical tendencies, not certainties."),
            ("What it captures", "A dictionary says nṯr means 'god' and nbw means 'gold.'\nOnly the embedding space says they're the same word."),
        ]

        prev = None
        for title_str, detail_str in caveats:
            t = Text(title_str, color=GOLD).scale(0.45)
            d = Text(detail_str, color=MUTED).scale(0.3)
            group = VGroup(t, d).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

            if prev is None:
                group.move_to(UP * 1.5 + LEFT * 1)
            else:
                group.next_to(prev, DOWN, buff=0.4, aligned_edge=LEFT)

            self.play(FadeIn(group), run_time=2)
            self.wait(2)
            prev = group

        self.wait(2)
```

- [ ] **Step 2: Test render S6_Discussion**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S6_Discussion`
Expected: Renders ~30s, three text cards stacking

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add S6 Discussion scene"
```

---

### Task 14: S7 Conclusion scene

**Files:**
- Modify: `docs/heiroglyphy_video.py`

- [ ] **Step 1: Write S7_Conclusion scene**

Replace current S6_Close with S7_Conclusion:

```python
class S7_Conclusion(Scene):
    def construct(self):
        top_glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=0.25)
        top_glyphs.set_opacity(0.2).move_to(UP * 3.2)
        self.add(top_glyphs)

        final = body_text(
            "Translation gave us the words.",
            color=WHITE
        ).scale(1.3).move_to(UP * 0.5)
        final2 = body_text(
            "The vectors gave us the world between them.",
            color=LAVENDER
        ).scale(1.2).move_to(DOWN * 0.8)

        self.play(Write(final), run_time=3)
        self.wait(1)
        self.play(FadeIn(final2, shift=UP * 0.1), run_time=2.5)
        self.wait(2)

        # Repo link
        self.play(FadeOut(final), FadeOut(final2), run_time=1)
        repo = body_text("github.com/ebrinz/heiroglyphy", color=GOLD).scale(1.1)
        self.play(FadeIn(repo, shift=UP * 0.1), run_time=2)
        self.wait(4)
```

- [ ] **Step 2: Test render S7**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py S7_Conclusion`
Expected: Renders ~15s

- [ ] **Step 3: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add S7 Conclusion scene"
```

---

### Task 15: Update compositor classes

**Files:**
- Modify: `docs/heiroglyphy_video.py` (HeiroglyphyVideo class + new HeiroglyphyVideo3Min)

- [ ] **Step 1: Replace HeiroglyphyVideo and add HeiroglyphyVideo3Min**

Replace the existing HeiroglyphyVideo class and add the 3-minute version:

```python
class HeiroglyphyVideo(Scene):
    """
    Full version (~6:20).
    Run: manim -pqh heiroglyphy_video.py HeiroglyphyVideo
    """
    def construct(self):
        for SceneClass in [
            S1_Hook,
            S2_Idea,
            S3_Alignment,
            S4_Journey,
            D1_Gold,
            D2_Silence,
            D3_Seeing,
            D4_Snake,
            D5_Temple,
            D6_Mother,
            D7_Truth,
            D8_Eternity,
            S6_Discussion,
            S7_Conclusion,
        ]:
            SceneClass.construct(self)
            self.clear()
            self.wait(0.5)


class HeiroglyphyVideo3Min(Scene):
    """
    3-minute cut for encore.pillar.vc application (~2:53).
    Run: manim -pqh heiroglyphy_video.py HeiroglyphyVideo3Min
    """
    def construct(self):
        for SceneClass in [
            S1_Hook,
            S2_Idea,
            S3_Alignment,
            S4_Journey,
            D1_Gold,
            D2_Silence,
            D5_Temple,
            S7_Conclusion,
        ]:
            SceneClass.construct(self)
            self.clear()
            self.wait(0.5)
```

- [ ] **Step 2: Remove old S5_Discoveries and S6_Close classes**

Delete the S5_Discoveries and S6_Close classes entirely — they're replaced by D1–D8, S6_Discussion, and S7_Conclusion.

- [ ] **Step 3: Test render 3-minute cut (low quality for speed)**

Run: `cd /Users/crashy/Development/heiroglyphy/docs && manim -pql heiroglyphy_video.py HeiroglyphyVideo3Min`
Expected: Renders without error, plays through all 8 scenes

- [ ] **Step 4: Commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: add full and 3-minute compositor classes, remove old S5/S6"
```

---

### Task 16: Write audio_timing.json for full version

**Files:**
- Rewrite: `docs/audio/audio_timing.json`

- [ ] **Step 1: Write updated audio_timing.json**

Write the full narration timing manifest. Timings are estimates — the TTS pipeline handles overlap detection automatically.

```json
{
  "total_duration": null,
  "tts_model": "tts-1-hd",
  "tts_voice": "echo",
  "scenes": [
    {
      "id": "S1_Hook",
      "start": 0.0,
      "end": 25.0,
      "narration_segments": [
        {"id": "s1_01", "text": "These symbols are four thousand years old.", "start": 0.0},
        {"id": "s1_02", "text": "Scholars have been translating them for two centuries.", "start": 5.5},
        {"id": "s1_03", "text": "But translation is lossy. When you compress a word into a single English equivalent, the web of meaning around it disappears.", "start": 11.0},
        {"id": "s1_04", "text": "What if we could get it back?", "start": 19.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 0.0}
      ]
    },
    {
      "id": "S2_Idea",
      "start": 25.5,
      "end": 46.0,
      "narration_segments": [
        {"id": "s2_01", "text": "Here's the key insight. If you train a computer on enough text, every word ends up as a point in space.", "start": 25.5},
        {"id": "s2_02", "text": "And words with similar meanings cluster together. Water, river, flood, they're neighbors. King, throne, crown, another cluster.", "start": 32.5},
        {"id": "s2_03", "text": "This works for every language. Including Ancient Egyptian.", "start": 42.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 25.5}
      ]
    },
    {
      "id": "S3_Alignment",
      "start": 46.5,
      "end": 83.0,
      "narration_segments": [
        {"id": "s3_01", "text": "So we trained embeddings on a hundred thousand Ancient Egyptian sentences. And we trained them separately on English.", "start": 46.5},
        {"id": "s3_02", "text": "Both languages form a shape, a cloud of points where similar words cluster. The shapes are similar, but rotated.", "start": 55.0},
        {"id": "s3_03", "text": "The challenge is to find that rotation.", "start": 63.0},
        {"id": "s3_04", "text": "If you get it right, Egyptian words land next to their English meanings.", "start": 70.0},
        {"id": "s3_05", "text": "We got it right thirty-two percent of the time, without ever using a dictionary.", "start": 77.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 46.5}
      ]
    },
    {
      "id": "S4_Journey",
      "start": 83.5,
      "end": 101.0,
      "narration_segments": [
        {"id": "s4_01", "text": "It took fifteen attempts to get there. We tried neural networks, they failed.", "start": 83.5},
        {"id": "s4_02", "text": "We tried the latest language models, they failed spectacularly.", "start": 89.5},
        {"id": "s4_03", "text": "In the end, simple linear algebra outperformed everything. Sometimes the best tool is the oldest one.", "start": 95.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 83.5}
      ]
    },
    {
      "id": "D1_Gold",
      "start": 101.5,
      "end": 131.0,
      "narration_segments": [
        {"id": "d1_01", "text": "What happens when you find the midpoint of gold and divine in English, then look for the nearest Egyptian words?", "start": 103.0},
        {"id": "d1_02", "text": "You find n-t-r-i, meaning divine, and n-b-w, meaning gold. They're in the same place.", "start": 112.0},
        {"id": "d1_03", "text": "Modern readers call the phrase, the flesh of the gods is gold, a metaphor. The embedding space says it's not. Gold and divinity aren't compared in the texts. They're the same concept. Not metaphor. Ontology.", "start": 119.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 101.5},
        {"type": "discovery_reveal", "time": 103.0}
      ]
    },
    {
      "id": "D2_Silence",
      "start": 131.5,
      "end": 161.0,
      "narration_segments": [
        {"id": "d2_01", "text": "Find the midpoint of silence and death. Every single nearest neighbor is a variant of the word to die.", "start": 133.0},
        {"id": "d2_02", "text": "There is no Egyptian word between silence and death. They are the same point in space.", "start": 141.0},
        {"id": "d2_03", "text": "The Egyptians called the necropolis the silent land. The dead were the silent ones. What the dead lost was not life. It was voice.", "start": 148.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 131.5},
        {"type": "discovery_reveal", "time": 133.0}
      ]
    },
    {
      "id": "D3_Seeing",
      "start": 161.5,
      "end": 186.0,
      "narration_segments": [
        {"id": "d3_01", "text": "The midpoint of eye and knowledge finds the Egyptian word for eyes at the top, and heka, magic, as the third result.", "start": 163.0},
        {"id": "d3_02", "text": "The Eye of Horus was an organ, an amulet, and a unit of measurement. Seeing was not passive observation. It was an act of power.", "start": 173.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 161.5},
        {"type": "discovery_reveal", "time": 163.0}
      ]
    },
    {
      "id": "D4_Snake",
      "start": 186.5,
      "end": 211.0,
      "narration_segments": [
        {"id": "d4_01", "text": "Find the midpoint of snake and wisdom. In the Greek tradition, you'd expect knowledge and cunning.", "start": 188.0},
        {"id": "d4_02", "text": "In the Egyptian space, every result is a variant of god. The uraeus cobra was divine power, not wisdom. Two cultures, separated by geometry.", "start": 196.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 186.5},
        {"type": "discovery_reveal", "time": 188.0}
      ]
    },
    {
      "id": "D5_Temple",
      "start": 211.5,
      "end": 241.0,
      "narration_segments": [
        {"id": "d5_01", "text": "Take the vector from house to temple, and apply it to man. The result? God.", "start": 213.0},
        {"id": "d5_02", "text": "A temple is a god's house, in the same geometric sense that a house is a man's dwelling.", "start": 221.0},
        {"id": "d5_03", "text": "This is the king minus man plus woman equals queen trick, working across a four-thousand-year language boundary.", "start": 229.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 211.5},
        {"type": "discovery_reveal", "time": 213.0}
      ]
    },
    {
      "id": "D6_Mother",
      "start": 241.5,
      "end": 266.0,
      "narration_segments": [
        {"id": "d6_01", "text": "The midpoint of mother and earth. You'd expect soil, land, fertility. The earth-mother archetype.", "start": 243.0},
        {"id": "d6_02", "text": "Instead, every result is royal. King's wife. King's daughter. The Egyptian mother-goddess is not earthy. She is regal. Motherhood is a crown, not the earth.", "start": 251.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 241.5},
        {"type": "discovery_reveal", "time": 243.0}
      ]
    },
    {
      "id": "D7_Truth",
      "start": 266.5,
      "end": 291.0,
      "narration_segments": [
        {"id": "d7_01", "text": "The midpoint of truth and power finds authority at number one and enemies at number four. They all occupy the same region.", "start": 268.0},
        {"id": "d7_02", "text": "This is maat. Usually translated as truth or justice, but really it's cosmic order. The active force that holds the universe together. Truth is not correctness. It is force.", "start": 277.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 266.5},
        {"type": "discovery_reveal", "time": 268.0}
      ]
    },
    {
      "id": "D8_Eternity",
      "start": 291.5,
      "end": 321.0,
      "narration_segments": [
        {"id": "d8_01", "text": "The midpoint of love and fear. What do you find? Eternity.", "start": 293.0},
        {"id": "d8_02", "text": "Between love and fear, the Egyptians placed forever. The gods' love is awe-inspiring. Their wrath is terrifying. Both extend beyond time.", "start": 300.0},
        {"id": "d8_03", "text": "The offering formula asks that the dead be loved and feared for eternity. These aren't opposites. They're the same prayer.", "start": 310.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 291.5},
        {"type": "discovery_reveal", "time": 293.0}
      ]
    },
    {
      "id": "S6_Discussion",
      "start": 321.5,
      "end": 351.0,
      "narration_segments": [
        {"id": "s6_01", "text": "A few honest caveats. The surviving Egyptian texts are mostly funerary and religious. This is the language of temples and tombs, not markets and homes.", "start": 323.0},
        {"id": "s6_02", "text": "And thirty-two percent accuracy means two-thirds of words don't find their match. These are statistical tendencies, not certainties.", "start": 333.0},
        {"id": "s6_03", "text": "But what this approach captures is something translation destroys: the distances between words. A dictionary tells you the words. Only the embedding space tells you they're the same word.", "start": 341.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 321.5}
      ]
    },
    {
      "id": "S7_Conclusion",
      "start": 351.5,
      "end": 370.0,
      "narration_segments": [
        {"id": "s7_01", "text": "Translation gave us the words.", "start": 353.0},
        {"id": "s7_02", "text": "The vectors gave us the world between them.", "start": 357.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 351.5}
      ]
    }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add docs/audio/audio_timing.json
git commit -m "feat: update audio timing for full video with 8 discoveries"
```

---

### Task 17: Create audio_timing_3min.json

**Files:**
- Create: `docs/audio/audio_timing_3min.json`

- [ ] **Step 1: Write 3-minute audio timing**

This uses the same segment IDs as the full version (so cached TTS files are reused), but only includes scenes in the 3-minute cut with compressed timings.

```json
{
  "total_duration": null,
  "tts_model": "tts-1-hd",
  "tts_voice": "echo",
  "scenes": [
    {
      "id": "S1_Hook",
      "start": 0.0,
      "end": 25.0,
      "narration_segments": [
        {"id": "s1_01", "text": "These symbols are four thousand years old.", "start": 0.0},
        {"id": "s1_02", "text": "Scholars have been translating them for two centuries.", "start": 5.5},
        {"id": "s1_03", "text": "But translation is lossy. When you compress a word into a single English equivalent, the web of meaning around it disappears.", "start": 11.0},
        {"id": "s1_04", "text": "What if we could get it back?", "start": 19.0}
      ],
      "cues": [{"type": "scene_start", "time": 0.0}]
    },
    {
      "id": "S2_Idea",
      "start": 25.5,
      "end": 46.0,
      "narration_segments": [
        {"id": "s2_01", "text": "Here's the key insight. If you train a computer on enough text, every word ends up as a point in space.", "start": 25.5},
        {"id": "s2_02", "text": "And words with similar meanings cluster together. Water, river, flood, they're neighbors. King, throne, crown, another cluster.", "start": 32.5},
        {"id": "s2_03", "text": "This works for every language. Including Ancient Egyptian.", "start": 42.0}
      ],
      "cues": [{"type": "scene_start", "time": 25.5}]
    },
    {
      "id": "S3_Alignment",
      "start": 46.5,
      "end": 80.0,
      "narration_segments": [
        {"id": "s3_01", "text": "So we trained embeddings on a hundred thousand Ancient Egyptian sentences. And we trained them separately on English.", "start": 46.5},
        {"id": "s3_02", "text": "Both languages form a shape, a cloud of points where similar words cluster. The shapes are similar, but rotated.", "start": 55.0},
        {"id": "s3_03", "text": "The challenge is to find that rotation.", "start": 63.0},
        {"id": "s3_04", "text": "If you get it right, Egyptian words land next to their English meanings.", "start": 68.0},
        {"id": "s3_05", "text": "We got it right thirty-two percent of the time, without ever using a dictionary.", "start": 74.0}
      ],
      "cues": [{"type": "scene_start", "time": 46.5}]
    },
    {
      "id": "S4_Journey",
      "start": 80.5,
      "end": 97.0,
      "narration_segments": [
        {"id": "s4_01", "text": "It took fifteen attempts to get there. We tried neural networks, they failed.", "start": 80.5},
        {"id": "s4_02", "text": "We tried the latest language models, they failed spectacularly.", "start": 86.5},
        {"id": "s4_03", "text": "In the end, simple linear algebra outperformed everything. Sometimes the best tool is the oldest one.", "start": 92.0}
      ],
      "cues": [{"type": "scene_start", "time": 80.5}]
    },
    {
      "id": "D1_Gold",
      "start": 97.5,
      "end": 122.0,
      "narration_segments": [
        {"id": "d1_01", "text": "What happens when you find the midpoint of gold and divine in English, then look for the nearest Egyptian words?", "start": 99.0},
        {"id": "d1_02", "text": "You find n-t-r-i, meaning divine, and n-b-w, meaning gold. They're in the same place.", "start": 107.0},
        {"id": "d1_03", "text": "Modern readers call the phrase, the flesh of the gods is gold, a metaphor. The embedding space says it's not. Gold and divinity aren't compared in the texts. They're the same concept. Not metaphor. Ontology.", "start": 113.0}
      ],
      "cues": [{"type": "scene_start", "time": 97.5}]
    },
    {
      "id": "D2_Silence",
      "start": 122.5,
      "end": 147.0,
      "narration_segments": [
        {"id": "d2_01", "text": "Find the midpoint of silence and death. Every single nearest neighbor is a variant of the word to die.", "start": 124.0},
        {"id": "d2_02", "text": "There is no Egyptian word between silence and death. They are the same point in space.", "start": 131.0},
        {"id": "d2_03", "text": "The Egyptians called the necropolis the silent land. The dead were the silent ones. What the dead lost was not life. It was voice.", "start": 137.0}
      ],
      "cues": [{"type": "scene_start", "time": 122.5}]
    },
    {
      "id": "D5_Temple",
      "start": 147.5,
      "end": 168.0,
      "narration_segments": [
        {"id": "d5_01", "text": "Take the vector from house to temple, and apply it to man. The result? God.", "start": 149.0},
        {"id": "d5_02", "text": "A temple is a god's house, in the same geometric sense that a house is a man's dwelling.", "start": 155.0},
        {"id": "d5_03", "text": "This is the king minus man plus woman equals queen trick, working across a four-thousand-year language boundary.", "start": 161.0}
      ],
      "cues": [{"type": "scene_start", "time": 147.5}]
    },
    {
      "id": "S7_Conclusion",
      "start": 168.5,
      "end": 180.0,
      "narration_segments": [
        {"id": "s7_01", "text": "Translation gave us the words.", "start": 170.0},
        {"id": "s7_02", "text": "The vectors gave us the world between them.", "start": 174.0}
      ],
      "cues": [{"type": "scene_start", "time": 168.5}]
    }
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add docs/audio/audio_timing_3min.json
git commit -m "feat: add 3-minute cut audio timing manifest"
```

---

### Task 18: Delete old cached voice files and regenerate TTS

**Files:**
- Delete: `docs/audio/voice/voice_s5_*.wav` (old S5 discovery segments)
- Keep: `docs/audio/voice/voice_s1_*.wav` through `voice_s4_*.wav` (unchanged text = reusable)

- [ ] **Step 1: Clear old S5 voice cache**

```bash
rm -f docs/audio/voice/voice_s5_*.wav
rm -f docs/audio/voice_full.wav
rm -f docs/audio/mixed_audio.wav
```

- [ ] **Step 2: Generate voice for full version**

```bash
cd /Users/crashy/Development/heiroglyphy/docs/audio && python generate_voice.py
```

Expected: Generates new segments d1_01 through d8_03, s6_01–s6_03, s7_01–s7_02. Reuses cached s1–s4 segments.

- [ ] **Step 3: Generate drone and mix**

```bash
cd /Users/crashy/Development/heiroglyphy/docs/audio && python generate_drone.py && python mix_audio.py
```

- [ ] **Step 4: Commit timing files (voice files are gitignored)**

```bash
git add docs/audio/audio_timing.json docs/audio/audio_timing_3min.json
git commit -m "feat: regenerate audio for full video"
```

---

### Task 19: Render and verify full video

- [ ] **Step 1: Render full version (high quality)**

```bash
cd /Users/crashy/Development/heiroglyphy/docs && manim -qh heiroglyphy_video.py HeiroglyphyVideo
```

This will take several minutes. Expected output: `docs/media/videos/heiroglyphy_video/1080p60/HeiroglyphyVideo.mp4`

- [ ] **Step 2: Merge audio**

```bash
cd /Users/crashy/Development/heiroglyphy/docs/audio && python mix_audio.py
```

Expected: `docs/media/HeiroglyphyVideo_final.mp4`

- [ ] **Step 3: Watch and verify**

Open `docs/media/HeiroglyphyVideo_final.mp4` and check:
- All scenes play in order
- Real data visible in S3
- Each discovery has its bespoke visualization
- Narration syncs with visuals
- Total duration ~6:00–6:30

- [ ] **Step 4: Render 3-minute cut**

```bash
cd /Users/crashy/Development/heiroglyphy/docs && manim -qh heiroglyphy_video.py HeiroglyphyVideo3Min
```

Then generate audio for 3-minute cut (will need to point generate_voice.py at audio_timing_3min.json — pass as argument or temporarily swap).

- [ ] **Step 5: Final commit**

```bash
git add docs/heiroglyphy_video.py
git commit -m "feat: complete video redesign — full version + 3-minute cut"
```
