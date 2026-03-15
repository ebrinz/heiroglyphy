"""
Heiroglyphy: 3-Minute Explainer Video
Style: 3Blue1Brown-inspired (Manim Community Edition)
Target Duration: ~3:15

For a lay audience. Each scene includes a NARRATION block that
can be read aloud at ~130 words/minute (comfortable pace).
Animations are timed to match the spoken narration.

Prerequisites:
    pip install manim
    python docs/generate_viz_data.py

Run:
    cd docs
    manim -pqh heiroglyphy_video.py HeiroglyphyVideo

    # Preview single scene:
    manim -pql heiroglyphy_video.py S3_Alignment
"""

from manim import *
import numpy as np
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
FONT_PATH = str(REPO_DIR / "final_output" / "EgyptianHiero.ttf")
VIZ_DATA = DOCS_DIR / "viz_data.json"

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#0f0f1a"
GOLD     = "#f5c518"
TEAL     = "#3ec9a7"
LAVENDER = "#c4b5fd"
MUTED    = "#888899"
WHITE    = "#f0f0f0"
SOFT_RED = "#e74c3c"

config.background_color = BG

# ── Register hieroglyphic font ─────────────────────────────────────────────────
import manimpango
manimpango.register_font(FONT_PATH)
HIERO_FONT = "EgyptianHiero"

# ── Hieroglyphic characters ───────────────────────────────────────────────────
HIERO = {
    "eye":    "\U00013080",
    "water":  "\U00013217",
    "house":  "\U00013250",
    "scarab": "\U000131A3",
    "sky":    "\U000131EF",
    "lion":   "\U000130AD",
    "snake":  "\U00013196",
    "man":    "\U00013000",
    "djed":   "\U000132BD",
    "cloth":  "\U00013374",
    "bread":  "\U00013300",
    "mouth":  "\U0001337F",
    "hand":   "\U0001339B",
}

GLYPH_STRIP = " ".join([
    HIERO["eye"], HIERO["lion"], HIERO["scarab"], HIERO["sky"],
    HIERO["house"], HIERO["djed"], HIERO["cloth"], HIERO["bread"],
    HIERO["mouth"], HIERO["hand"],
])

# ── Helpers ────────────────────────────────────────────────────────────────────

def title_text(s, color=WHITE, scale=0.9):
    return Text(s, color=color).scale(scale)

def body_text(s, color=MUTED, scale=0.55):
    return Text(s, color=color).scale(scale)

def hiero_text(s, color=GOLD, scale=0.9):
    return Text(s, font=HIERO_FONT, color=color).scale(scale)

def make_cloud(n, spread, color, seed, offset=(0, 0)):
    """Representative word-embedding cloud with 3 sub-clusters."""
    rng = np.random.default_rng(seed)
    centers = [(-0.5, 0.4), (0.3, -0.3), (0.1, 0.5)]
    dots = VGroup()
    for _ in range(n):
        cx, cy = centers[rng.integers(0, len(centers))]
        x = cx + rng.normal(0, 0.35) * spread
        y = cy + rng.normal(0, 0.35) * spread
        dots.add(
            Dot(point=[x + offset[0], y + offset[1], 0],
                radius=0.035, color=color).set_opacity(rng.uniform(0.4, 0.85))
        )
    return dots


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



# ══════════════════════════════════════════════════════════════════════════════
# S1 — Hook  (~30s)
#
# NARRATION (38 words, ~18s speaking + 12s visual breathing):
#   "These symbols are four thousand years old. Scholars have been
#    translating them for two centuries. But translation is lossy.
#    When you compress a word into a single English equivalent, the
#    web of meaning around it disappears. What if we could get it back?"
#
# ══════════════════════════════════════════════════════════════════════════════
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

        # "But translation is lossy. When you compress a word into a single
        #  English equivalent, the web of meaning around it disappears."
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


# ══════════════════════════════════════════════════════════════════════════════
# S2 — The Idea  (~35s)
#
# NARRATION (52 words, ~24s speaking + 11s visual):
#   "Here's the key insight. If you train a computer on enough text,
#    every word ends up as a point in space. And words with similar
#    meanings cluster together. 'Water,' 'river,' 'flood' — they're
#    neighbors. 'King,' 'throne,' 'crown' — another cluster.
#    This works for every language. Including Ancient Egyptian."
#
# ══════════════════════════════════════════════════════════════════════════════
class S2_Idea(Scene):
    def construct(self):
        # "Here's the key insight."
        header = title_text("Words live in space", color=WHITE, scale=0.75)
        header.move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)
        self.wait(1)

        # "If you train a computer on enough text, every word ends up
        #  as a point in space. And words with similar meanings cluster together."
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

        # "'King,' 'throne,' 'crown' — another cluster."
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

        # "This works for every language. Including Ancient Egyptian."
        explain2 = body_text(
            "This works for every language — including Ancient Egyptian.",
            color=LAVENDER
        ).scale(1.0).move_to(DOWN * 2.8)
        self.play(FadeIn(explain2, shift=UP * 0.1), run_time=2)
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# S3 — The Alignment  (~45s)
#
# NARRATION (62 words, ~29s speaking + 16s visual):
#   "So we trained embeddings on a hundred thousand Ancient Egyptian
#    sentences. And we trained them separately on English. Both languages
#    form a shape — a cloud of points where similar words cluster. The
#    shapes are similar, but rotated. The challenge is to find that
#    rotation. If you get it right, Egyptian words land next to their
#    English meanings. We got it right thirty-two percent of the time —
#    without ever using a dictionary."
#
# ══════════════════════════════════════════════════════════════════════════════
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

        # Phase 3: Clouds merge
        self.play(FadeOut(explain), run_time=0.5)
        finding = body_text("Find the rotation...", color=LAVENDER).scale(1.1)
        finding.move_to(DOWN * 3.0)
        self.play(FadeIn(finding), run_time=1)

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

        anchor_lines = VGroup()
        rng = np.random.default_rng(42)
        anchor_indices = rng.choice(len(eg_cloud), size=min(10, len(eg_cloud)), replace=False)
        for idx in anchor_indices:
            eg_dot = eg_cloud[int(idx)]
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

        # Phase 5: Golden hits
        self.play(FadeOut(anchor_lines), FadeOut(overlap), run_time=0.8)

        highlights = data.get("highlights", [])
        highlight_lines = VGroup()
        highlight_labels = VGroup()

        for hl in highlights:
            best_eg = eg_cloud[0]
            best_en = en_cloud[0]
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


# ══════════════════════════════════════════════════════════════════════════════
# S4 — The Journey  (~25s)
#
# NARRATION (40 words, ~18s speaking + 7s visual):
#   "It took fifteen attempts to get there. We tried neural networks —
#    they failed. We tried the latest language models — they failed
#    spectacularly. In the end, simple linear algebra outperformed
#    everything. Sometimes the best tool is the oldest one."
#
# ══════════════════════════════════════════════════════════════════════════════
class S4_Journey(Scene):
    def construct(self):
        header = title_text("15 attempts to get there", color=WHITE, scale=0.7)
        header.move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)
        self.wait(0.5)

        bars_data = [
            ("V3", 22.0, TEAL),
            ("V5", 24.5, TEAL),
            ("V6\nBERT", 0.5, SOFT_RED),
            ("V7", 29.1, TEAL),
            ("V9", 30.5, TEAL),
            ("V13", 31.6, TEAL),
            ("V15", 32.4, GOLD),
        ]

        baseline = Line(LEFT * 3.5, RIGHT * 3.5, color=MUTED, stroke_width=0.5)
        baseline.move_to(DOWN * 1.5)
        self.play(Create(baseline), run_time=0.3)

        bar_groups = VGroup()
        for i, (lbl, val, col) in enumerate(bars_data):
            h = max(val / 32.4 * 3.0, 0.06)
            bar = Rectangle(width=0.65, height=h, fill_color=col,
                            fill_opacity=0.85, stroke_width=0)
            x = i * 0.95 - 2.85
            bar.move_to(RIGHT * x + DOWN * 1.5 + UP * h / 2)
            bl = Text(lbl, color=MUTED).scale(0.22).next_to(bar, DOWN, buff=0.08)
            pct = Text(f"{val:.0f}%", color=col).scale(0.22).next_to(bar, UP, buff=0.06)
            bar_groups.add(VGroup(bar, bl, pct))

        # "We tried neural networks — they failed."
        for i, bg in enumerate(bar_groups):
            self.play(GrowFromEdge(bg, DOWN), run_time=1.0)
            if i == 2:
                # "We tried the latest language models — they failed spectacularly."
                self.wait(1.0)
            else:
                self.wait(0.5)

        self.wait(0.5)

        # "In the end, simple linear algebra outperformed everything."
        lesson = body_text(
            "Simple linear algebra beat every neural network we tried.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 3.2)
        self.play(FadeIn(lesson), run_time=2)
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# S5 — The Discoveries  (~55s)
#
# NARRATION (108 words, ~50s speaking + 5s transitions):
#
#   [Gold] "Here's what the geometry revealed. The midpoint of 'gold'
#    and 'divine' in English maps to the same region of the Egyptian
#    space. This isn't metaphor. The texts don't distinguish them.
#    Gold IS divinity."
#
#   [Silence] "The midpoint of 'silence' and 'death' — every single
#    result is a variant of 'to die.' The Egyptians called the
#    necropolis 'the silent land.' What the dead lost was not life.
#    It was voice."
#
#   [Eye] "The Eye of Horus sits between 'knowledge' and 'spellcasting.'
#    Seeing was not observation. It was an act of magical power."
#
#   [Snake] "In Greek tradition, the snake means wisdom. In Egyptian
#    vectors, it means the gods. Two cultures, separated by geometry."
#
#   [Closing] "Translation gave us the words. The vectors gave us the
#    world between them."
#
# ══════════════════════════════════════════════════════════════════════════════


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

        # Dots merge
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
        punchline = body_text("Not metaphor. Ontology.", color=WHITE).scale(1.2).move_to(DOWN * 3.5)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)


class D2_Silence(Scene):
    def construct(self):
        glyph = hiero_text("\U000131EF\U0001337F\U000132BD", color=GOLD, scale=0.4)
        title = Text("Silence Is the Condition of the Dead", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        # Sound wave
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

        # m(w)t variants cluster
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
        punchline = body_text("What the dead lost was not life. It was voice.", color=WHITE).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)


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

        # Three concept points
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

        # Vectors from eye to each concept
        vectors = VGroup()
        for dot in concept_dots:
            vec = Arrow(
                eye.get_center(), dot.get_center(),
                color=GOLD, stroke_width=2, buff=0.3
            ).set_opacity(0.5)
            vectors.add(vec)

        self.play(Create(vectors), run_time=2)
        self.wait(1)

        # Triangle connects concepts
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
        punchline = body_text("Sight was not observation. It was power.", color=WHITE).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)


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
        punchline = body_text("Two cultures, separated by geometry.", color=WHITE).scale(1.1).move_to(DOWN * 3.3)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(3)


# ══════════════════════════════════════════════════════════════════════════════
class S5_Discoveries(Scene):
    def construct(self):
        header = title_text("What the geometry reveals", color=GOLD, scale=0.7)
        header.move_to(UP * 3.3)
        self.play(FadeIn(header), run_time=1.5)
        self.wait(1.5)

        discoveries = [
            (
                HIERO["djed"],
                "Gold = Divinity",
                "Not metaphor. The embedding space cannot\n"
                "distinguish gold from the divine — because\n"
                "the Egyptian texts don't distinguish them.",
                10,  # hold time — fits ~10.6s narration
            ),
            (
                HIERO["sky"],
                "Silence = Death",
                "Every word between 'silence' and 'death'\n"
                "is a variant of 'to die.' What the dead\n"
                "lost was not life — it was voice.",
                10,  # hold time — fits ~11s narration
            ),
            (
                HIERO["eye"],
                "Seeing = Magic",
                "The Eye of Horus sits between 'knowledge'\n"
                "and 'spellcasting.' Sight was not passive\n"
                "observation — it was an act of power.",
                7,  # hold time — fits ~7.2s narration
            ),
            (
                HIERO["snake"],
                "The Snake Is Divine, Not Wise",
                "Greek tradition links snakes to wisdom.\n"
                "Egyptian vectors link them to the gods.\n"
                "Two cultures, separated by geometry.",
                7,  # hold time — fits ~7.3s narration
            ),
        ]

        prev_group = None
        for glyph_char, title_str, detail_str, hold in discoveries:
            glyph = hiero_text(glyph_char, color=GOLD, scale=0.6)
            title = Text(title_str, color=WHITE).scale(0.5)
            detail = Text(detail_str, color=MUTED).scale(0.3)

            row = VGroup(glyph, title).arrange(RIGHT, buff=0.3)
            group = VGroup(row, detail).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            group.move_to(ORIGIN + DOWN * 0.3)

            if prev_group:
                self.play(FadeOut(prev_group), run_time=1)

            self.play(FadeIn(group), run_time=2)
            self.wait(hold)
            prev_group = group

        # "Translation gave us the words. The vectors gave us the world
        #  between them."
        self.play(FadeOut(prev_group), FadeOut(header), run_time=1)

        final = body_text(
            "Translation gave us the words.",
            color=WHITE
        ).scale(1.3).move_to(UP * 0.5)
        final2 = body_text(
            "The vectors gave us the world between them.",
            color=LAVENDER
        ).scale(1.2).move_to(DOWN * 0.8)

        self.play(Write(final), run_time=3)
        self.wait(2)
        self.play(FadeIn(final2, shift=UP * 0.1), run_time=2.5)
        self.wait(5)


# ══════════════════════════════════════════════════════════════════════════════
# S6 — Close  (~8s)
#
# NARRATION: (none — just the repo link on screen)
#
# ══════════════════════════════════════════════════════════════════════════════
class S6_Close(Scene):
    def construct(self):
        top_glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=0.25)
        top_glyphs.set_opacity(0.2).move_to(UP * 3.2)
        self.add(top_glyphs)

        repo = body_text("github.com/ebrinz/heiroglyphy", color=GOLD).scale(1.1)
        self.play(FadeIn(repo, shift=UP * 0.1), run_time=2)
        self.wait(5)


# ══════════════════════════════════════════════════════════════════════════════
# Full Video  (~3:15)
# ══════════════════════════════════════════════════════════════════════════════
class HeiroglyphyVideo(Scene):
    """
    Run: manim -pqh heiroglyphy_video.py HeiroglyphyVideo

    Scene timing:
      S1 Hook:        ~30s
      S2 Idea:        ~35s
      S3 Alignment:   ~45s
      S4 Journey:     ~25s
      S5 Discoveries: ~55s
      S6 Close:       ~8s
      Total:          ~3:18
    """
    def construct(self):
        for SceneClass in [
            S1_Hook,
            S2_Idea,
            S3_Alignment,
            S4_Journey,
            S5_Discoveries,
            S6_Close,
        ]:
            SceneClass.construct(self)
            self.clear()
            self.wait(0.5)
