"""
Heiroglyphy: 3-Minute Explainer Video
Style: 3Blue1Brown-inspired (Manim Community Edition)
Target Duration: ~3:00

For a lay audience. Tells the story:
  Hook → What are embeddings → Two clouds align → What we found → The gems

Prerequisites:
    pip install manim
    python docs/generate_viz_data.py      # generates docs/viz_data.json

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

# ── Load viz data ─────────────────────────────────────────────────────────────
_viz = None
def get_viz_data():
    global _viz
    if _viz is None and VIZ_DATA.exists():
        with open(VIZ_DATA) as f:
            _viz = json.load(f)
    return _viz

# ── Helpers ────────────────────────────────────────────────────────────────────

def title_text(s, color=WHITE, scale=0.9):
    return Text(s, color=color).scale(scale)

def body_text(s, color=MUTED, scale=0.55):
    return Text(s, color=color).scale(scale)

def hiero_text(s, color=GOLD, scale=0.9):
    return Text(s, font=HIERO_FONT, color=color).scale(scale)

def make_cloud(n, spread, color, seed, offset=(0,0)):
    """Representative word-embedding cloud."""
    rng = np.random.default_rng(seed)
    # Clustered: 3 sub-clusters to suggest semantic neighborhoods
    dots = VGroup()
    centers = [(-0.5, 0.4), (0.3, -0.3), (0.1, 0.5)]
    for _ in range(n):
        cx, cy = centers[rng.integers(0, len(centers))]
        x = cx + rng.normal(0, 0.35) * spread
        y = cy + rng.normal(0, 0.35) * spread
        dots.add(
            Dot(point=[x + offset[0], y + offset[1], 0],
                radius=0.035, color=color).set_opacity(rng.uniform(0.4, 0.85))
        )
    return dots


# ══════════════════════════════════════════════════════════════════════════════
# S1 — Hook  (0:00 – 0:25)
#      "4,000 years of translation missed something"
# ══════════════════════════════════════════════════════════════════════════════
class S1_Hook(Scene):
    def construct(self):
        # Hieroglyphs fade in across the screen
        glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=0.7)
        glyphs.move_to(UP * 1.5)

        self.play(FadeIn(glyphs, shift=UP * 0.2), run_time=3)
        self.wait(1)

        line1 = body_text(
            "For 200 years, scholars have translated these symbols.",
            color=WHITE
        ).scale(1.1).next_to(glyphs, DOWN, buff=0.8)
        self.play(Write(line1), run_time=3)
        self.wait(1.5)

        line2 = body_text(
            "But translation compresses meaning.\n"
            "The relationships between words are lost.",
            color=MUTED
        ).scale(1.0).next_to(line1, DOWN, buff=0.5)
        self.play(Write(line2), run_time=3)
        self.wait(1)

        line3 = body_text(
            "What if we could recover them?",
            color=LAVENDER
        ).scale(1.1).next_to(line2, DOWN, buff=0.6)
        self.play(FadeIn(line3, shift=UP * 0.1), run_time=2)
        self.wait(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# S2 — The Idea  (0:25 – 0:55)
#      "Words live in space. Similar words cluster together."
# ══════════════════════════════════════════════════════════════════════════════
class S2_Idea(Scene):
    def construct(self):
        header = title_text("Words live in space", color=WHITE, scale=0.75)
        header.move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)
        self.wait(0.5)

        # Show English example: related words cluster
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

        self.play(LaggedStart(*[FadeIn(m) for m in dots_and_labels], lag_ratio=0.08), run_time=3)
        self.wait(1)

        explain = body_text(
            "Words that appear in similar contexts\n"
            "end up close together in this space.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 1.5)
        self.play(Write(explain), run_time=2.5)
        self.wait(1.5)

        explain2 = body_text(
            "This works for every language — including Ancient Egyptian.",
            color=LAVENDER
        ).scale(1.0).move_to(DOWN * 2.8)
        self.play(FadeIn(explain2, shift=UP * 0.1), run_time=2)
        self.wait(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# S3 — The Alignment  (0:55 – 1:35)
#      Two clouds rotate into each other
# ══════════════════════════════════════════════════════════════════════════════
class S3_Alignment(Scene):
    def construct(self):
        # Egyptian cloud (left, gold, rotated ~30 degrees)
        eg_cloud = make_cloud(80, 1.0, GOLD, seed=42, offset=(-3, 0))
        en_cloud = make_cloud(80, 1.0, TEAL, seed=99, offset=(3, 0))

        label_eg = VGroup(
            hiero_text(HIERO["eye"], color=GOLD, scale=0.35),
            Text("Egyptian", color=GOLD).scale(0.45),
        ).arrange(RIGHT, buff=0.15).move_to(LEFT * 3 + UP * 2.8)

        label_en = Text("English", color=TEAL).scale(0.45).move_to(RIGHT * 3 + UP * 2.8)

        self.play(FadeIn(label_eg), Create(eg_cloud), run_time=2.5)
        self.wait(0.5)
        self.play(FadeIn(label_en), Create(en_cloud), run_time=2.5)
        self.wait(1)

        # Explain
        explain = body_text(
            "Both languages form a shape.\n"
            "The shapes are similar — but rotated.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 3.0)
        self.play(Write(explain), run_time=2.5)
        self.wait(1.5)

        # Animate: clouds rotate and slide together
        self.play(FadeOut(explain), run_time=0.8)

        finding = body_text("Find the rotation...", color=LAVENDER).scale(1.1)
        finding.move_to(DOWN * 3.0)
        self.play(FadeIn(finding), run_time=1)

        self.play(
            eg_cloud.animate.shift(RIGHT * 2.2).rotate(0.3),
            en_cloud.animate.shift(LEFT * 2.2).rotate(-0.1),
            run_time=4, rate_func=smooth
        )
        self.wait(0.5)

        # Overlapping — show connection lines between nearby dots
        self.play(FadeOut(finding), run_time=0.5)
        overlap = body_text(
            "...and the words align across 4,000 years.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 3.0)
        self.play(FadeIn(overlap), run_time=1.5)

        # Flash a few connection lines
        lines = VGroup()
        rng = np.random.default_rng(7)
        for _ in range(8):
            eg_dot = eg_cloud[rng.integers(0, len(eg_cloud))]
            en_dot = en_cloud[rng.integers(0, len(en_cloud))]
            line = Line(
                eg_dot.get_center(), en_dot.get_center(),
                color=LAVENDER, stroke_width=1
            ).set_opacity(0.4)
            lines.add(line)

        self.play(Create(lines), run_time=2)
        self.wait(2)

        # Accuracy
        self.play(FadeOut(overlap), FadeOut(lines), run_time=0.8)
        acc = body_text("32.35% accuracy — no dictionary needed.",
                        color=GOLD).scale(1.1).move_to(DOWN * 3.0)
        self.play(FadeIn(acc), run_time=1.5)
        self.wait(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# S4 — The Journey  (1:35 – 1:55)
#      Quick montage of 15 attempts
# ══════════════════════════════════════════════════════════════════════════════
class S4_Journey(Scene):
    def construct(self):
        header = title_text("15 attempts to get there", color=WHITE, scale=0.7)
        header.move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)

        bars_data = [
            ("V3", 22.0, TEAL),
            ("V5", 24.5, TEAL),
            ("V6", 0.5, SOFT_RED),
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
            bl = Text(lbl, color=MUTED).scale(0.25).next_to(bar, DOWN, buff=0.08)
            pct = Text(f"{val:.0f}%", color=col).scale(0.22).next_to(bar, UP, buff=0.06)
            bar_groups.add(VGroup(bar, bl, pct))

        for i, bg in enumerate(bar_groups):
            rt = 1.0 if i != 2 else 1.2
            self.play(GrowFromEdge(bg, DOWN), run_time=rt)
            if i == 2:  # BERT failure
                note = body_text("BERT: 0.5%", color=SOFT_RED).scale(0.8)
                note.next_to(bg, UP, buff=0.3)
                self.play(FadeIn(note), run_time=0.5)
                self.wait(0.8)
                self.play(FadeOut(note), run_time=0.3)
            else:
                self.wait(0.3)

        lesson = body_text(
            "Simple linear algebra beat every neural network we tried.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 3.2)
        self.play(FadeIn(lesson), run_time=1.5)
        self.wait(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# S5 — The Discoveries  (1:55 – 2:45)
#      The gems — what the geometry reveals
# ══════════════════════════════════════════════════════════════════════════════
class S5_Discoveries(Scene):
    def construct(self):
        header = title_text("What the geometry reveals", color=GOLD, scale=0.7)
        header.move_to(UP * 3.3)
        self.play(FadeIn(header), run_time=1.5)
        self.wait(0.5)

        discoveries = [
            (
                HIERO["djed"],
                "Gold = Divinity",
                "Not metaphor. The embedding space cannot\n"
                "distinguish gold from the divine — because\n"
                "the Egyptian texts don't distinguish them.",
            ),
            (
                HIERO["sky"],
                "Silence = Death",
                "Every word between 'silence' and 'death'\n"
                "is a variant of 'to die.' What the dead\n"
                "lost was not life — it was voice.",
            ),
            (
                HIERO["eye"],
                "Seeing = Magic",
                "The Eye of Horus sits between 'knowledge'\n"
                "and 'spellcasting.' Sight was not passive\n"
                "observation — it was an act of power.",
            ),
            (
                HIERO["snake"],
                "The Snake Is Divine, Not Wise",
                "Greek tradition links snakes to wisdom.\n"
                "Egyptian vectors link them to the gods.\n"
                "Two cultures, separated by geometry.",
            ),
        ]

        prev_group = None
        for glyph_char, title_str, detail_str in discoveries:
            glyph = hiero_text(glyph_char, color=GOLD, scale=0.6)
            title = Text(title_str, color=WHITE).scale(0.5)
            detail = Text(detail_str, color=MUTED).scale(0.3)

            row = VGroup(glyph, title).arrange(RIGHT, buff=0.3)
            group = VGroup(row, detail).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            group.move_to(ORIGIN + DOWN * 0.3)

            if prev_group:
                self.play(FadeOut(prev_group), run_time=0.8)

            self.play(FadeIn(group), run_time=2)
            self.wait(3)
            prev_group = group

        # Final discovery as a full-screen statement
        self.play(FadeOut(prev_group), FadeOut(header), run_time=0.8)

        final = body_text(
            "Translation gave us the words.",
            color=WHITE
        ).scale(1.3).move_to(UP * 0.5)
        final2 = body_text(
            "The vectors gave us the world between them.",
            color=LAVENDER
        ).scale(1.2).move_to(DOWN * 0.8)

        self.play(Write(final), run_time=2.5)
        self.wait(0.5)
        self.play(FadeIn(final2, shift=UP * 0.1), run_time=2)
        self.wait(3)


# ══════════════════════════════════════════════════════════════════════════════
# S6 — Close  (2:45 – 3:00)
# ══════════════════════════════════════════════════════════════════════════════
class S6_Close(Scene):
    def construct(self):
        # Subtle hieroglyph strip
        top_glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=0.25)
        top_glyphs.set_opacity(0.2).move_to(UP * 3.2)
        self.add(top_glyphs)

        repo = body_text("github.com/ebrinz/heiroglyphy", color=GOLD).scale(1.1)

        self.play(FadeIn(repo, shift=UP * 0.1), run_time=2)
        self.wait(4)


# ══════════════════════════════════════════════════════════════════════════════
# Full Video
# ══════════════════════════════════════════════════════════════════════════════
class HeiroglyphyVideo(Scene):
    """
    Run: manim -pqh heiroglyphy_video.py HeiroglyphyVideo
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
