"""
Heiroglyphy: Encode Fellowship Application Video
Style: 3Blue1Brown-inspired (Manim Community Edition)
Target Duration: ~3:10

Prerequisites:
    pip install manim
    python docs/generate_viz_data.py      # generates docs/viz_data.json

Run:
    cd docs
    manim -pqh heiroglyphy_video.py HeiroglyphyVideo

    # Preview a single scene:
    manim -pql heiroglyphy_video.py S4_Discoveries

Scenes:
    S1  - The Question           (0:00 - 0:25)   hieroglyphs + hook
    S2  - The Motivation         (0:25 - 0:50)   why embedding space
    S3  - The Hypothesis         (0:50 - 1:25)   real embedding viz
    S4  - The Journey            (1:25 - 2:05)   bar chart, one at a time
    S5  - Discoveries            (2:05 - 2:50)   golden hits & failures
    S6  - What We Learned        (2:50 - 3:10)   methodological takeaways
    S7  - The Claim              (3:10 - 3:20)
    S8  - Close                  (3:20 - 3:30)
"""

from manim import *
import numpy as np
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
FONT_PATH = str(REPO_DIR / "final_output" / "EgyptianHiero.ttf")
VIZ_DATA  = DOCS_DIR / "viz_data.json"

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

# ── Hieroglyphic characters (Unicode Egyptian Hieroglyphs block) ───────────────
# Used throughout the video as decorative and illustrative elements
HIERO = {
    "eye":     "\U00013080",  # D10 - Eye of Horus
    "water":   "\U00013217",  # N35A - water ripple
    "house":   "\U00013250",  # O1 - house
    "scarab":  "\U000131A3",  # L1 - scarab
    "sky":     "\U000131EF",  # N1 - sky
    "lion":    "\U000130AD",  # E22 - lion
    "snake":   "\U00013196",  # I9 - horned viper
    "man":     "\U00013000",  # A1 - seated man
    "djed":    "\U000132BD",  # R11 - djed pillar
    "cloth":   "\U00013374",  # S29 - folded cloth
    "bread":   "\U00013300",  # T14
    "mouth":   "\U0001337F",  # D21 - mouth
    "hand":    "\U0001339B",  # D46 - hand
}

# A decorative strip of mixed glyphs
GLYPH_STRIP = " ".join([
    HIERO["eye"], HIERO["lion"], HIERO["scarab"], HIERO["sky"],
    HIERO["house"], HIERO["djed"], HIERO["cloth"], HIERO["bread"],
    HIERO["mouth"], HIERO["hand"],
])


# ── Load visualization data (if available) ─────────────────────────────────────
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

def make_random_cloud(n=60, spread=1.4, color=TEAL, seed=0):
    rng = np.random.default_rng(seed)
    return VGroup(*[
        Dot(point=[rng.uniform(-spread, spread),
                   rng.uniform(-spread, spread), 0],
            radius=0.04, color=color).set_opacity(0.7)
        for _ in range(n)
    ])

def hiero_divider(y=0):
    """A subtle row of small hieroglyphs as a section divider."""
    div = hiero_text(
        f"{HIERO['scarab']}  {HIERO['djed']}  {HIERO['eye']}  {HIERO['djed']}  {HIERO['scarab']}",
        color=GOLD, scale=0.3
    ).set_opacity(0.4).move_to(UP * y)
    return div


# ══════════════════════════════════════════════════════════════════════════════
# S1 — The Question  (0:00 – 0:25)
# ══════════════════════════════════════════════════════════════════════════════
class S1_Question(Scene):
    def construct(self):
        question = title_text(
            "Can meaning be recovered from a language\nno living person speaks?",
            color=WHITE, scale=0.8
        ).move_to(UP * 0.5)

        # Full hieroglyphic strip rendered with the actual Egyptian font
        glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=0.8)
        glyphs.next_to(question, DOWN, buff=0.6)

        sub = body_text("No dictionary.  No teacher.  Just structure.",
                        color=MUTED).next_to(glyphs, DOWN, buff=0.5)

        self.play(Write(question), run_time=4)
        self.wait(0.5)
        self.play(FadeIn(glyphs, shift=UP * 0.2), run_time=2.5)
        self.wait(0.5)
        self.play(FadeIn(sub), run_time=2)
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# S2 — The Motivation  (0:25 – 0:50)
#       Why we chose embedding space for ancient languages
# ══════════════════════════════════════════════════════════════════════════════
class S2_Motivation(Scene):
    def construct(self):
        # Decorative hieroglyphs at top
        top_glyphs = hiero_text(
            f"{HIERO['water']}  {HIERO['eye']}  {HIERO['sky']}",
            color=GOLD, scale=0.45
        ).set_opacity(0.3).move_to(UP * 3.3)
        self.add(top_glyphs)

        header = title_text("Why embedding space?", color=GOLD, scale=0.75)
        header.move_to(UP * 2.3)
        self.play(FadeIn(header), run_time=1.5)
        self.wait(0.5)

        line1 = body_text(
            "Ancient Egyptian has been translated by scholars for centuries —",
            color=WHITE
        ).scale(1.0).next_to(header, DOWN, buff=0.6)
        self.play(Write(line1), run_time=2.5)
        self.wait(1)

        line2 = body_text(
            "but translation is lossy. Nuance, connotation, and context\n"
            "are compressed into a single modern word.",
            color=MUTED
        ).scale(0.95).next_to(line1, DOWN, buff=0.35)
        self.play(Write(line2), run_time=3)
        self.wait(1.5)

        line3 = body_text(
            "Word embeddings preserve the full geometry of meaning —\n"
            "every relationship, every shade, every neighbor.",
            color=WHITE
        ).scale(0.95).next_to(line2, DOWN, buff=0.35)
        self.play(Write(line3), run_time=3)
        self.wait(1)

        line4 = body_text(
            "If we can align that geometry to English,\n"
            "we recover what translation left behind.",
            color=LAVENDER
        ).scale(1.0).next_to(line3, DOWN, buff=0.5)
        self.play(FadeIn(line4, shift=UP * 0.1), run_time=2)
        self.wait(3)


# ══════════════════════════════════════════════════════════════════════════════
# S3 — The Hypothesis  (0:50 – 1:25)
#       Real embedding data projected onto semantic axes
# ══════════════════════════════════════════════════════════════════════════════
class S3_Hypothesis(Scene):
    def construct(self):
        viz = get_viz_data()

        if viz is None:
            self._fallback()
            return

        axes_info = viz["axes"]
        eg_pts = viz["egyptian"]
        en_pts = viz["english"]

        # ── Egyptian cloud — LARGE, fills left half ──────────────────────
        eg_scale = 0.85
        eg_offset = np.array([-2.8, 0, 0])

        eg_dots = VGroup()
        for pt in eg_pts:
            pos = np.array([pt["x"] * eg_scale, pt["y"] * eg_scale, 0]) + eg_offset
            eg_dots.add(Dot(point=pos, radius=0.035, color=GOLD).set_opacity(0.65))

        # Label with a hieroglyph accent
        label_eg = VGroup(
            hiero_text(HIERO["eye"], color=GOLD, scale=0.4),
            Text("Ancient Egyptian", color=GOLD).scale(0.5),
        ).arrange(RIGHT, buff=0.2).move_to(eg_offset + UP * 3.2)

        self.play(FadeIn(label_eg), run_time=1.5)
        self.play(Create(eg_dots), run_time=3)
        self.wait(1.5)

        # ── Axis labels — spaced at edges ────────────────────────────────
        ax_x_neg = Text(f"← {axes_info['x_label_neg']}", color=MUTED).scale(0.35)
        ax_x_pos = Text(f"{axes_info['x_label_pos']} →", color=MUTED).scale(0.35)
        ax_x_neg.move_to(eg_offset + LEFT * 2.5 + DOWN * 2.8)
        ax_x_pos.move_to(eg_offset + RIGHT * 2.5 + DOWN * 2.8)

        ax_y_top = Text(f"{axes_info['y_label_pos']} ↑", color=MUTED).scale(0.35)
        ax_y_bot = Text(f"↓ {axes_info['y_label_neg']}", color=MUTED).scale(0.35)
        ax_y_top.move_to(eg_offset + LEFT * 3.2 + UP * 2.0)
        ax_y_bot.move_to(eg_offset + LEFT * 3.2 + DOWN * 2.0)

        self.play(
            FadeIn(ax_x_neg), FadeIn(ax_x_pos),
            FadeIn(ax_y_top), FadeIn(ax_y_bot),
            run_time=2
        )
        self.wait(1.5)

        # ── English cloud on right ───────────────────────────────────────
        en_scale = 0.85
        en_offset = np.array([3.2, 0, 0])

        en_dots = VGroup()
        for pt in en_pts:
            cat = pt.get("category", "other")
            col = viz["category_colors"].get(cat, TEAL)
            pos = np.array([pt["x"] * en_scale, pt["y"] * en_scale, 0]) + en_offset
            en_dots.add(Dot(point=pos, radius=0.035, color=col).set_opacity(0.6))

        label_en = Text("Modern English", color=TEAL).scale(0.5)
        label_en.move_to(en_offset + UP * 3.2)

        self.play(FadeIn(label_en), run_time=1)
        self.play(Create(en_dots), run_time=3)
        self.wait(1)

        # ── Hypothesis text ──────────────────────────────────────────────
        hyp = body_text(
            "Every language has a geometric shape.\n"
            "Similar words cluster together.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 3.3)
        self.play(Write(hyp), run_time=3)
        self.wait(1)

        # ── Arrow ────────────────────────────────────────────────────────
        arrow = Arrow(LEFT * 0.6, RIGHT * 0.6, color=LAVENDER, stroke_width=3)
        arrow.move_to(UP * 0.3)
        bridge_label = body_text("find the rotation", color=LAVENDER)
        bridge_label.scale(1.0).next_to(arrow, UP, buff=0.15)
        self.play(GrowArrow(arrow), FadeIn(bridge_label), run_time=2)
        self.wait(2)

    def _fallback(self):
        cloud_eg = make_random_cloud(n=80, spread=2.0, color=GOLD, seed=1).move_to(LEFT * 3)
        cloud_en = make_random_cloud(n=55, color=TEAL, seed=2).move_to(RIGHT * 3)
        label_eg = Text("Ancient Egyptian", color=GOLD).scale(0.5).move_to(LEFT * 3 + UP * 3)
        label_en = Text("Modern English", color=TEAL).scale(0.5).move_to(RIGHT * 3 + UP * 3)
        self.play(FadeIn(label_eg), Create(cloud_eg), run_time=3)
        self.wait(1)
        self.play(FadeIn(label_en), Create(cloud_en), run_time=3)
        hyp = body_text("Every language has a geometric shape.\nSimilar words cluster together.",
                        color=WHITE).move_to(DOWN * 3.0).scale(1.1)
        self.play(Write(hyp), run_time=3)
        arrow = Arrow(LEFT * 0.6, RIGHT * 0.6, color=LAVENDER, stroke_width=3)
        bridge = body_text("find the rotation", color=LAVENDER).next_to(arrow, UP, buff=0.15)
        self.play(GrowArrow(arrow), FadeIn(bridge), run_time=2)
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# S4 — The Journey  (1:25 – 2:05)
#       Clouds merge + accuracy bar chart (slow, one at a time)
# ══════════════════════════════════════════════════════════════════════════════
class S4_Journey(Scene):
    def construct(self):
        viz = get_viz_data()

        if viz is None:
            cloud_eg = make_random_cloud(n=60, spread=1.8, color=GOLD, seed=3).move_to(LEFT * 2.5 + UP * 0.3)
            cloud_en = make_random_cloud(n=50, color=TEAL, seed=4).move_to(RIGHT * 2.5 + DOWN * 0.3)
        else:
            eg_pts = viz["egyptian"]
            en_pts = viz["english"]
            scale = 0.5
            cloud_eg = VGroup(*[
                Dot(point=[pt["x"] * scale - 2.5, pt["y"] * scale + 0.3, 0],
                    radius=0.035, color=GOLD).set_opacity(0.6)
                for pt in eg_pts
            ])
            cloud_en = VGroup(*[
                Dot(point=[pt["x"] * scale + 2.5, pt["y"] * scale - 0.3, 0],
                    radius=0.035, color=TEAL).set_opacity(0.6)
                for pt in en_pts
            ])

        self.play(Create(cloud_eg), Create(cloud_en), run_time=2.5)
        self.wait(0.5)

        # ── Clouds slide together ────────────────────────────────────────
        label = body_text("12 iterations.  Neural networks failed.\nLinear algebra succeeded.",
                          color=WHITE).move_to(DOWN * 3.0).scale(1.1)
        self.play(Write(label), run_time=2.5)
        self.wait(0.5)

        self.play(
            cloud_eg.animate.shift(RIGHT * 2.0),
            cloud_en.animate.shift(LEFT * 2.0),
            run_time=3.5, rate_func=smooth
        )
        self.wait(1.5)

        # ── Transition to bar chart ──────────────────────────────────────
        self.play(FadeOut(label), FadeOut(cloud_eg), FadeOut(cloud_en), run_time=1.5)

        # ── Bar chart — slow, one at a time ──────────────────────────────
        bars_data = [
            ("V3\nProcrustes",  22.0,  TEAL),
            ("V5\n10x Data",    24.53, TEAL),
            ("V6\nBERT",        0.47,  SOFT_RED),
            ("V7\nFastText",    29.10, TEAL),
            ("V8\nCoptic",      28.16, MUTED),
            ("V9\nVisual",      30.52, TEAL),
            ("V10\nSOTA",       30.67, GOLD),
        ]

        baseline = Line(LEFT * 3.2, RIGHT * 3.2, color=MUTED, stroke_width=0.5).move_to(DOWN * 1.5)
        self.play(Create(baseline), run_time=0.5)

        bar_groups = VGroup()
        for i, (lbl, val, col) in enumerate(bars_data):
            h = max(val / 30.67 * 2.8, 0.06)
            bar = Rectangle(width=0.55, height=h, fill_color=col,
                            fill_opacity=0.85, stroke_width=0)
            x_pos = i * 0.85 - 2.55
            bar.move_to(RIGHT * x_pos + DOWN * 1.5 + UP * h / 2)
            bl = Text(lbl, color=MUTED).scale(0.2).next_to(bar, DOWN, buff=0.1)
            pct = Text(f"{val:.1f}%", color=col).scale(0.22).next_to(bar, UP, buff=0.06)
            bar_groups.add(VGroup(bar, bl, pct))

        for i, bg in enumerate(bar_groups):
            rt = 1.2 if i != 2 else 1.5
            self.play(GrowFromEdge(bg, DOWN), run_time=rt)
            if i == 2:
                self.wait(1.2)  # linger on BERT failure
            elif i == len(bar_groups) - 1:
                self.wait(0.8)  # linger on SOTA
            else:
                self.wait(0.4)

        peak = body_text("30.67% Top-1 accuracy  •  unsupervised",
                         color=GOLD).scale(1.1).move_to(UP * 2.5)
        self.play(FadeIn(peak), run_time=1.5)
        self.wait(3)


# ══════════════════════════════════════════════════════════════════════════════
# S5 — Discoveries  (2:05 – 2:50)
#       Golden hits, failures, and surprises — mined from DISCOVERIES.md
#       Uses hieroglyphic characters as visual anchors
# ══════════════════════════════════════════════════════════════════════════════
class S5_Discoveries(Scene):
    def construct(self):
        # ── Title with hieroglyphic accents ───────────────────────────────
        title = title_text("What the model found", color=WHITE, scale=0.75)
        title.move_to(UP * 3.3)
        div = hiero_divider(y=2.8)
        self.play(FadeIn(title), FadeIn(div), run_time=1.5)
        self.wait(0.5)

        # ── Golden Hit: mw → water ───────────────────────────────────────
        water_glyph = hiero_text(HIERO["water"], color=GOLD, scale=0.7)
        water_eg = Text("mw", color=GOLD).scale(0.4)
        water_arrow = Text("→", color=LAVENDER).scale(0.5)
        water_en = Text("water", color=TEAL).scale(0.4)
        water_score = Text("score: 15.71", color=MUTED).scale(0.28)

        water_row = VGroup(water_glyph, water_eg, water_arrow, water_en).arrange(RIGHT, buff=0.3)
        water_row.move_to(UP * 1.8)
        water_score.next_to(water_row, RIGHT, buff=0.3)

        water_note = body_text(
            "Perfect hit. Overwhelmingly confident — basic physical\n"
            "concepts align across 4,000 years.",
            color=MUTED
        ).scale(0.85).next_to(water_row, DOWN, buff=0.2)

        self.play(FadeIn(water_glyph), FadeIn(water_eg), run_time=1.5)
        self.play(FadeIn(water_arrow), FadeIn(water_en), FadeIn(water_score), run_time=1.5)
        self.play(FadeIn(water_note), run_time=2)
        self.wait(3)

        # ── Golden Hit: Anubis → imiut (not "dog") ──────────────────────
        anubis_glyph = hiero_text(HIERO["snake"], color=GOLD, scale=0.7)
        anubis_eg = Text("inpw (Anubis)", color=GOLD).scale(0.35)
        anubis_arrow = Text("→", color=LAVENDER).scale(0.5)
        anubis_en = Text("imiut", color=TEAL).scale(0.4)

        anubis_row = VGroup(anubis_glyph, anubis_eg, anubis_arrow, anubis_en).arrange(RIGHT, buff=0.25)
        anubis_row.move_to(DOWN * 0.1)

        anubis_note = body_text(
            'Not "dog" or "god" — his ritual symbol.\n'
            "The model captured context, not definition.",
            color=MUTED
        ).scale(0.85).next_to(anubis_row, DOWN, buff=0.2)

        # Fade out water hit, bring in Anubis
        self.play(
            FadeOut(water_note), FadeOut(water_score),
            water_row.animate.set_opacity(0.3),
            run_time=1.2
        )
        self.play(FadeIn(anubis_glyph), FadeIn(anubis_eg), run_time=1.5)
        self.play(FadeIn(anubis_arrow), FadeIn(anubis_en), run_time=1.5)
        self.play(FadeIn(anubis_note), run_time=2)
        self.wait(3)

        # ── Failure: Beer Tragedy ────────────────────────────────────────
        beer_glyph = hiero_text(HIERO["bread"], color=GOLD, scale=0.7)
        beer_eg = Text("hqt (beer)", color=GOLD).scale(0.35)
        beer_arrow = Text("→", color=SOFT_RED).scale(0.5)
        beer_en = Text('"good", "royal"', color=SOFT_RED).scale(0.35)

        beer_row = VGroup(beer_glyph, beer_eg, beer_arrow, beer_en).arrange(RIGHT, buff=0.25)
        beer_row.move_to(DOWN * 1.8)

        beer_note = body_text(
            '"An offering which the King gives... bread and beer."\n'
            "Offering formulas made beer = royalty.",
            color=MUTED
        ).scale(0.85).next_to(beer_row, DOWN, buff=0.2)

        self.play(
            FadeOut(anubis_note),
            anubis_row.animate.set_opacity(0.3),
            run_time=1.2
        )
        self.play(FadeIn(beer_glyph), FadeIn(beer_eg), run_time=1.5)
        self.play(FadeIn(beer_arrow), FadeIn(beer_en), run_time=1.5)
        self.play(FadeIn(beer_note), run_time=2)
        self.wait(3)

        # ── Core insight ─────────────────────────────────────────────────
        self.play(
            FadeOut(beer_note),
            beer_row.animate.set_opacity(0.3),
            run_time=1.2
        )

        insight = body_text(
            "The models don't know what words are —\n"
            "only what they appear next to.",
            color=WHITE
        ).scale(1.1).move_to(DOWN * 2.8)

        self.play(Write(insight), run_time=3)
        self.wait(4)


# ══════════════════════════════════════════════════════════════════════════════
# S6 — What We Learned  (2:50 – 3:10)
#       Methodological takeaways with hieroglyphic accents
# ══════════════════════════════════════════════════════════════════════════════
class S6_Lessons(Scene):
    def construct(self):
        header = VGroup(
            hiero_text(HIERO["eye"], color=GOLD, scale=0.35),
            title_text("What we learned", color=GOLD, scale=0.7),
            hiero_text(HIERO["eye"], color=GOLD, scale=0.35),
        ).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)

        lessons = [
            ("Linear > Neural",
             "Simple algebra beat deep learning on this task."),
            ("Quality > Quantity",
             "Fewer clean anchors outperformed more noisy ones."),
            ("Dimensionality matters",
             "Bigger spaces helped — even padded with zeros."),
            ("Modern NLP fails here",
             "BERT's tokenizer destroyed hieroglyphic structure."),
        ]

        items = VGroup()
        for t, d in lessons:
            title = Text(t, color=WHITE).scale(0.4)
            detail = Text(d, color=MUTED).scale(0.3)
            pair = VGroup(title, detail).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            items.add(pair)

        items.arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        items.next_to(header, DOWN, buff=0.6)
        items.move_to(ORIGIN + DOWN * 0.2)

        self.play(FadeIn(header), run_time=1.5)

        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.15), run_time=1.5)
            self.wait(0.8)

        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# S7 — The Claim  (3:10 – 3:20)
# ══════════════════════════════════════════════════════════════════════════════
class S7_Claim(Scene):
    def construct(self):
        # Subtle hieroglyphs in background
        bg_glyphs = hiero_text(
            f"{HIERO['scarab']}   {HIERO['djed']}   {HIERO['water']}   {HIERO['sky']}   {HIERO['eye']}",
            color=GOLD, scale=0.5
        ).set_opacity(0.12).move_to(UP * 0.3)
        self.add(bg_glyphs)

        lines = VGroup(
            title_text("The bottleneck isn't the data.", color=WHITE, scale=0.85),
            body_text("It's the resolution of our interpretive tools.",
                      color=LAVENDER).scale(1.2),
        ).arrange(DOWN, buff=0.5)

        self.play(Write(lines[0]), run_time=3)
        self.wait(0.5)
        self.play(FadeIn(lines[1], shift=UP * 0.1), run_time=2.5)
        self.wait(3)


# ══════════════════════════════════════════════════════════════════════════════
# S8 — Close  (3:20 – 3:30)
# ══════════════════════════════════════════════════════════════════════════════
class S8_Close(Scene):
    def construct(self):
        # Decorative hieroglyph strip at top
        top_glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=0.3)
        top_glyphs.set_opacity(0.25).move_to(UP * 3.0)
        self.add(top_glyphs)

        name   = title_text("Erik Brinsmead", color=WHITE, scale=0.8)
        fellow = body_text("Encode: AI for Science Fellowship — Cohort 2",
                           color=TEAL).scale(1.0)
        repo   = body_text("github.com/ebrinz/heiroglyphy",
                           color=GOLD).scale(0.95)

        group = VGroup(name, fellow, repo).arrange(DOWN, buff=0.5)
        self.play(
            LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in group], lag_ratio=0.4),
            run_time=3.5
        )
        self.wait(4)


# ══════════════════════════════════════════════════════════════════════════════
# Full Video — renders all scenes in sequence
# ══════════════════════════════════════════════════════════════════════════════
class HeiroglyphyVideo(Scene):
    """
    Composite scene. Renders all sections in order.
    Run with: manim -pqh heiroglyphy_video.py HeiroglyphyVideo
    """
    def construct(self):
        for SceneClass in [
            S1_Question,
            S2_Motivation,
            S3_Hypothesis,
            S4_Journey,
            S5_Discoveries,
            S6_Lessons,
            S7_Claim,
            S8_Close,
        ]:
            SceneClass.construct(self)
            self.clear()
            self.wait(0.5)
