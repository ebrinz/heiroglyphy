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

# ── Utterance 213 — Pyramid Text ─────────────────────────────────────────────
UTT_213 = {
    "glyphs": {
        "ankh":  "\U000132F9",   # ꜥnḫ — life (S34)
        "god":   "\U000132B9",   # nṯr — god (R8)
        "soul":  "\U00013161",   # bꜣ — soul (G29)
        "power": "\U00013302",   # sḫm — power (S42)
        "spirit":"\U0001315C",   # ꜣḫ — spirit (G25)
        "thoth": "\U00013043",   # Thoth (C3)
    },
    "literal": (
        "Live! Live! — for this is your name among the gods.\n"
        "A soul indeed, foremost of the living.\n"
        "Powerful indeed, foremost of the spirits."
    ),
    "reframed": [
        ("Live", "\U000132F9", "ꜥnḫ", "not biological life — divine permanence"),
        ("gods", "\U000132B9", "nṯr", "indistinguishable from gold — not metaphor, ontology"),
        ("soul", "\U00013161", "bꜣ", "the animating force — not ethereal, but powerful"),
        ("powerful", "\U00013302", "sḫm", "the same force as truth — māʿat"),
        ("spirits", "\U0001315C", "ꜣḫ.w", "the transfigured — power through knowledge"),
    ],
    "contextual": (
        "Endure! Endure! — for this is your name among the golden ones.\n"
        "A force indeed, foremost of those who persist.\n"
        "True indeed, foremost of those who achieved power through knowing."
    ),
}

GLYPH_STRIP = " ".join([
    UTT_213["glyphs"]["ankh"], UTT_213["glyphs"]["ankh"],
    UTT_213["glyphs"]["god"], UTT_213["glyphs"]["god"],
    UTT_213["glyphs"]["soul"],
    UTT_213["glyphs"]["ankh"],
    UTT_213["glyphs"]["power"],
    UTT_213["glyphs"]["spirit"],
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


BRIDGE_SCORES = DOCS_DIR / "bridge_scores.json"

def load_bridge_scores():
    with open(BRIDGE_SCORES) as f:
        return json.load(f)

def score_overlay(scene, term, literal, alignment, midpoint):
    """Show alignment + midpoint score overlay in upper-right. Returns VGroup."""
    lines = []
    lines.append(Text(f"{term} → \"{literal}\"", color=MUTED).scale(0.22))
    if alignment is not None and alignment > 0:
        lines.append(Text(f"alignment: {alignment:.3f}", color=TEAL).scale(0.2))
    if midpoint is not None:
        lines.append(Text(f"midpoint: {midpoint:.3f}", color=LAVENDER).scale(0.2))
    overlay = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
    overlay.move_to(RIGHT * 5.5 + UP * 2.8)
    bg = SurroundingRectangle(overlay, color=MUTED, fill_color=BG,
                               fill_opacity=0.85, stroke_width=0.5, buff=0.1)
    overlay_group = VGroup(bg, overlay)
    scene.play(FadeIn(overlay_group), run_time=1)
    return overlay_group


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
# S0 — Title Card  (~8s, no narration)
# ══════════════════════════════════════════════════════════════════════════════
class S0_Title(Scene):
    """Dramatic title card — Thoth glyph + full paper title. No narration."""
    def construct(self):
        # Thoth emerges from darkness — large, centered
        thoth = hiero_text("\U00013043", color=GOLD, scale=3.0)
        thoth.move_to(UP * 0.8)
        thoth.set_opacity(0)

        # Glow behind Thoth
        glow = Dot(point=thoth.get_center(), radius=1.5, color=GOLD).set_opacity(0)

        self.play(
            glow.animate.set_opacity(0.08),
            thoth.animate.set_opacity(1),
            run_time=2, rate_func=smooth
        )

        # Title text
        main_title = Text("The Geometry of Meaning", color=WHITE, weight=BOLD).scale(0.75)
        subtitle = Text(
            "Vector Space Alignment and the Ancient Egyptian Worldview",
            color=MUTED
        ).scale(0.32)
        title_group = VGroup(main_title, subtitle).arrange(DOWN, buff=0.2)
        title_group.move_to(DOWN * 1.8)

        self.play(FadeIn(main_title, shift=UP * 0.15), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(1.5)

        self.play(
            FadeOut(thoth), FadeOut(glow),
            FadeOut(main_title), FadeOut(subtitle),
            run_time=1
        )


# ══════════════════════════════════════════════════════════════════════════════
class S1_Hook(Scene):
    def construct(self):
        # ── Glyph strip + narration flow ──
        glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=1.0)
        glyphs.move_to(UP * 1.5)

        # [Glyphs appear] "These symbols are four thousand years old."
        self.play(FadeIn(glyphs, shift=UP * 0.2), run_time=3)
        self.wait(3)

        # "Scholars have been translating them for two centuries."
        line1 = body_text(
            "For 200 years, scholars have translated these symbols.",
            color=WHITE
        ).scale(1.1).next_to(glyphs, DOWN, buff=0.8)
        self.play(Write(line1), run_time=3)
        self.wait(3)

        # "But translation is lossy. When you compress a word into a single
        #  English equivalent, the web of meaning around it disappears."
        line2 = body_text(
            "But translation compresses meaning.\n"
            "The relationships between words are lost.",
            color=MUTED
        ).scale(1.0).next_to(line1, DOWN, buff=0.5)
        self.play(Write(line2), run_time=3.5)
        self.wait(3.5)

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
        self.wait(2)

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
        self.wait(2.5)

        explain = body_text(
            "Words that appear in similar contexts\n"
            "end up close together in this space.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 1.5)
        self.play(Write(explain), run_time=3)
        self.wait(2.5)

        # "This works for every language. Including Ancient Egyptian."
        explain2 = body_text(
            "This works for every language — including Ancient Egyptian.",
            color=LAVENDER
        ).scale(1.0).move_to(DOWN * 2.8)
        self.play(FadeIn(explain2, shift=UP * 0.1), run_time=2)
        self.wait(3)


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
        self.wait(1.5)
        self.play(FadeIn(label_en), Create(en_cloud), run_time=3)
        self.wait(1.5)

        # Phase 2: "The shapes are similar — but rotated."
        explain = body_text(
            "Both languages form a shape.\n"
            "The shapes are similar — but rotated.",
            color=WHITE
        ).scale(1.0).move_to(DOWN * 3.0)
        self.play(Write(explain), run_time=3)
        self.wait(2)

        # Phase 3: Clouds merge — both move toward center
        self.play(FadeOut(explain), run_time=0.5)
        finding = body_text("Find the rotation...", color=LAVENDER).scale(1.1)
        finding.move_to(DOWN * 3.0)
        self.play(FadeIn(finding), run_time=1)

        self.play(
            eg_cloud.animate.shift(RIGHT * 3.2).scale(1.3).rotate(0.25),
            en_cloud.animate.shift(LEFT * 3.2).rotate(-0.1),
            label_eg.animate.shift(RIGHT * 2),
            label_en.animate.shift(LEFT * 2),
            run_time=5, rate_func=smooth
        )
        self.wait(1)

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
        self.wait(1.7)

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

            # Use hieroglyphic glyph for Egyptian label where possible
            eg_glyph_map = {
                "mw": "\U00013217",      # water
                "nṯr": "\U000132B9",     # god
                "nswt": "\U00013216",     # king
            }
            eg_char = eg_glyph_map.get(hl["egyptian"])
            if eg_char:
                eg_label = hiero_text(eg_char, color=GOLD, scale=0.3)
            else:
                eg_label = Text(hl["egyptian"], color=GOLD).scale(0.25)
            eg_label.next_to(best_eg, LEFT, buff=0.1)
            en_label = Text(hl["english"], color=TEAL).scale(0.25)
            en_label.next_to(best_en, RIGHT, buff=0.1)
            highlight_labels.add(eg_label, en_label)

        self.play(Create(highlight_lines), FadeIn(highlight_labels), run_time=2)
        self.wait(3)

        self.wait(3)


class S_Bridge(Scene):
    """Explain 32% accuracy and introduce bridge/midpoint score concepts. ~25s"""
    def construct(self):
        # Phase 1: "1 in 3" stat
        big_stat = Text("1 in 3", color=GOLD).scale(1.5)
        sub = body_text("Egyptian words land on their correct English meaning", color=WHITE).scale(0.9)
        stat_group = VGroup(big_stat, sub).arrange(DOWN, buff=0.3).move_to(UP * 1)
        self.play(FadeIn(big_stat, scale=0.8), run_time=2)
        self.wait(2)
        self.play(FadeIn(sub), run_time=1.5)
        self.wait(3)

        qualifier = body_text("No dictionary. No bilingual text.\nJust the shape of meaning.", color=MUTED).scale(0.8)
        qualifier.next_to(stat_group, DOWN, buff=0.5)
        self.play(FadeIn(qualifier, shift=UP * 0.1), run_time=2)
        self.wait(3)

        self.play(FadeOut(stat_group), FadeOut(qualifier), run_time=1)

        # Phase 2: Bridge score concept (left side)
        bridge_title = Text("Alignment score", color=TEAL).scale(0.6).move_to(UP * 2.5 + LEFT * 3)
        dot_eg = Dot(point=[-4, 0.5, 0], radius=0.1, color=GOLD)
        dot_en = Dot(point=[-2, 0.5, 0], radius=0.1, color=TEAL)
        lbl_eg = Text("nṯr", color=GOLD).scale(0.3).next_to(dot_eg, DOWN, buff=0.1)
        lbl_en = Text("god", color=TEAL).scale(0.3).next_to(dot_en, DOWN, buff=0.1)
        bridge_line = Line(dot_eg.get_center(), dot_en.get_center(), color=LAVENDER, stroke_width=2)
        score_lbl = Text("0.929", color=LAVENDER).scale(0.25).next_to(bridge_line, UP, buff=0.05)
        bridge_desc = body_text("How closely a word's\nneighborhood matches\nacross languages", color=MUTED).scale(0.7)
        bridge_desc.move_to(LEFT * 3 + DOWN * 1.2)

        self.play(FadeIn(bridge_title), run_time=1)
        self.play(FadeIn(dot_eg, lbl_eg), FadeIn(dot_en, lbl_en), Create(bridge_line), FadeIn(score_lbl), run_time=2)
        self.play(FadeIn(bridge_desc), run_time=1.5)
        self.wait(2)

        # Phase 3: Midpoint score concept (right side)
        mid_title = Text("Midpoint score", color=TEAL).scale(0.6).move_to(UP * 2.5 + RIGHT * 3)
        dot_a = Dot(point=[1.5, 0.5, 0], radius=0.1, color=TEAL)
        dot_b = Dot(point=[4.5, 0.5, 0], radius=0.1, color=TEAL)
        lbl_a = Text("gold", color=TEAL).scale(0.3).next_to(dot_a, DOWN, buff=0.1)
        lbl_b = Text("divine", color=TEAL).scale(0.3).next_to(dot_b, DOWN, buff=0.1)
        dot_mid = Dot(point=[3, 0.5, 0], radius=0.08, color=WHITE).set_opacity(0.6)
        dash_l = DashedLine(dot_a.get_center(), dot_mid.get_center(), color=MUTED, stroke_width=1)
        dash_r = DashedLine(dot_mid.get_center(), dot_b.get_center(), color=MUTED, stroke_width=1)
        mid_score = Text("0.642", color=LAVENDER).scale(0.25).next_to(dot_mid, UP, buff=0.1)
        mid_desc = body_text("How strongly two concepts\nconverge in the Egyptian\nworldview", color=MUTED).scale(0.7)
        mid_desc.move_to(RIGHT * 3 + DOWN * 1.2)

        self.play(FadeIn(mid_title), run_time=1)
        self.play(FadeIn(dot_a, lbl_a), FadeIn(dot_b, lbl_b), run_time=1)
        self.play(FadeIn(dot_mid), Create(dash_l), Create(dash_r), FadeIn(mid_score), run_time=1.5)
        self.play(FadeIn(mid_desc), run_time=1.5)
        self.wait(2)

        closing = body_text("Here's what those numbers revealed.", color=LAVENDER).scale(1.0)
        closing.move_to(DOWN * 2.8)
        self.play(FadeIn(closing, shift=UP * 0.1), run_time=2)
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
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1.5)

        # Two concept dots on opposite sides
        dot_gold = Dot(point=[-3, 0.5, 0], radius=0.12, color="#f1c40f").set_opacity(0.9)
        dot_divine = Dot(point=[3, 0.5, 0], radius=0.12, color=LAVENDER).set_opacity(0.9)
        lbl_gold = Text("gold", color="#f1c40f").scale(0.4).next_to(dot_gold, DOWN, buff=0.15)
        lbl_divine = Text("divine", color=LAVENDER).scale(0.4).next_to(dot_divine, DOWN, buff=0.15)

        self.play(FadeIn(dot_gold, lbl_gold), FadeIn(dot_divine, lbl_divine), run_time=2)
        self.wait(2)

        # Midpoint marker
        midpoint = Dot(point=[0, 0.5, 0], radius=0.08, color=WHITE).set_opacity(0.6)
        mid_label = Text("midpoint", color=MUTED).scale(0.25).next_to(midpoint, UP, buff=0.1)
        dashed_left = DashedLine(dot_gold.get_center(), midpoint.get_center(), color=MUTED, stroke_width=1)
        dashed_right = DashedLine(midpoint.get_center(), dot_divine.get_center(), color=MUTED, stroke_width=1)

        self.play(Create(dashed_left), Create(dashed_right), FadeIn(midpoint, mid_label), run_time=2)
        self.wait(1.5)

        # Arrow projects down into "Egyptian space"
        eg_label = Text("Egyptian space", color=GOLD).scale(0.3).move_to(DOWN * 0.7)
        arrow = Arrow(midpoint.get_center(), DOWN * 1.0, color=MUTED, stroke_width=2)
        self.play(Create(arrow), FadeIn(eg_label), run_time=1.5)

        # Egyptian results appear — spread apart to show they start as separate concepts
        dot_ntri = Dot(point=[-2, -1.5, 0], radius=0.1, color=GOLD)
        dot_nbw = Dot(point=[2, -1.5, 0], radius=0.1, color=GOLD)
        # Use hieroglyphic glyphs
        glyph_ntri = hiero_text("\U00013291", color=GOLD, scale=0.5)
        lbl_ntri_eng = Text("(divine)", color=GOLD).scale(0.28)
        lbl_ntri = VGroup(glyph_ntri, lbl_ntri_eng).arrange(RIGHT, buff=0.1).next_to(dot_ntri, DOWN, buff=0.15)
        glyph_nbw = hiero_text("\U00013210", color=GOLD, scale=0.5)
        lbl_nbw_eng = Text("(gold)", color=GOLD).scale(0.28)
        lbl_nbw = VGroup(glyph_nbw, lbl_nbw_eng).arrange(RIGHT, buff=0.1).next_to(dot_nbw, DOWN, buff=0.15)

        self.play(FadeIn(dot_ntri, lbl_ntri), FadeIn(dot_nbw, lbl_nbw), run_time=2)
        self.wait(2)

        scores = load_bridge_scores()["discoveries"]["D1_Gold"]
        overlay = score_overlay(self, scores["primary_term"], scores["literal"], scores.get("alignment_score"), scores["midpoint_score"])
        self.wait(1)

        # Dots merge — they converge to show they're the same concept
        self.play(
            dot_ntri.animate.move_to([0, -1.5, 0]),
            dot_nbw.animate.move_to([0, -1.5, 0]),
            lbl_ntri.animate.move_to([-1.5, -2.2, 0]),
            lbl_nbw.animate.move_to([1.5, -2.2, 0]),
            run_time=3
        )

        # Glow effect
        glow = Dot(point=[0, -1.5, 0], radius=0.3, color=GOLD).set_opacity(0.3)
        self.play(FadeIn(glow), run_time=1)
        self.wait(3)

        # Punchline
        punchline = body_text("Not metaphor. It's an ontology.", color=WHITE).scale(1.2).move_to(DOWN * 2.8)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(12)


class D2_Silence(Scene):
    def construct(self):
        glyph = hiero_text("\U000131EF\U0001337F\U000132BD", color=GOLD, scale=0.4)
        title = Text("Silence Is the Condition of the Dead", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1.5)

        # Sound wave
        wave_dots = VGroup()
        n_pts = 80
        for i in range(n_pts):
            x = (i / n_pts) * 10 - 5
            y = 0.6 * np.sin(i * 0.3) * np.exp(-abs(x) * 0.05)
            wave_dots.add(
                Dot(point=[x, y + 1.0, 0], radius=0.03, color=TEAL).set_opacity(0.7)
            )

        self.play(Create(wave_dots), run_time=2)
        self.wait(1.5)

        # Wave flattens to silence
        flat_targets = []
        for i, dot in enumerate(wave_dots):
            x = dot.get_center()[0]
            flat_targets.append(dot.animate.move_to([x, 1.0, 0]).set_opacity(0.2))

        self.play(*flat_targets, run_time=3, rate_func=smooth)
        self.wait(1)

        # "silence" and "death" dots converge
        dot_silence = Dot(point=[-2, -0.5, 0], radius=0.1, color=LAVENDER)
        dot_death = Dot(point=[2, -0.5, 0], radius=0.1, color=SOFT_RED)
        lbl_silence = Text("silence", color=LAVENDER).scale(0.35).next_to(dot_silence, DOWN, buff=0.1)
        lbl_death = Text("death", color=SOFT_RED).scale(0.35).next_to(dot_death, DOWN, buff=0.1)

        self.play(FadeIn(dot_silence, lbl_silence), FadeIn(dot_death, lbl_death), run_time=1.5)
        self.wait(1.5)

        # Converge to same point
        converge_pt = [0, -1.0, 0]
        self.play(
            dot_silence.animate.move_to(converge_pt),
            dot_death.animate.move_to(converge_pt),
            lbl_silence.animate.next_to(converge_pt, LEFT, buff=0.3),
            lbl_death.animate.next_to(converge_pt, RIGHT, buff=0.3),
            run_time=2.5
        )
        self.wait(1.5)

        # death/silence hieroglyphic cluster — use glyphs instead of Latin transliterations
        mwt_glyphs = [
            "\U00013000",   # seated man (determinative)
            "\U00013196",   # snake
            "\U000132BD",   # djed pillar
            "\U00013000\U00013196",  # man + snake
            "\U000132BD\U00013000",  # djed + man
        ]
        mwt_group = VGroup()
        rng = np.random.default_rng(7)
        for g in mwt_glyphs:
            offset = rng.uniform(-0.3, 0.3, 2)
            lbl = hiero_text(g, color=GOLD, scale=0.35)
            lbl.move_to([converge_pt[0] + offset[0], converge_pt[1] - 0.6 + offset[1], 0])
            mwt_group.add(lbl)

        self.play(FadeIn(mwt_group), run_time=1.5)
        self.wait(1.5)

        scores = load_bridge_scores()["discoveries"]["D2_Silence"]
        overlay = score_overlay(self, scores["primary_term"], scores["literal"], scores.get("alignment_score"), scores["midpoint_score"])
        self.wait(1.5)

        # Punchline
        punchline = body_text("What the dead lost was not life. It was voice.", color=WHITE).scale(1.1).move_to(DOWN * 2.8)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(6)


class D3_Seeing(Scene):
    def construct(self):
        glyph = hiero_text("\U00013080\U000133DB\U000131B4", color=GOLD, scale=0.4)
        title = Text("Seeing Was an Act of Magical Power", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1.5)

        # Phase 1: English distances — these concepts are far apart
        en_title = Text("In English", color=TEAL).scale(0.4).move_to(UP * 2.2)
        self.play(FadeIn(en_title), run_time=1)

        # English dots spread far apart
        en_concepts = [
            ("eye", [-3, 0.8, 0], TEAL),
            ("knowledge", [0, 1.2, 0], TEAL),
            ("magic", [3, 0.5, 0], TEAL),
            ("power", [1, -0.5, 0], TEAL),
        ]
        en_dots = VGroup()
        en_labels = VGroup()
        for word, pos, color in en_concepts:
            dot = Dot(point=pos, radius=0.08, color=color).set_opacity(0.7)
            label = Text(word, color=color).scale(0.3).next_to(dot, DOWN, buff=0.08)
            en_dots.add(dot)
            en_labels.add(label)

        self.play(
            LaggedStart(*[FadeIn(d) for d in en_dots], lag_ratio=0.2),
            LaggedStart(*[FadeIn(l) for l in en_labels], lag_ratio=0.2),
            run_time=2
        )

        # Show distance scores between pairs
        dist_labels = VGroup()
        pairs = [(0, 2, "0.22"), (0, 3, "0.19"), (1, 2, "0.27")]
        for i, j, score in pairs:
            mid = (en_dots[i].get_center() + en_dots[j].get_center()) / 2
            line = DashedLine(en_dots[i].get_center(), en_dots[j].get_center(),
                              color=MUTED, stroke_width=1).set_opacity(0.3)
            lbl = Text(score, color=MUTED).scale(0.2).move_to(mid + UP * 0.15)
            dist_labels.add(line, lbl)

        self.play(Create(dist_labels), run_time=2)
        self.wait(2)

        # Phase 2: Egyptian — these concepts collapse together
        self.play(
            FadeOut(en_dots), FadeOut(en_labels), FadeOut(dist_labels), FadeOut(en_title),
            run_time=1
        )

        eg_title = Text("In Egyptian", color=GOLD).scale(0.4).move_to(UP * 2.2)
        self.play(FadeIn(eg_title), run_time=1)

        # Eye of Horus glyph — large, dramatic, upper left
        eye = hiero_text("\U00013080", color=GOLD, scale=2.0)
        eye.move_to(LEFT * 3.5 + UP * 0.5)
        self.play(FadeIn(eye, scale=0.8), run_time=2)
        self.wait(0.5)

        # Egyptian dots — spread out with clear spacing
        # Eye dot near the glyph
        dot_eye = Dot(point=[-1.5, 0.5, 0], radius=0.1, color=GOLD).set_opacity(0.8)
        lbl_eye = hiero_text("\U00013080", color=GOLD, scale=0.35)
        lbl_eye_eng = Text("eye", color=GOLD).scale(0.25)
        lbl_eye_group = VGroup(lbl_eye, lbl_eye_eng).arrange(RIGHT, buff=0.08).next_to(dot_eye, UP, buff=0.12)

        # Power dot — far right
        dot_power = Dot(point=[3, 0.8, 0], radius=0.1, color=LAVENDER).set_opacity(0.8)
        lbl_power = hiero_text("\U00013302", color=LAVENDER, scale=0.35)
        lbl_power_eng = Text("power", color=LAVENDER).scale(0.25)
        lbl_power_group = VGroup(lbl_power, lbl_power_eng).arrange(RIGHT, buff=0.08).next_to(dot_power, UP, buff=0.12)

        # Magic dot — lower center
        dot_magic = Dot(point=[0.5, -1.5, 0], radius=0.1, color=TEAL).set_opacity(0.8)
        lbl_magic_glyph = hiero_text("\U0001339B\U00013093", color=TEAL, scale=0.35)
        lbl_magic_eng = Text("magic", color=TEAL).scale(0.25)
        lbl_magic = VGroup(lbl_magic_glyph, lbl_magic_eng).arrange(RIGHT, buff=0.08).next_to(dot_magic, DOWN, buff=0.12)

        self.play(FadeIn(dot_eye, lbl_eye_group), run_time=1)
        self.play(FadeIn(dot_power, lbl_power_group), run_time=1)
        self.play(FadeIn(dot_magic, lbl_magic), run_time=1)
        self.wait(0.5)

        # Distance lines with scores — clearly positioned
        line_ep = Line(dot_eye.get_center(), dot_power.get_center(),
                       color=LAVENDER, stroke_width=2).set_opacity(0.5)
        score_ep = Text("0.62", color=LAVENDER).scale(0.25)
        score_ep.move_to((dot_eye.get_center() + dot_power.get_center()) / 2 + UP * 0.25)

        line_em = Line(dot_eye.get_center(), dot_magic.get_center(),
                       color=TEAL, stroke_width=2).set_opacity(0.5)
        score_em = Text("0.44", color=TEAL).scale(0.25)
        score_em.move_to((dot_eye.get_center() + dot_magic.get_center()) / 2 + LEFT * 0.3)

        self.play(Create(line_ep), FadeIn(score_ep), run_time=1.5)
        self.play(Create(line_em), FadeIn(score_em), run_time=1.5)
        self.wait(1)

        scores = load_bridge_scores()["discoveries"]["D3_Seeing"]
        overlay = score_overlay(self, scores["primary_term"], scores["literal"], scores.get("alignment_score"), scores["midpoint_score"])
        self.wait(2)

        # Punchline
        punch1 = body_text("In English, sight and power are unrelated.", color=WHITE).scale(0.9)
        punch2 = body_text("In Egyptian, the eye IS power. Sight is magic.", color=WHITE).scale(0.9)
        punchline = VGroup(punch1, punch2).arrange(DOWN, buff=0.15).move_to(DOWN * 2.5)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(6)


class D4_Snake(Scene):
    def construct(self):
        glyph = hiero_text("\U00013196", color=GOLD, scale=0.5)
        title = Text("What Does a Snake Mean?", color=GOLD).scale(0.55)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1.5)

        # Big snake glyph — dramatic center
        snake = hiero_text("\U00013196", color=GOLD, scale=2.5)
        snake.move_to(ORIGIN + UP * 0.3)
        self.play(FadeIn(snake, scale=0.7), run_time=2)
        self.wait(1.5)

        # Phase 1: English neighbors — animals
        self.play(snake.animate.scale(0.35).move_to(ORIGIN + UP * 2.0), run_time=1.5)

        en_title = Text("In English", color=TEAL).scale(0.35).next_to(snake, LEFT, buff=0.3)
        self.play(FadeIn(en_title), run_time=0.5)

        en_neighbors = ["frog", "monkey", "crocodile", "fish", "cat"]
        en_group = VGroup()
        for i, w in enumerate(en_neighbors):
            lbl = Text(w, color=TEAL).scale(0.28).set_opacity(0.7)
            en_group.add(lbl)
        en_group.arrange(RIGHT, buff=0.5).move_to(UP * 1.0)

        self.play(
            LaggedStart(*[FadeIn(l) for l in en_group], lag_ratio=0.15),
            run_time=2
        )
        self.wait(2)

        # Phase 2: Egyptian neighbors — gods
        eg_title = Text("In Egyptian", color=GOLD).scale(0.35).move_to(LEFT * 2 + DOWN * 0.3)
        self.play(FadeIn(eg_title), run_time=0.5)

        eg_results = [
            ("\U000132B9", "god"),
            ("\U000132B9\U000132B9", "gods"),
            ("\U000132B9", "great god"),
        ]
        eg_group = VGroup()
        for eg_glyph, eng in eg_results:
            g = hiero_text(eg_glyph, color=GOLD, scale=0.35)
            e = Text(eng, color=GOLD).scale(0.25)
            pair = VGroup(g, e).arrange(RIGHT, buff=0.08)
            eg_group.add(pair)
        eg_group.arrange(RIGHT, buff=0.8).move_to(DOWN * 1.0)

        self.play(
            LaggedStart(*[FadeIn(l) for l in eg_group], lag_ratio=0.2),
            run_time=2
        )
        self.wait(1)

        scores = load_bridge_scores()["discoveries"]["D4_Snake"]
        overlay = score_overlay(self, scores["primary_term"], scores["literal"], scores.get("alignment_score"), scores["midpoint_score"])
        self.wait(2)

        # Clear the scene for punchline
        self.play(
            FadeOut(en_group), FadeOut(en_title), FadeOut(snake),
            FadeOut(eg_group), FadeOut(eg_title),
            run_time=1
        )

        # Punchline — clean, centered
        punch1 = body_text("In English, a snake is an animal.", color=MUTED).scale(0.95)
        punch2 = body_text("In Egyptian, the serpent is sacred.", color=GOLD).scale(1.05)
        punchline = VGroup(punch1, punch2).arrange(DOWN, buff=0.25).move_to(ORIGIN)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(5)


class D5_Temple(Scene):
    def construct(self):
        glyph = hiero_text("\U00013250\U000132BD\U00013000", color=GOLD, scale=0.4)
        title = Text("Temple : House :: God : Man", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1.5)

        # Four points of the parallelogram
        pts = {
            "house":  [-2.5, -0.5, 0],
            "temple": [-2.5, 1.5, 0],
            "man":    [2.5, -0.5, 0],
            "?":      [2.5, 1.5, 0],
        }

        dots = {}
        labels = {}
        top_words = {"temple", "?"}
        for word, pos in pts.items():
            color = TEAL if word != "?" else MUTED
            dot = Dot(point=pos, radius=0.12, color=color)
            direction = UP if word in top_words else DOWN
            label = Text(word, color=color).scale(0.4).next_to(dot, direction, buff=0.15)
            dots[word] = dot
            labels[word] = label

        self.play(
            *[FadeIn(dots[w], labels[w]) for w in ["house", "temple", "man"]],
            FadeIn(dots["?"], labels["?"]),
            run_time=2
        )
        self.wait(1.5)

        # Arrow from house → temple
        arrow_left = Arrow(
            dots["house"].get_center(), dots["temple"].get_center(),
            color=LAVENDER, stroke_width=3, buff=0.2
        )
        sacred_label = Text("sacred", color=LAVENDER).scale(0.25)
        sacred_label.next_to(arrow_left, LEFT, buff=0.1)
        self.play(Create(arrow_left), FadeIn(sacred_label), run_time=2)
        self.wait(2)

        # Same arrow from man → ?
        arrow_right = Arrow(
            dots["man"].get_center(), dots["?"].get_center(),
            color=LAVENDER, stroke_width=3, buff=0.2
        )
        sacred_label2 = Text("sacred", color=LAVENDER).scale(0.25)
        sacred_label2.next_to(arrow_right, RIGHT, buff=0.1)
        self.play(Create(arrow_right), FadeIn(sacred_label2), run_time=2)
        self.wait(1.5)

        # ? resolves to "god"
        god_label = Text("god", color=GOLD).scale(0.5).next_to(dots["?"], UP, buff=0.15)
        self.play(
            dots["?"].animate.set_color(GOLD),
            Transform(labels["?"], god_label),
            Flash(dots["?"].get_center(), color=GOLD, line_length=0.3, num_lines=8),
            run_time=1.5
        )
        self.wait(1)

        scores = load_bridge_scores()["discoveries"]["D5_Temple"]
        overlay = score_overlay(self, scores["primary_term"], scores["literal"], scores.get("alignment_score"), scores.get("analogy_score"))
        self.wait(2)

        # Parallel lines
        parallel_top = DashedLine(
            dots["temple"].get_center(), dots["?"].get_center(),
            color=MUTED, stroke_width=1
        ).set_opacity(0.4)
        parallel_bot = DashedLine(
            dots["house"].get_center(), dots["man"].get_center(),
            color=MUTED, stroke_width=1
        ).set_opacity(0.4)
        self.play(Create(parallel_top), Create(parallel_bot), run_time=1)
        self.wait(1.5)

        punchline = body_text("Vector arithmetic across 4,000 years.", color=WHITE).scale(1.1).move_to(DOWN * 2.8)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(7)


class D6_Mother(Scene):
    def construct(self):
        glyph = hiero_text("\U000130AD\U00013300\U00013000", color=GOLD, scale=0.4)
        title = Text("Mother Is Royalty, Not Earth", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1.5)

        dot_mother = Dot(point=[-2, 1.5, 0], radius=0.1, color=LAVENDER)
        dot_earth = Dot(point=[2, 1.5, 0], radius=0.1, color=TEAL)
        lbl_mother = Text("mother", color=LAVENDER).scale(0.35).next_to(dot_mother, UP, buff=0.1)
        lbl_earth = Text("earth", color=TEAL).scale(0.35).next_to(dot_earth, UP, buff=0.1)

        self.play(FadeIn(dot_mother, lbl_mother), FadeIn(dot_earth, lbl_earth), run_time=1.5)
        self.wait(1.5)

        # Expected results — flash each one sequentially
        expect_header = Text("expected:", color=MUTED).scale(0.3).set_opacity(0.5)
        expect_header.move_to(ORIGIN + UP * 0.2)
        self.play(FadeIn(expect_header), run_time=0.5)

        expected = ["soil", "fertility", "land", "harvest"]
        for w in expected:
            lbl = Text(w, color=MUTED).scale(0.45).set_opacity(0.4)
            lbl.move_to(ORIGIN + DOWN * 0.5)
            self.play(FadeIn(lbl), run_time=0.5)
            self.wait(0.3)
            self.play(FadeOut(lbl), run_time=0.3)

        self.play(FadeOut(expect_header), run_time=0.5)
        self.wait(1)

        # Actual results — flash each one, brighter
        actual_header = Text("actual:", color=GOLD).scale(0.3)
        actual_header.move_to(ORIGIN + UP * 0.2)
        self.play(FadeIn(actual_header), run_time=0.5)

        actual = ["king's wife", "king's daughter", "queen", "princess"]
        actual_labels = VGroup()
        for w in actual:
            lbl = Text(w, color=GOLD).scale(0.5)
            lbl.move_to(ORIGIN + DOWN * 0.5)
            self.play(FadeIn(lbl, scale=0.9), run_time=0.8)
            self.wait(0.5)
            actual_labels.add(lbl)
            if w != actual[-1]:
                self.play(FadeOut(lbl), run_time=0.3)

        # Keep the last one visible
        self.wait(1)

        scores = load_bridge_scores()["discoveries"]["D6_Mother"]
        overlay = score_overlay(self, scores["primary_term"], scores["literal"], scores.get("alignment_score"), scores["midpoint_score"])
        self.wait(1)

        # Clear for punchline
        self.play(FadeOut(actual_header), FadeOut(actual_labels), run_time=1)

        punchline = body_text("Motherhood is a crown, not the earth.", color=WHITE).scale(1.1).move_to(ORIGIN)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(5)


class D7_Truth(Scene):
    def construct(self):
        glyph = hiero_text("\U00013080\U000131B4\U000132BD", color=GOLD, scale=0.4)
        title = Text("Truth and Power Are the Same Force", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1.5)

        # Query: midpoint of "truth" and "power"
        dot_truth = Dot(point=[-2.5, 1.0, 0], radius=0.1, color=TEAL)
        dot_power = Dot(point=[2.5, 1.0, 0], radius=0.1, color=LAVENDER)
        lbl_truth = Text("truth", color=TEAL).scale(0.35).next_to(dot_truth, UP, buff=0.1)
        lbl_power = Text("power", color=LAVENDER).scale(0.35).next_to(dot_power, UP, buff=0.1)

        midpoint_dot = Dot(point=[0, 1.0, 0], radius=0.08, color=WHITE).set_opacity(0.6)
        dash_l = DashedLine(dot_truth.get_center(), midpoint_dot.get_center(), color=MUTED, stroke_width=1)
        dash_r = DashedLine(midpoint_dot.get_center(), dot_power.get_center(), color=MUTED, stroke_width=1)

        self.play(FadeIn(dot_truth, lbl_truth), FadeIn(dot_power, lbl_power), run_time=1.5)
        self.play(FadeIn(midpoint_dot), Create(dash_l), Create(dash_r), run_time=1.5)
        self.wait(1.5)

        # Results from Egyptian space — flash sequentially
        result_header = Text("Egyptian results:", color=GOLD).scale(0.3)
        result_header.move_to(ORIGIN + DOWN * 0.2)
        self.play(FadeIn(result_header), run_time=0.5)

        results = [
            ("\U00013302", "power/authority", "#1"),      # sḫm glyph (S42)
            ("\U000132B9", "gods", "#2"),                  # nṯr glyph (R8)
            ("\U000130C3\U000133CF", "enemies", "#4"),     # ḫft glyphs
        ]
        prev_row = None
        for glyph_char, eng, rank in results:
            rank_text = Text(rank, color=MUTED).scale(0.2)
            term_text = hiero_text(glyph_char, color=GOLD, scale=0.4)
            eng_text = Text(f"({eng})", color=MUTED).scale(0.25)
            row = VGroup(rank_text, term_text, eng_text).arrange(RIGHT, buff=0.15)
            row.move_to(ORIGIN + DOWN * 0.8)
            if prev_row:
                self.play(FadeOut(prev_row), run_time=0.3)
            self.play(FadeIn(row, shift=UP * 0.1), run_time=1)
            self.wait(0.8)
            prev_row = row

        self.wait(0.5)

        # Reveal: māʿat sits right next to sḫm
        self.play(FadeOut(prev_row), FadeOut(result_header), run_time=0.5)

        maat_glyph = hiero_text("\U00013184", color=GOLD, scale=0.5)
        maat_eng = Text("māʿat (truth/cosmic order)", color=GOLD).scale(0.3)
        maat_label = VGroup(maat_glyph, maat_eng).arrange(RIGHT, buff=0.15)
        power_glyph = hiero_text("\U00013302", color=LAVENDER, scale=0.4)
        power_eng = Text("(power)", color=LAVENDER).scale(0.25)
        maat_sub = VGroup(
            Text("sits next to", color=MUTED).scale(0.25),
            power_glyph, power_eng,
            Text("— distance 0.637", color=MUTED).scale(0.25)
        ).arrange(RIGHT, buff=0.1)
        maat_group = VGroup(maat_label, maat_sub).arrange(DOWN, buff=0.15).move_to(DOWN * 0.5)
        self.play(FadeIn(maat_group, scale=0.8), run_time=1.5)
        self.wait(1)

        scores = load_bridge_scores()["discoveries"]["D7_Truth"]
        overlay = score_overlay(self, scores["primary_term"], scores["literal"], scores.get("alignment_score"), scores["midpoint_score"])
        self.wait(1.5)

        # Clear for punchline
        self.play(
            FadeOut(dot_truth), FadeOut(dot_power), FadeOut(lbl_truth), FadeOut(lbl_power),
            FadeOut(midpoint_dot), FadeOut(dash_l), FadeOut(dash_r), FadeOut(maat_group),
            run_time=1
        )

        punchline = body_text("Truth is not correctness. It is force.", color=WHITE).scale(1.1).move_to(ORIGIN)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(5)


class D8_Eternity(Scene):
    def construct(self):
        glyph = hiero_text("\U000131A3\U000131EF\U000131B4", color=GOLD, scale=0.4)
        title = Text("Love and Fear Meet at Eternity", color=GOLD).scale(0.5)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
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

        # Suspense pause before reveal
        self.wait(1)

        # Midpoint appears
        dot_mid = Dot(point=[0, 0, 0], radius=0.15, color=GOLD).set_opacity(0)
        self.play(dot_mid.animate.set_opacity(0.9), run_time=2, rate_func=smooth)

        eternity_glyph = hiero_text("\U0001339B\U000131F3\U0001339B", color=GOLD, scale=0.4)
        eternity_eng = Text("eternity", color=WHITE).scale(0.45)
        eternity_group = VGroup(eternity_glyph, eternity_eng).arrange(DOWN, buff=0.08)
        eternity_group.next_to(dot_mid, UP, buff=0.2)

        self.play(FadeIn(eternity_group, shift=DOWN * 0.1), run_time=2)
        self.wait(1)

        scores = load_bridge_scores()["discoveries"]["D8_Eternity"]
        overlay = score_overlay(self, scores["primary_term"], scores["literal"], scores.get("alignment_score"), scores["midpoint_score"])
        self.wait(2)

        # Radiating rings
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
        self.wait(3)

        # Fade rings outward
        self.play(
            *[ring.animate.set_opacity(0).scale(1.2) for ring in rings],
            run_time=2
        )

        punchline = body_text("Between love and fear: forever.", color=WHITE).scale(1.1).move_to(DOWN * 2.8)
        self.play(FadeIn(punchline, shift=UP * 0.1), run_time=2)
        self.wait(6)


# ══════════════════════════════════════════════════════════════════════════════
class S6_Discussion(Scene):
    def construct(self):
        header = title_text("Honest caveats", color=WHITE, scale=0.7)
        header.move_to(UP * 3.2)
        self.play(FadeIn(header), run_time=1.5)
        self.wait(4)

        caveats = [
            ("Corpus bias", "The surviving texts are funerary and religious —\ntemples and tombs, not markets and homes."),
            ("32% accuracy", "Two-thirds of words don't find their match.\nThese are statistical tendencies, not certainties."),
            ("What it captures", "A dictionary says nṯr means 'god' and nbw means 'gold.'\nOnly the embedding space says they're the same word."),
        ]

        prev = None
        for title_str, detail_str in caveats:
            t = Text(title_str, color=GOLD).scale(0.55)
            d = Text(detail_str, color=MUTED).scale(0.38)
            group = VGroup(t, d).arrange(DOWN, buff=0.15)

            if prev is None:
                group.move_to(UP * 1.5)
            else:
                group.next_to(prev, DOWN, buff=0.5)

            self.play(FadeIn(group), run_time=2)
            self.wait(8)
            prev = group

        self.wait(4)


# ══════════════════════════════════════════════════════════════════════════════
class S7_Conclusion(Scene):
    """Full-circle: Utt. 213 returns, literal translation, relaxed word-by-word reframing."""
    def construct(self):
        # Phase 1: The same glyphs from S1 reappear
        glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=1.0)
        glyphs.move_to(UP * 2.5)
        self.play(FadeIn(glyphs, shift=UP * 0.2), run_time=3)
        self.wait(3)

        # Phase 2: Literal translation
        literal = Text(UTT_213["literal"], color=MUTED, line_spacing=1.3).scale(0.35)
        literal.next_to(glyphs, DOWN, buff=0.5)
        self.play(Write(literal), run_time=5)
        self.wait(3)

        # Phase 3: Word-by-word reframing — relaxed, one at a time, with glyphs
        reframe_group = VGroup()
        for eng_word, glyph_char, eg_term, insight in UTT_213["reframed"]:
            glyph_label = hiero_text(glyph_char, color=GOLD, scale=0.3)
            term_text = Text(f"{eng_word} ({eg_term})", color=GOLD).scale(0.28)
            insight_text = Text(f"→ {insight}", color=LAVENDER).scale(0.23)
            row = VGroup(glyph_label, term_text, insight_text).arrange(RIGHT, buff=0.15)
            reframe_group.add(row)

        reframe_group.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        reframe_group.next_to(literal, DOWN, buff=0.4)

        for row in reframe_group:
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=1.5)
            self.wait(2)

        self.wait(3)

        # Phase 4: Contextual re-translation
        self.play(FadeOut(literal), FadeOut(reframe_group), run_time=1)

        context_header = body_text("A new reading:", color=TEAL).scale(0.9)
        context_header.next_to(glyphs, DOWN, buff=0.4)
        contextual = Text(UTT_213["contextual"], color=WHITE, line_spacing=1.3).scale(0.35)
        contextual.next_to(context_header, DOWN, buff=0.3)

        self.play(FadeIn(context_header), run_time=1.5)
        self.play(Write(contextual), run_time=5)
        self.wait(5)


class S8_Outro(Scene):
    """Closing statement and repo link."""
    def construct(self):
        final = body_text("Translation gave us the words.", color=WHITE).scale(1.3).move_to(UP * 0.5)
        final2 = body_text("The vectors gave us the world between them.", color=LAVENDER).scale(1.2).move_to(DOWN * 0.8)

        self.play(Write(final), run_time=3)
        self.wait(2)
        self.play(FadeIn(final2, shift=UP * 0.1), run_time=2.5)
        self.wait(3)

        self.play(FadeOut(final), FadeOut(final2), run_time=1)
        repo = body_text("github.com/ebrinz/heiroglyphy", color=GOLD).scale(1.1)
        self.play(FadeIn(repo, shift=UP * 0.1), run_time=2)
        self.wait(5)


# ══════════════════════════════════════════════════════════════════════════════
# Full Video  (~6:20)
# ══════════════════════════════════════════════════════════════════════════════
class HeiroglyphyVideo(Scene):
    """
    Full version (~6:20).
    Run: manim -pqh heiroglyphy_video.py HeiroglyphyVideo
    """
    def construct(self):
        for SceneClass in [
            S0_Title,
            S1_Hook,
            S2_Idea,
            S3_Alignment,
            S_Bridge,
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
            S8_Outro,
        ]:
            SceneClass.construct(self)
            self.clear()
            self.wait(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Short Version Scenes (~3 min)
# ══════════════════════════════════════════════════════════════════════════════

class S0_Title_Short(Scene):
    """Title card — trimmed to 5s."""
    def construct(self):
        thoth = hiero_text("\U00013043", color=GOLD, scale=2.0)
        thoth.move_to(UP * 0.8)
        thoth.set_opacity(0)
        glow = Dot(point=[0, 0.8, 0], radius=1.5, color=GOLD).set_opacity(0)
        self.play(glow.animate.set_opacity(0.08), thoth.animate.set_opacity(1), run_time=1.5)
        main_title = Text("The Geometry of Meaning", color=WHITE, weight=BOLD).scale(0.75)
        subtitle = Text("Vector Space Alignment and the Ancient Egyptian Worldview", color=MUTED).scale(0.32)
        title_group = VGroup(main_title, subtitle).arrange(DOWN, buff=0.2).move_to(DOWN * 1.8)
        self.play(FadeIn(main_title, shift=UP * 0.15), run_time=1)
        self.play(FadeIn(subtitle), run_time=0.5)
        self.wait(1)
        self.play(FadeOut(thoth), FadeOut(glow), FadeOut(main_title), FadeOut(subtitle), run_time=1)


class S1_Short(Scene):
    """Hook — compact. ~18s"""
    def construct(self):
        glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=1.0)
        glyphs.move_to(UP * 1.0)
        self.play(FadeIn(glyphs, shift=UP * 0.2), run_time=2)
        self.wait(2)

        line1 = body_text(
            "These symbols are four thousand years old.\n"
            "Translation compresses their meaning.",
            color=WHITE
        ).scale(1.0).next_to(glyphs, DOWN, buff=0.6)
        self.play(Write(line1), run_time=4)
        self.wait(4)

        line2 = body_text("What if we could recover it?", color=LAVENDER).scale(1.1)
        line2.next_to(line1, DOWN, buff=0.5)
        self.play(FadeIn(line2, shift=UP * 0.1), run_time=1.5)
        self.wait(3)


class S2_Short(Scene):
    """Embeddings concept — fast. ~18s"""
    def construct(self):
        header = title_text("Words live in space", color=WHITE, scale=0.75)
        header.move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1)

        rng = np.random.default_rng(42)
        cluster1 = VGroup(*[
            Dot(point=[-1.4 + rng.normal(0, 0.15), 0.9 + rng.normal(0, 0.15), 0],
                radius=0.06, color=TEAL).set_opacity(0.7)
            for _ in range(5)
        ])
        lbl1 = Text("water, river, flood...", color=TEAL).scale(0.25).move_to(LEFT * 1.4 + DOWN * 0.1)

        cluster2 = VGroup(*[
            Dot(point=[1.6 + rng.normal(0, 0.15), 0.7 + rng.normal(0, 0.15), 0],
                radius=0.06, color=TEAL).set_opacity(0.7)
            for _ in range(5)
        ])
        lbl2 = Text("king, throne, crown...", color=TEAL).scale(0.25).move_to(RIGHT * 1.6 + DOWN * 0.1)

        self.play(FadeIn(cluster1), FadeIn(lbl1), FadeIn(cluster2), FadeIn(lbl2), run_time=2)
        self.wait(3)

        explain = body_text(
            "Similar meanings cluster together.\n"
            "This works for every language — including Ancient Egyptian.",
            color=WHITE
        ).scale(0.9).move_to(DOWN * 1.5)
        self.play(Write(explain), run_time=4)
        self.wait(4)


class S3_Short(Scene):
    """Alignment — just the cloud merge visual. ~15s"""
    def construct(self):
        data = load_viz_data()
        eg_norm = normalize_points(data["egyptian"], target_range=1.8, center=(-3, 0))
        en_norm = normalize_points(data["english"], target_range=1.8, center=(3, 0))

        eg_cloud = VGroup(*[
            Dot(point=[nx, ny, 0], radius=0.02, color=GOLD).set_opacity(0.5)
            for nx, ny, pt in eg_norm
        ])
        en_cloud = VGroup(*[
            Dot(point=[nx, ny, 0], radius=0.02, color=TEAL).set_opacity(0.5)
            for nx, ny, pt in en_norm
        ])

        label_eg = Text("Egyptian", color=GOLD).scale(0.35).move_to(LEFT * 3 + UP * 2.5)
        label_en = Text("English", color=TEAL).scale(0.35).move_to(RIGHT * 3 + UP * 2.5)

        self.play(FadeIn(label_eg), Create(eg_cloud), FadeIn(label_en), Create(en_cloud), run_time=3)
        self.wait(1)

        explain = body_text("Find the rotation...", color=LAVENDER).scale(1.0).move_to(DOWN * 2.8)
        self.play(FadeIn(explain), run_time=1)
        self.play(
            eg_cloud.animate.shift(RIGHT * 3.2).scale(1.3).rotate(0.25),
            en_cloud.animate.shift(LEFT * 3.2).rotate(-0.1),
            run_time=4, rate_func=smooth
        )
        self.wait(1)

        result = body_text("...and the words align across 4,000 years.", color=WHITE).scale(0.9)
        result.move_to(DOWN * 2.8)
        self.play(FadeOut(explain), FadeIn(result), run_time=1.5)
        self.wait(2)


class SB_Short(Scene):
    """Just the '1 in 3' stat. ~12s"""
    def construct(self):
        big_stat = Text("1 in 3", color=GOLD).scale(1.8)
        sub = body_text("correct translations — no dictionary needed", color=WHITE).scale(0.85)
        group = VGroup(big_stat, sub).arrange(DOWN, buff=0.3).move_to(UP * 0.5)
        self.play(FadeIn(big_stat, scale=0.8), run_time=2)
        self.wait(1)
        self.play(FadeIn(sub), run_time=1.5)
        self.wait(2)

        closing = body_text("Here's what the geometry revealed.", color=LAVENDER).scale(0.9)
        closing.move_to(DOWN * 1.5)
        self.play(FadeIn(closing, shift=UP * 0.1), run_time=1.5)
        self.wait(2)


class D1_Gold_Short(Scene):
    """Gold = Divine — trimmed waits. ~25s"""
    def construct(self):
        glyph = hiero_text("\U000131B4\U00013208\U000130C3\U000130F1", color=GOLD, scale=0.4)
        title = Text("Gold Is Divine Flesh", color=GOLD).scale(0.55)
        header = VGroup(glyph, title).arrange(RIGHT, buff=0.3).move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1)

        dot_gold = Dot(point=[-3, 0.5, 0], radius=0.12, color="#f1c40f").set_opacity(0.9)
        dot_divine = Dot(point=[3, 0.5, 0], radius=0.12, color=LAVENDER).set_opacity(0.9)
        lbl_gold = Text("gold", color="#f1c40f").scale(0.4).next_to(dot_gold, DOWN, buff=0.15)
        lbl_divine = Text("divine", color=LAVENDER).scale(0.4).next_to(dot_divine, DOWN, buff=0.15)
        self.play(FadeIn(dot_gold, lbl_gold), FadeIn(dot_divine, lbl_divine), run_time=1.5)
        self.wait(1)

        dot_ntri = Dot(point=[-2, -1.5, 0], radius=0.1, color=GOLD)
        dot_nbw = Dot(point=[2, -1.5, 0], radius=0.1, color=GOLD)
        glyph_ntri = hiero_text("\U00013291", color=GOLD, scale=0.5)
        lbl_ntri_eng = Text("(divine)", color=GOLD).scale(0.28)
        lbl_ntri = VGroup(glyph_ntri, lbl_ntri_eng).arrange(RIGHT, buff=0.08).next_to(dot_ntri, DOWN, buff=0.15)
        glyph_nbw = hiero_text("\U00013210", color=GOLD, scale=0.5)
        lbl_nbw_eng = Text("(gold)", color=GOLD).scale(0.28)
        lbl_nbw = VGroup(glyph_nbw, lbl_nbw_eng).arrange(RIGHT, buff=0.08).next_to(dot_nbw, DOWN, buff=0.15)

        self.play(FadeIn(dot_ntri, lbl_ntri), FadeIn(dot_nbw, lbl_nbw), run_time=1.5)
        self.wait(1)

        self.play(
            dot_ntri.animate.move_to([0, -1.5, 0]),
            dot_nbw.animate.move_to([0, -1.5, 0]),
            lbl_ntri.animate.move_to([-1.5, -2.2, 0]),
            lbl_nbw.animate.move_to([1.5, -2.2, 0]),
            run_time=2
        )
        glow = Dot(point=[0, -1.5, 0], radius=0.3, color=GOLD).set_opacity(0.3)
        self.play(FadeIn(glow), run_time=0.5)
        self.wait(3)


class D_Montage(Scene):
    """Discovery montage with simple geometry visuals. ~35s"""
    def construct(self):
        header = title_text("What else the geometry reveals", color=GOLD, scale=0.6)
        header.move_to(UP * 3.0)
        self.play(FadeIn(header), run_time=1)

        insights = [
            ("\U000131EF", "Silence = Death", "What the dead lost was not life. It was voice.",
             "converge"),  # two dots converge
            ("\U00013080", "Seeing = Power", "In Egyptian, the eye IS power. Sight is magic.",
             "triangle"),  # triangle of concepts
            ("\U00013196", "Snake = Sacred", "In English, an animal. In Egyptian, the divine.",
             "shift"),  # dot shifts from one cluster to another
            ("\U00013250", "Temple : House :: God : Man", "Vector arithmetic across 4,000 years.",
             "parallel"),  # parallel arrows
            ("\U000130AD", "Mother = Royalty", "Motherhood is a crown, not the earth.",
             "swap"),  # expected fades, actual appears
            ("\U00013302", "Truth = Force", "Truth is not correctness. It is force.",
             "cluster"),  # dots cluster tight
            ("\U0001339B\U000131F3\U0001339B", "Love + Fear = Eternity", "Between love and fear: forever.",
             "midpoint"),  # midpoint between two poles
        ]

        for glyph_char, title_str, punchline_str, viz_type in insights:
            # Title row
            g = hiero_text(glyph_char, color=GOLD, scale=0.7)
            t = Text(title_str, color=WHITE).scale(0.45)
            title_row = VGroup(g, t).arrange(RIGHT, buff=0.2).move_to(UP * 1.5)

            # Simple geometry visualization
            viz = VGroup()
            if viz_type == "converge":
                d1 = Dot([-1.5, 0, 0], radius=0.08, color=LAVENDER)
                d2 = Dot([1.5, 0, 0], radius=0.08, color=SOFT_RED)
                viz = VGroup(d1, d2)
            elif viz_type == "triangle":
                d1 = Dot([-1, 0.5, 0], radius=0.08, color=GOLD)
                d2 = Dot([1, 0.5, 0], radius=0.08, color=LAVENDER)
                d3 = Dot([0, -0.5, 0], radius=0.08, color=TEAL)
                l1 = Line(d1.get_center(), d2.get_center(), color=MUTED, stroke_width=1).set_opacity(0.3)
                l2 = Line(d2.get_center(), d3.get_center(), color=MUTED, stroke_width=1).set_opacity(0.3)
                l3 = Line(d3.get_center(), d1.get_center(), color=MUTED, stroke_width=1).set_opacity(0.3)
                viz = VGroup(d1, d2, d3, l1, l2, l3)
            elif viz_type == "shift":
                d1 = Dot([-1.5, 0, 0], radius=0.08, color=TEAL).set_opacity(0.3)
                d2 = Dot([1.5, 0, 0], radius=0.08, color=GOLD)
                arr = Arrow([-1, 0, 0], [1, 0, 0], color=LAVENDER, stroke_width=2)
                viz = VGroup(d1, d2, arr)
            elif viz_type == "parallel":
                a1 = Arrow([-1.5, -0.3, 0], [-1.5, 0.5, 0], color=LAVENDER, stroke_width=2)
                a2 = Arrow([1.5, -0.3, 0], [1.5, 0.5, 0], color=LAVENDER, stroke_width=2)
                viz = VGroup(a1, a2)
            elif viz_type == "swap":
                d1 = Text("earth?", color=MUTED).scale(0.3).set_opacity(0.3).move_to(LEFT * 1)
                d2 = Text("royalty", color=GOLD).scale(0.35).move_to(RIGHT * 1)
                viz = VGroup(d1, d2)
            elif viz_type == "cluster":
                rng = np.random.default_rng(77)
                dots = VGroup(*[
                    Dot([rng.normal(0, 0.3), rng.normal(0, 0.3), 0], radius=0.06, color=GOLD).set_opacity(0.6)
                    for _ in range(6)
                ])
                viz = dots
            elif viz_type == "midpoint":
                d1 = Dot([-1.5, 0, 0], radius=0.08, color=TEAL)
                d2 = Dot([1.5, 0, 0], radius=0.08, color=SOFT_RED)
                dm = Dot([0, 0, 0], radius=0.1, color=GOLD)
                viz = VGroup(d1, d2, dm)

            viz.move_to(ORIGIN)

            # Punchline
            p = body_text(punchline_str, color=LAVENDER).scale(0.8).move_to(DOWN * 1.5)

            # Animate
            self.play(FadeIn(title_row), FadeIn(viz), run_time=1)
            self.play(FadeIn(p, shift=UP * 0.1), run_time=0.8)
            self.wait(2.5)
            self.play(FadeOut(title_row), FadeOut(viz), FadeOut(p), run_time=0.5)

        self.wait(1)


class S7_Short(Scene):
    """Short conclusion — glyphs return + contextual re-translation only. ~20s"""
    def construct(self):
        glyphs = hiero_text(GLYPH_STRIP, color=GOLD, scale=1.0)
        glyphs.move_to(UP * 2.0)
        self.play(FadeIn(glyphs, shift=UP * 0.2), run_time=2)
        self.wait(1)

        context_header = body_text("A new reading:", color=TEAL).scale(0.9)
        context_header.next_to(glyphs, DOWN, buff=0.4)
        contextual = Text(UTT_213["contextual"], color=WHITE, line_spacing=1.3).scale(0.35)
        contextual.next_to(context_header, DOWN, buff=0.3)

        self.play(FadeIn(context_header), run_time=1)
        self.play(Write(contextual), run_time=4)
        self.wait(5)


class S8_Outro_Short(Scene):
    """Short outro — 10s."""
    def construct(self):
        final = body_text("Translation gave us the words.", color=WHITE).scale(1.3).move_to(UP * 0.5)
        final2 = body_text("The vectors gave us the world between them.", color=LAVENDER).scale(1.2).move_to(DOWN * 0.8)
        self.play(Write(final), run_time=2)
        self.wait(1)
        self.play(FadeIn(final2, shift=UP * 0.1), run_time=2)
        self.wait(2)
        self.play(FadeOut(final), FadeOut(final2), run_time=0.5)
        repo = body_text("github.com/ebrinz/heiroglyphy", color=GOLD).scale(1.1)
        self.play(FadeIn(repo, shift=UP * 0.1), run_time=1)
        self.wait(2)


class HeiroglyphyVideo3Min(Scene):
    """
    3-minute cut for encore.pillar.vc application.
    Run: manim -pqh heiroglyphy_video.py HeiroglyphyVideo3Min
    """
    def construct(self):
        for SceneClass in [
            S0_Title_Short,
            S1_Short,
            S2_Short,
            S3_Short,
            SB_Short,
            D1_Gold_Short,
            D_Montage,
            S7_Short,
            S8_Outro_Short,
        ]:
            SceneClass.construct(self)
            self.clear()
            self.wait(0.3)
