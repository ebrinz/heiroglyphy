# Generation Pipelines

The project includes pipelines for generating a video explainer, audio narration, and PDF publications. Each has its own dependencies beyond the core research requirements.

## System Dependencies

These must be installed before running any generation pipeline:

| Tool | Purpose | Install |
|------|---------|---------|
| **ffmpeg** / **ffprobe** | Audio/video merging | `brew install ffmpeg` (macOS) or [ffmpeg.org](https://ffmpeg.org/download.html) |
| **XeLaTeX** | PDF compilation with Unicode/custom fonts | `brew install --cask mactex` (macOS) or install [TeX Live](https://tug.org/texlive/) |
| **Manim** (Community Edition) | 3Blue1Brown-style video rendering | `pip install manim` ([docs](https://docs.manim.community/)) |

## Python Dependencies

Install all generation dependencies on top of core requirements:

```bash
pip install -r requirements.txt
pip install manim manimpango openai
```

> **Note:** `openai` is only needed for TTS voice generation and requires an `OPENAI_API_KEY` environment variable.

---

## Pipeline Overview

```
generate_viz_data.py          # 1. Project embeddings onto semantic axes
        ↓
heiroglyphy_video.py          # 2. Render 14 animated scenes with Manim
        ↓
audio/generate_voice.py       # 3a. Generate TTS narration (OpenAI)
audio/generate_drone.py       # 3b. Synthesize ambient drone score
audio/mix_audio.py            # 3c. Duck, normalize, merge with video
        ↓
HeiroglyphyVideo_final.mp4   # 4. Final output
```

---

## Stage 1: Visualization Data

**Script:** `generate_viz_data.py`

Projects the aligned Egyptian and English word vectors onto two interpretable semantic axes:

- **X-axis:** mortal <-> divine (god - man)
- **Y-axis:** death <-> life (life - death, orthogonalized via Gram-Schmidt)

Outputs `viz_data.json` containing ~200 Egyptian points, 279 English concept points, anchor connections, and highlighted "golden hit" pairs (e.g. mw -> water, wsjr -> god).

**Requires:** `final_output/egyptian_aligned_vectors.npz`, `final_output/concept_vectors.npz`

## Stage 2: Manim Rendering

**Script:** `heiroglyphy_video.py`

14 scene classes composed into a single video:

| Scene | Content |
|-------|---------|
| S1_Hook | Hieroglyph strip, problem statement |
| S2_Idea | Word embeddings as clustered points |
| S3_Alignment | Two embedding clouds rotate and align |
| S4_Journey | Bar chart of V3 -> V15 progression |
| D1-D8 | 8 discovery scenes (Gold, Silence, Seeing, Snake, Temple, Mother, Truth, Eternity) |
| S6_Discussion | Caveats and limitations |
| S7_Conclusion | Thesis recap, GitHub link |

Uses `EgyptianHiero.ttf` for hieroglyphic Unicode rendering via `manimpango`. Styled in a 3Blue1Brown dark theme.

**Two output versions:**
- `HeiroglyphyVideo` - full (~6:20)
- `HeiroglyphyVideo3Min` - condensed (~2:53)

```bash
cd docs
manim -pqh heiroglyphy_video.py HeiroglyphyVideo
```

## Stage 3: Audio

### 3a. Voice Narration

**Script:** `audio/generate_voice.py`

Reads `audio/audio_timing.json` for narration text and timing windows, calls OpenAI TTS (model: `tts-1-hd`, voice: `echo`), and assembles segments into `audio/voice_full.wav`. Auto-adjusts playback speed (up to 1.25x) if a segment doesn't fit its timing window. Individual segments are cached in `audio/voice/`.

**Requires:** `OPENAI_API_KEY` environment variable

### 3b. Drone Score

**Script:** `audio/generate_drone.py`

Pure DSP synthesis with no external samples:

1. Pink noise base (Voss-McCartney algorithm)
2. Sine-wave cues at scene transitions (220 Hz) and discovery reveals (330 Hz)
3. Convolution reverb (synthetic IR, RT60 = 1.5s)
4. Low-pass Butterworth filter (~400 Hz cutoff)

Output: `audio/drone/drone_full.wav`

### 3c. Mixing & Merge

**Script:** `audio/mix_audio.py`

1. Voice-activated ducking: drone attenuates ~18 dB when narration is active
2. Peak normalization to -1 dB
3. Merges audio with video via ffmpeg

Output: `HeiroglyphyVideo_final.mp4`

---

## 📄 PDF / LaTeX

Two LaTeX documents live in `paper/`:

| Document | Source | Output |
|----------|--------|--------|
| Main paper | `heiroglyphy.tex` | `heiroglyphy.pdf` |
| Supplementary appendix | `appendix_insights.tex` | `appendix_insights.pdf` |

Both use the **EgyptianHiero.ttf** font from `final_output/` for hieroglyphic rendering and require **XeLaTeX** (not pdflatex) for Unicode support.

**Compile:**

```bash
cd docs/paper

# Main paper (run twice for references/TOC)
xelatex heiroglyphy.tex
xelatex heiroglyphy.tex

# Supplementary appendix
xelatex appendix_insights.tex
xelatex appendix_insights.tex
```

**Required LaTeX packages:** `fontspec`, `amsmath`, `amssymb`, `graphicx`, `booktabs`, `hyperref`, `geometry`, `natbib`, `xcolor`, `enumitem`, `float`, `multicol`, `titlesec`, `fancyhdr`, `framed`. These ship with a standard TeX Live or MacTeX installation.

---

## Timing Configuration

`audio/audio_timing.json` is the single source of truth for narration timing, scene boundaries, and audio cues. A separate `audio/audio_timing_3min.json` drives the condensed version.

## 🔄 Full Regeneration

To regenerate everything from scratch:

```bash
# 1. Visualization data (requires final_output/ vectors)
python docs/generate_viz_data.py

# 2. Video (requires Manim)
cd docs && manim -pqh heiroglyphy_video.py HeiroglyphyVideo && cd ..

# 3. Audio (requires OpenAI key + ffmpeg)
cd docs/audio
export OPENAI_API_KEY="your-key-here"
python generate_voice.py
python generate_drone.py
python mix_audio.py
cd ../..

# 4. PDFs (requires XeLaTeX)
cd docs/paper
xelatex heiroglyphy.tex && xelatex heiroglyphy.tex
xelatex appendix_insights.tex && xelatex appendix_insights.tex
cd ../..
```

## Output Directory Structure

```
docs/
├── media/
│   ├── videos/heiroglyphy_video/1080p60/
│   │   └── HeiroglyphyVideo.mp4        # Raw video (no audio)
│   └── HeiroglyphyVideo_final.mp4      # Final video with audio
├── audio/
│   ├── voice/                           # Cached TTS segments
│   ├── drone/
│   │   └── drone_full.wav              # Synthesized drone track
│   ├── voice_full.wav                   # Assembled voice track
│   └── mixed_audio.wav                  # Final mixed audio
└── paper/
    ├── heiroglyphy.pdf                  # Main paper
    └── appendix_insights.pdf            # Supplementary appendix
```

## Dependencies Summary

- **Python:** numpy, scipy, manim, manimpango, openai, gensim, scikit-learn
- **System:** ffmpeg, ffprobe, XeLaTeX (for PDFs)
- **Assets:** `final_output/EgyptianHiero.ttf`, aligned embedding files in `final_output/`
