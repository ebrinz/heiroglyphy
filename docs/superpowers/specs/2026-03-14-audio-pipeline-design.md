# Audio Pipeline Design: Voice Narration + Generative Drone Score

**Date:** 2026-03-14
**Status:** Approved
**Video:** `docs/heiroglyphy_video.py` (6 scenes, ~3:15)

## Overview

Add a complete audio layer to the Heiroglyphy Manim explainer video: synthesized voice narration (OpenAI TTS) and a generative ambient drone score with reactive cues at scene transitions and subtitle drops. Audio is generated as standalone files and merged with the rendered video via FFmpeg.

## Architecture: Modular Scripts with Shared Timing Manifest

Three independent scripts share a single timing manifest (`audio_timing.json`). Each can be re-run independently without affecting the others.

```
audio_timing.json  (source of truth)
       |
       ├── generate_voice.py  →  voice/*.wav + voice_full.wav
       ├── generate_drone.py  →  drone/drone_full.wav
       └── mix_audio.py       →  mixed_audio.wav → FFmpeg → final video
```

## 1. Timing Manifest (`audio_timing.json`)

Derived from the Manim scene structure. Maps narration text and audio cues to absolute timestamps.

```json
{
  "total_duration": null,
  "scenes": [
    {
      "id": "S1_Hook",
      "start": 0.0,
      "end": 30.0,
      "narration_segments": [
        {
          "id": "s1_01",
          "text": "These symbols are four thousand years old.",
          "start": 0.0
        },
        {
          "id": "s1_02",
          "text": "Scholars have been translating them for two centuries.",
          "start": 6.0
        },
        {
          "id": "s1_03",
          "text": "But translation is lossy. When you compress a word into a single English equivalent, the web of meaning around it disappears.",
          "start": 12.0
        },
        {
          "id": "s1_04",
          "text": "What if we could get it back?",
          "start": 19.5
        }
      ],
      "cues": [
        {"type": "scene_start", "time": 0.0},
        {"type": "subtitle_drop", "time": 6.0},
        {"type": "subtitle_drop", "time": 12.0},
        {"type": "subtitle_drop", "time": 19.5}
      ]
    },
    {
      "id": "S2_Idea",
      "start": 30.5,
      "end": 65.5,
      "narration_segments": [
        {
          "id": "s2_01",
          "text": "Here's the key insight. If you train a computer on enough text, every word ends up as a point in space.",
          "start": 30.5
        },
        {
          "id": "s2_02",
          "text": "And words with similar meanings cluster together. Water, river, flood, they're neighbors. King, throne, crown, another cluster.",
          "start": 37.0
        },
        {
          "id": "s2_03",
          "text": "This works for every language. Including Ancient Egyptian.",
          "start": 51.5
        }
      ],
      "cues": [
        {"type": "scene_start", "time": 30.5},
        {"type": "subtitle_drop", "time": 37.0},
        {"type": "subtitle_drop", "time": 51.5}
      ]
    },
    {
      "id": "S3_Alignment",
      "start": 66.0,
      "end": 111.0,
      "narration_segments": [
        {
          "id": "s3_01",
          "text": "So we trained embeddings on a hundred thousand Ancient Egyptian sentences. And we trained them separately on English.",
          "start": 66.0
        },
        {
          "id": "s3_02",
          "text": "Both languages form a shape, a cloud of points where similar words cluster. The shapes are similar, but rotated.",
          "start": 73.0
        },
        {
          "id": "s3_03",
          "text": "The challenge is to find that rotation.",
          "start": 82.0
        },
        {
          "id": "s3_04",
          "text": "If you get it right, Egyptian words land next to their English meanings.",
          "start": 90.0
        },
        {
          "id": "s3_05",
          "text": "We got it right thirty-two percent of the time, without ever using a dictionary.",
          "start": 100.0
        }
      ],
      "cues": [
        {"type": "scene_start", "time": 66.0},
        {"type": "subtitle_drop", "time": 73.0},
        {"type": "subtitle_drop", "time": 82.0},
        {"type": "subtitle_drop", "time": 90.0},
        {"type": "subtitle_drop", "time": 100.0}
      ]
    },
    {
      "id": "S4_Journey",
      "start": 111.5,
      "end": 136.5,
      "narration_segments": [
        {
          "id": "s4_01",
          "text": "It took fifteen attempts to get there. We tried neural networks, they failed.",
          "start": 111.5
        },
        {
          "id": "s4_02",
          "text": "We tried the latest language models, they failed spectacularly.",
          "start": 118.0
        },
        {
          "id": "s4_03",
          "text": "In the end, simple linear algebra outperformed everything. Sometimes the best tool is the oldest one.",
          "start": 124.0
        }
      ],
      "cues": [
        {"type": "scene_start", "time": 111.5},
        {"type": "subtitle_drop", "time": 118.0},
        {"type": "subtitle_drop", "time": 124.0}
      ]
    },
    {
      "id": "S5_Discoveries",
      "start": 137.0,
      "end": 192.0,
      "narration_segments": [
        {
          "id": "s5_01",
          "text": "Here's what the geometry revealed. The midpoint of gold and divine in English maps to the same region of the Egyptian space. This isn't metaphor. The texts don't distinguish them. Gold is divinity.",
          "start": 140.0
        },
        {
          "id": "s5_02",
          "text": "The midpoint of silence and death, every single result is a variant of to die. The Egyptians called the necropolis the silent land. What the dead lost was not life. It was voice.",
          "start": 151.0
        },
        {
          "id": "s5_03",
          "text": "The Eye of Horus sits between knowledge and spellcasting. Seeing was not observation. It was an act of magical power.",
          "start": 163.0
        },
        {
          "id": "s5_04",
          "text": "In Greek tradition, the snake means wisdom. In Egyptian vectors, it means the gods. Two cultures, separated by geometry.",
          "start": 173.0
        },
        {
          "id": "s5_05",
          "text": "Translation gave us the words. The vectors gave us the world between them.",
          "start": 183.0
        }
      ],
      "cues": [
        {"type": "scene_start", "time": 137.0},
        {"type": "discovery_reveal", "time": 137.0},
        {"type": "discovery_reveal", "time": 151.0},
        {"type": "discovery_reveal", "time": 163.0},
        {"type": "discovery_reveal", "time": 173.0},
        {"type": "subtitle_drop", "time": 183.0}
      ]
    },
    {
      "id": "S6_Close",
      "start": 192.5,
      "end": 200.0,
      "narration_segments": [],
      "cues": [
        {"type": "scene_start", "time": 192.5}
      ]
    }
  ]
}
```

**`total_duration`** is set to `null` in the manifest. At mix time, `mix_audio.py` probes the actual video duration via `ffprobe` and uses that as the authoritative length. This avoids drift between estimated and actual scene timing.

**S5 narration flows continuously over visuals.** The discovery narration segments are not constrained to individual discovery hold windows — they flow as continuous speech while the visuals change underneath. Segments may overlap visual transitions. The start timestamps are approximate anchors; `generate_voice.py` will measure actual TTS output durations and warn if any segment would overlap the next.

Timestamps are approximate and will be refined against the actual rendered video duration. The manifest is manually authored (not auto-extracted from Manim) since Manim doesn't expose absolute timestamps across scenes.

## 2. Voice Generation (`generate_voice.py`)

**API:** OpenAI TTS (`tts-1-hd` model, `echo` voice)

**Process:**
1. Read `audio_timing.json`
2. For each narration segment, check if `voice/{id}.wav` already exists (cache hit → skip API call)
3. Call OpenAI TTS API with `response_format="wav"`, retry up to 3 times with exponential backoff on transient errors
4. Save individual WAVs to `docs/audio/voice/voice_{id}.wav`
5. Measure each segment's actual duration; warn if it would overlap the next segment's start time
6. If overlap detected: re-generate with `speed=1.1` (up to `1.25` max); if still overlapping, truncate with 50ms fade-out at the boundary
7. Resample all segments from TTS native rate (24 kHz) to 44100 Hz during assembly
8. Assemble `voice_full.wav` (44100 Hz, 16-bit mono) by placing each segment at its `start` timestamp, padding with silence between segments

**Design decisions:**
- `tts-1-hd` for quality (documentary-grade audio for a 3-minute video is worth the marginal cost difference)
- `echo` voice — warm storytelling tone suited to reverent, wonder-driven narration
- `response_format="wav"` — returns WAV with headers; avoids mp3 decode dependency
- Speed 1.0 default — the `speed` parameter (0.25-4.0) is available as a tuning knob if segments don't fit their windows
- Per-segment caching enables selective re-generation without redundant API calls
- Resampling to 44100 Hz at assembly time ensures consistent sample rate with the drone track

## 3. Drone Score Generation (`generate_drone.py`)

All synthesis via numpy/scipy. No external audio libraries.

### Base Layer (continuous, full duration)

| Component | Parameters |
|---|---|
| Fundamental drone | ~65 Hz (low C), sine wave with soft saturation |
| Harmonics | 2nd (130 Hz, -6dB), 3rd (195 Hz, -12dB), 5th (325 Hz, -18dB) |
| Amplitude LFO | ~0.08 Hz sine, depth ~15% — slow breathing motion |
| Noise bed | Pink noise (Voss-McCartney algorithm), bandpass 100-800 Hz, -24dB below fundamental |
| Stereo width | L/R detuned by ~0.5 Hz for gentle phasing |

### Reactive Cue Layers

Triggered by cue entries in the timing manifest:

**`scene_start` — Crystalline shimmer swell:**
- Cluster of 4-6 sine partials in 2-4 kHz range, randomized within ±50 Hz
- Attack: ~2s exponential fade-in
- Decay: ~2s exponential fade-out
- Simple convolution reverb (synthetic IR: exponential decay noise burst, ~1.5s RT60). Trim convolution output to original signal length + 0.5s tail, then apply fade-out window to the tail
- Level: -12dB below drone fundamental at peak

**`subtitle_drop` — Soft tonal ping:**
- Single sine partial at ~1 kHz (tuned to a harmonic of the drone fundamental)
- Attack: 10ms
- Decay: ~1.5s exponential
- Level: -18dB below drone — barely perceptible, felt more than heard

**`discovery_reveal` — Rich harmonic swell (S5 only):**
- Current drone fundamental + perfect fifth above (fundamental × 1.5). The fifth is always relative to the current shifted fundamental
- Additional shimmer partials (3-5 kHz range)
- Attack: ~1.5s, sustain: ~1s, decay: ~1.5s
- Each successive discovery shifts the drone fundamental up by a minor third (65 → 77 → 92 → 109 Hz), and the fifth shifts with it (97.5 → 115.5 → 138 → 163.5 Hz) — builds momentum
- Level: -6dB below drone at peak — these are meant to be noticed

### Scene-Level Shaping

| Scene | Drone behavior |
|---|---|
| S1 Hook | Fade in from silence over first 3s |
| S2 Idea | Steady state, warm |
| S3 Alignment | Subtle pitch bend (+~5 Hz) during cloud rotation (~82-90s), tension |
| S4 Journey | Slightly reduced level during bar chart (voice-heavy) |
| S5 Discoveries | Ascending key shifts per discovery (minor thirds) |
| S6 Close | Fade to silence over final 5s |

**Inter-scene gaps:** The drone sustains continuously through the 0.5s gaps between scenes. No crossfade or interruption — the drone is oblivious to scene boundaries except for the explicit scene-level shaping above.

### Output
- Sample rate: 44100 Hz, 16-bit stereo
- File: `docs/audio/drone/drone_full.wav`

## 4. Mix & Merge (`mix_audio.py`)

**Mixing:**
1. Probe video duration via `ffprobe` — this is the authoritative total duration
2. Load `voice_full.wav` and `drone_full.wav`, pad/trim both to match video duration
3. Apply voice-activated ducking to drone:
   - Compute voice envelope (RMS with ~100ms window)
   - When voice is active (envelope > -40 dBFS threshold), attenuate drone by ~3dB
   - Smooth the ducking with ~200ms attack/release to avoid pumping
4. Mix: voice at 0dB, drone at -18dB baseline (further ducked when voice active)
5. Normalize final mix to -1dB peak
6. Output: `docs/audio/mixed_audio.wav`

**Video merge:**
```bash
ffmpeg -i docs/media/videos/heiroglyphy_video/1080p60/HeiroglyphyVideo.mp4 \
       -i docs/audio/mixed_audio.wav \
       -c:v copy -c:a aac -b:a 192k -shortest \
       docs/media/HeiroglyphyVideo_final.mp4
```

## 5. File Structure

```
docs/audio/
├── audio_timing.json          # timing manifest (source of truth)
├── generate_voice.py          # OpenAI TTS pipeline
├── generate_drone.py          # numpy/scipy drone synthesizer
├── mix_audio.py               # mix + FFmpeg merge
├── voice/                     # per-segment WAVs
│   ├── voice_s1_01.wav
│   ├── voice_s1_02.wav
│   └── ...
├── drone/
│   └── drone_full.wav
├── voice_full.wav             # assembled narration track
└── mixed_audio.wav            # final mixed audio
```

## 6. Dependencies

| Package | Purpose | Already installed? |
|---|---|---|
| `openai` | TTS API calls | Likely (check) |
| `numpy` | Drone synthesis, audio math | Yes (used by embeddings) |
| `scipy` | WAV I/O, signal processing | Yes (used by embeddings) |
| `ffmpeg` | Video/audio merge (system binary) | Check |

No new Python packages required beyond `openai`.

## 7. Workflow

```bash
# 1. Generate voice narration
python docs/audio/generate_voice.py

# 2. Generate drone score
python docs/audio/generate_drone.py

# 3. Mix and merge with video
python docs/audio/mix_audio.py
```

Each step is independent after step 1 produces voice files (step 3 needs both outputs). Steps 1 and 2 can run in parallel.
