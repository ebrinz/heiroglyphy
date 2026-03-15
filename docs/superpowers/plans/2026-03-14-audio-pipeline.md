# Audio Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add voice narration (OpenAI TTS) and generative ambient drone scoring to the Heiroglyphy Manim explainer video.

**Architecture:** Three independent Python scripts share a JSON timing manifest. `generate_voice.py` calls OpenAI TTS per narration segment and assembles a full voice track. `generate_drone.py` synthesizes an ambient drone score with reactive cues using numpy/scipy. `mix_audio.py` layers both tracks with ducking and merges with the rendered video via FFmpeg.

**Tech Stack:** Python 3, OpenAI API (`tts-1-hd`), numpy, scipy, FFmpeg

**Spec:** `docs/superpowers/specs/2026-03-14-audio-pipeline-design.md`

---

## Chunk 1: Project Setup + Timing Manifest

### Task 1: Create directory structure and timing manifest

**Files:**
- Create: `docs/audio/audio_timing.json`
- Create: `docs/audio/voice/` (directory)
- Create: `docs/audio/drone/` (directory)

- [ ] **Step 1: Create audio directories**

```bash
mkdir -p docs/audio/voice docs/audio/drone
```

- [ ] **Step 2: Write the timing manifest**

Create `docs/audio/audio_timing.json` with the complete timing data from the spec. This is the source of truth for all three scripts.

```json
{
  "total_duration": null,
  "tts_model": "tts-1-hd",
  "tts_voice": "echo",
  "scenes": [
    {
      "id": "S1_Hook",
      "start": 0.0,
      "end": 30.0,
      "narration_segments": [
        {"id": "s1_01", "text": "These symbols are four thousand years old.", "start": 0.0},
        {"id": "s1_02", "text": "Scholars have been translating them for two centuries.", "start": 6.0},
        {"id": "s1_03", "text": "But translation is lossy. When you compress a word into a single English equivalent, the web of meaning around it disappears.", "start": 12.0},
        {"id": "s1_04", "text": "What if we could get it back?", "start": 19.5}
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
        {"id": "s2_01", "text": "Here's the key insight. If you train a computer on enough text, every word ends up as a point in space.", "start": 30.5},
        {"id": "s2_02", "text": "And words with similar meanings cluster together. Water, river, flood, they're neighbors. King, throne, crown, another cluster.", "start": 37.0},
        {"id": "s2_03", "text": "This works for every language. Including Ancient Egyptian.", "start": 51.5}
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
        {"id": "s3_01", "text": "So we trained embeddings on a hundred thousand Ancient Egyptian sentences. And we trained them separately on English.", "start": 66.0},
        {"id": "s3_02", "text": "Both languages form a shape, a cloud of points where similar words cluster. The shapes are similar, but rotated.", "start": 73.0},
        {"id": "s3_03", "text": "The challenge is to find that rotation.", "start": 82.0},
        {"id": "s3_04", "text": "If you get it right, Egyptian words land next to their English meanings.", "start": 90.0},
        {"id": "s3_05", "text": "We got it right thirty-two percent of the time, without ever using a dictionary.", "start": 100.0}
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
        {"id": "s4_01", "text": "It took fifteen attempts to get there. We tried neural networks, they failed.", "start": 111.5},
        {"id": "s4_02", "text": "We tried the latest language models, they failed spectacularly.", "start": 118.0},
        {"id": "s4_03", "text": "In the end, simple linear algebra outperformed everything. Sometimes the best tool is the oldest one.", "start": 124.0}
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
        {"id": "s5_01", "text": "Here's what the geometry revealed. The midpoint of gold and divine in English maps to the same region of the Egyptian space. This isn't metaphor. The texts don't distinguish them. Gold is divinity.", "start": 140.0},
        {"id": "s5_02", "text": "The midpoint of silence and death, every single result is a variant of to die. The Egyptians called the necropolis the silent land. What the dead lost was not life. It was voice.", "start": 151.0},
        {"id": "s5_03", "text": "The Eye of Horus sits between knowledge and spellcasting. Seeing was not observation. It was an act of magical power.", "start": 163.0},
        {"id": "s5_04", "text": "In Greek tradition, the snake means wisdom. In Egyptian vectors, it means the gods. Two cultures, separated by geometry.", "start": 173.0},
        {"id": "s5_05", "text": "Translation gave us the words. The vectors gave us the world between them.", "start": 183.0}
      ],
      "cues": [
        {"type": "scene_start", "time": 137.0},
        {"type": "discovery_reveal", "time": 140.0},
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

Note: The S5 `discovery_reveal` cue at 137.0 from the spec has been moved to 140.0 to align with the first narration segment start — the cue should fire when the discovery content appears, not during the header animation.

- [ ] **Step 3: Verify dependencies are available**

```bash
python -c "import openai; print('openai', openai.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import scipy; print('scipy', scipy.__version__)"
ffmpeg -version | head -1
ffprobe -version | head -1
```

If `openai` is missing: `pip install openai`
If `ffmpeg` is missing: `brew install ffmpeg`

- [ ] **Step 4: Commit**

```bash
git add docs/audio/audio_timing.json
git commit -m "feat: add audio timing manifest for voice + drone pipeline"
```

---

## Chunk 2: Voice Generation

### Task 2: Create `generate_voice.py`

**Files:**
- Create: `docs/audio/generate_voice.py`

**Context:** This script reads `audio_timing.json`, calls OpenAI TTS for each narration segment (with caching), resamples from 24kHz to 44100Hz, and assembles all segments into a single `voice_full.wav` placed at the correct timestamps. It handles overlap detection and speed adjustment.

- [ ] **Step 1: Write `generate_voice.py`**

Create `docs/audio/generate_voice.py`:

```python
"""
Voice narration generator for Heiroglyphy video.

Reads audio_timing.json, calls OpenAI TTS (tts-1-hd, echo voice)
for each narration segment, and assembles voice_full.wav.

Usage:
    cd docs/audio
    python generate_voice.py

Requires: OPENAI_API_KEY environment variable
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from openai import OpenAI

AUDIO_DIR = Path(__file__).resolve().parent
TIMING_FILE = AUDIO_DIR / "audio_timing.json"
VOICE_DIR = AUDIO_DIR / "voice"
OUTPUT_FILE = AUDIO_DIR / "voice_full.wav"

TARGET_SR = 44100
TTS_SR = 24000  # OpenAI TTS native sample rate for WAV format


def load_timing():
    with open(TIMING_FILE) as f:
        return json.load(f)


def generate_segment(client, segment_id, text, model, voice, speed=1.0):
    """Call OpenAI TTS API with retry logic. Returns path to cached WAV."""
    out_path = VOICE_DIR / f"voice_{segment_id}.wav"
    if out_path.exists():
        print(f"  [cached] {segment_id}")
        return out_path

    print(f"  [generating] {segment_id} (speed={speed})")
    for attempt in range(3):
        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="wav",
                speed=speed,
            )
            response.write_to_file(str(out_path))
            return out_path
        except Exception as e:
            if attempt < 2:
                wait_time = 2 ** (attempt + 1)
                print(f"  [retry] {segment_id} in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                raise


def read_wav_as_float(path):
    """Read WAV file, return (sample_rate, float64 array normalized to [-1, 1])."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data


def resample_to_target(data, src_sr, target_sr):
    """Resample audio using polyphase filtering."""
    if src_sr == target_sr:
        return data
    from math import gcd
    g = gcd(target_sr, src_sr)
    up = target_sr // g
    down = src_sr // g
    return resample_poly(data, up, down)


def get_segment_duration(path):
    """Get duration in seconds of a WAV file."""
    sr, data = read_wav_as_float(path)
    return len(data) / sr


def main():
    timing = load_timing()
    model = timing.get("tts_model", "tts-1-hd")
    voice = timing.get("tts_voice", "echo")

    VOICE_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    # Collect all segments in order with their start times
    all_segments = []
    for scene in timing["scenes"]:
        for seg in scene["narration_segments"]:
            all_segments.append(seg)

    if not all_segments:
        print("No narration segments found.")
        return

    # Phase 1: Generate all TTS segments
    print("Phase 1: Generating TTS audio...")
    for seg in all_segments:
        generate_segment(client, seg["id"], seg["text"], model, voice)

    # Phase 2: Check for overlaps and re-generate if needed
    print("\nPhase 2: Checking segment durations...")
    for i, seg in enumerate(all_segments):
        path = VOICE_DIR / f"voice_{seg['id']}.wav"
        duration = get_segment_duration(path)
        seg["_duration"] = duration
        seg["_path"] = path

        if i < len(all_segments) - 1:
            next_start = all_segments[i + 1]["start"]
            available = next_start - seg["start"]
            if duration > available:
                print(f"  [overlap] {seg['id']}: {duration:.1f}s audio in {available:.1f}s window")
                # Try speed-up
                for speed in [1.1, 1.15, 1.2, 1.25]:
                    # Delete cached file to force re-generation
                    path.unlink(missing_ok=True)
                    generate_segment(client, seg["id"], seg["text"], model, voice, speed=speed)
                    duration = get_segment_duration(path)
                    seg["_duration"] = duration
                    if duration <= available:
                        print(f"  [fixed] {seg['id']}: {duration:.1f}s at speed={speed}")
                        break
                else:
                    print(f"  [truncating] {seg['id']}: will trim to {available:.1f}s")
                    seg["_truncate_to"] = available
            else:
                print(f"  [ok] {seg['id']}: {duration:.1f}s in {available:.1f}s window")
        else:
            print(f"  [ok] {seg['id']}: {duration:.1f}s (last segment)")

    # Phase 3: Assemble voice_full.wav
    print("\nPhase 3: Assembling voice_full.wav...")

    # Calculate total duration needed (last segment end)
    last_seg = all_segments[-1]
    total_duration = last_seg["start"] + last_seg["_duration"] + 1.0  # 1s padding
    total_samples = int(total_duration * TARGET_SR)
    output = np.zeros(total_samples, dtype=np.float64)

    for seg in all_segments:
        sr, data = read_wav_as_float(seg["_path"])
        data = resample_to_target(data, sr, TARGET_SR)

        # Truncate if needed
        if "_truncate_to" in seg:
            max_samples = int(seg["_truncate_to"] * TARGET_SR)
            if len(data) > max_samples:
                # Apply 50ms fade-out at truncation point
                fade_samples = int(0.05 * TARGET_SR)
                fade_start = max_samples - fade_samples
                data[fade_start:max_samples] *= np.linspace(1.0, 0.0, fade_samples)
                data = data[:max_samples]

        start_sample = int(seg["start"] * TARGET_SR)
        end_sample = start_sample + len(data)

        # Extend output array if needed
        if end_sample > len(output):
            output = np.pad(output, (0, end_sample - len(output)))

        output[start_sample:end_sample] += data

    # Write output as 16-bit WAV
    output_16 = np.clip(output * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(OUTPUT_FILE), TARGET_SR, output_16)

    print(f"\nDone! Voice track: {OUTPUT_FILE}")
    print(f"  Duration: {len(output) / TARGET_SR:.1f}s")
    print(f"  Sample rate: {TARGET_SR} Hz")
    print(f"  Segments: {len(all_segments)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `generate_voice.py` and verify output**

```bash
cd /Users/crashy/Development/heiroglyphy
python docs/audio/generate_voice.py
```

Expected output:
- Individual WAV files in `docs/audio/voice/` (one per narration segment, ~20 files)
- `docs/audio/voice_full.wav` (~200s of audio at 44100 Hz)
- Console output showing generation progress, overlap checks, and final stats

Verify:
```bash
python -c "
from scipy.io import wavfile
sr, data = wavfile.read('docs/audio/voice_full.wav')
print(f'Sample rate: {sr}')
print(f'Duration: {len(data)/sr:.1f}s')
print(f'Shape: {data.shape}')
print(f'Dtype: {data.dtype}')
"
```

Expected: Sample rate 44100, duration ~190s, shape (N,), dtype int16.

Listen to a few segments to confirm voice quality and pacing are good.

- [ ] **Step 3: Commit**

```bash
git add docs/audio/generate_voice.py
git commit -m "feat: add OpenAI TTS voice generation with overlap detection"
```

Note: Do NOT commit the generated WAV files — they are large binary outputs. Add to `.gitignore` if not already excluded.

---

## Chunk 3: Drone Score Generation

### Task 3: Create `generate_drone.py`

**Files:**
- Create: `docs/audio/generate_drone.py`

**Context:** This script reads `audio_timing.json` and synthesizes a complete ambient drone score using numpy/scipy. It produces:
- A warm base drone layer (65 Hz fundamental + harmonics + pink noise + LFO)
- Reactive cue layers triggered by timing manifest events (shimmer swells, tonal pings, discovery reveals)
- Scene-level shaping (fade in/out, pitch bends, key shifts)
- Output: stereo 44100 Hz WAV

- [ ] **Step 1: Write `generate_drone.py`**

Create `docs/audio/generate_drone.py`:

```python
"""
Generative drone score for Heiroglyphy video.

Reads audio_timing.json and synthesizes an ambient drone with reactive
audio cues at scene transitions, subtitle drops, and discovery reveals.

Usage:
    cd docs/audio
    python generate_drone.py

No external dependencies beyond numpy/scipy.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve

AUDIO_DIR = Path(__file__).resolve().parent
TIMING_FILE = AUDIO_DIR / "audio_timing.json"
DRONE_DIR = AUDIO_DIR / "drone"
OUTPUT_FILE = DRONE_DIR / "drone_full.wav"

SR = 44100
DURATION_FALLBACK = 200.0  # used if total_duration is null


def load_timing():
    with open(TIMING_FILE) as f:
        return json.load(f)


def db_to_amp(db):
    """Convert decibels to amplitude multiplier."""
    return 10.0 ** (db / 20.0)


# ── Pink Noise (Voss-McCartney algorithm) ─────────────────────────────────────

def pink_noise(n_samples, rng):
    """Generate pink noise using vectorized Voss-McCartney algorithm."""
    n_rows = 16
    array = rng.standard_normal((n_rows, n_samples))
    # Each row updates at a different rate (vectorized: repeat values)
    for i in range(1, n_rows):
        step = 2 ** i
        # Number of unique random values needed
        n_unique = (n_samples + step - 1) // step
        values = rng.standard_normal(n_unique)
        # Repeat each value 'step' times, then trim to n_samples
        array[i, :] = np.repeat(values, step)[:n_samples]
    result = array.sum(axis=0)
    result /= np.max(np.abs(result)) + 1e-10
    return result


# ── Synthesis primitives ──────────────────────────────────────────────────────

def sine_wave(freq, duration, sr=SR, phase=0.0):
    """Generate a sine wave."""
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t + phase)


def apply_envelope(signal, attack_s, decay_s, sustain_s=0.0, sr=SR):
    """Apply an attack-sustain-decay envelope to a signal."""
    n = len(signal)
    envelope = np.ones(n)
    attack_n = int(attack_s * sr)
    decay_n = int(decay_s * sr)
    sustain_n = int(sustain_s * sr)

    # Attack
    if attack_n > 0:
        envelope[:attack_n] = np.linspace(0, 1, attack_n) ** 2  # exponential-ish
    # Decay starts after attack + sustain
    decay_start = attack_n + sustain_n
    if decay_start < n and decay_n > 0:
        decay_end = min(decay_start + decay_n, n)
        actual_decay = decay_end - decay_start
        envelope[decay_start:decay_end] = np.linspace(1, 0, actual_decay) ** 2
        envelope[decay_end:] = 0.0

    return signal * envelope


def synthetic_reverb_ir(rt60=1.5, sr=SR, rng=None):
    """Generate a synthetic impulse response (exponential decay noise)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_samples = int(rt60 * 2 * sr)  # 2x RT60 for full tail
    noise = rng.standard_normal(n_samples)
    decay = np.exp(-3.0 * np.arange(n_samples) / (rt60 * sr))
    return noise * decay


def apply_reverb(signal, ir):
    """Apply convolution reverb. Trim to signal length + 0.5s tail with fade-out."""
    wet = fftconvolve(signal, ir, mode="full")
    # Trim to original length + 0.5s tail
    tail_samples = int(0.5 * SR)
    trim_len = len(signal) + tail_samples
    wet = wet[:trim_len]
    # Fade out the tail
    if tail_samples > 0 and len(wet) > len(signal):
        fade_region = wet[len(signal):]
        fade_region *= np.linspace(1.0, 0.0, len(fade_region))
    # Normalize
    peak = np.max(np.abs(wet)) + 1e-10
    wet /= peak
    return wet


def bandpass(signal, low, high, sr=SR, order=4):
    """Apply a bandpass filter."""
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, signal)


# ── Base drone layer ──────────────────────────────────────────────────────────

def generate_base_drone(total_samples, rng):
    """
    Base drone: 65 Hz fundamental + harmonics + pink noise + LFO.
    Returns stereo array (N, 2).
    """
    t = np.arange(total_samples) / SR
    fundamental = 65.0

    # Fundamental with soft saturation (tanh)
    drone = np.tanh(1.5 * sine_wave(fundamental, total_samples / SR))

    # Harmonics
    drone += db_to_amp(-6) * sine_wave(fundamental * 2, total_samples / SR)   # 2nd
    drone += db_to_amp(-12) * sine_wave(fundamental * 3, total_samples / SR)  # 3rd
    drone += db_to_amp(-18) * sine_wave(fundamental * 5, total_samples / SR)  # 5th

    # Amplitude LFO (slow breathing)
    lfo = 1.0 + 0.15 * sine_wave(0.08, total_samples / SR)
    drone *= lfo

    # Pink noise bed
    noise = pink_noise(total_samples, rng)
    noise = bandpass(noise, 100, 800, SR)
    noise *= db_to_amp(-24)
    drone += noise

    # Normalize drone
    drone /= np.max(np.abs(drone)) + 1e-10

    # Stereo: L/R detuned by 0.5 Hz for gentle phasing
    left = drone
    right_detune = np.tanh(1.5 * sine_wave(fundamental + 0.5, total_samples / SR))
    right_detune += db_to_amp(-6) * sine_wave(fundamental * 2 + 0.5, total_samples / SR)
    right_detune += db_to_amp(-12) * sine_wave(fundamental * 3 + 0.5, total_samples / SR)
    right_detune += db_to_amp(-18) * sine_wave(fundamental * 5 + 0.5, total_samples / SR)
    right_detune *= lfo
    right_detune += noise
    right_detune /= np.max(np.abs(right_detune)) + 1e-10
    right = right_detune

    stereo = np.column_stack([left, right])
    return stereo


# ── Cue layers ────────────────────────────────────────────────────────────────

def generate_shimmer_swell(rng):
    """
    scene_start cue: crystalline shimmer swell.
    Cluster of sine partials in 2-4 kHz, with reverb.
    Returns stereo array (~4.5s).
    """
    duration = 4.0
    n_partials = rng.integers(4, 7)
    signal = np.zeros(int(duration * SR))
    for _ in range(n_partials):
        freq = rng.uniform(2000, 4000)
        partial = sine_wave(freq, duration)
        signal += partial / n_partials

    # Apply attack/decay envelope
    signal = apply_envelope(signal, attack_s=2.0, decay_s=2.0)

    # Apply reverb
    ir = synthetic_reverb_ir(rt60=1.5, rng=rng)
    signal = apply_reverb(signal, ir)

    # Level: -12dB
    signal *= db_to_amp(-12)

    # Make stereo with slight spread
    left = signal
    right = np.roll(signal, int(0.002 * SR))  # 2ms delay for width
    return np.column_stack([left, right])


def generate_tonal_ping(fundamental=65.0, rng=None):
    """
    subtitle_drop cue: soft tonal ping.
    Single sine partial at ~1 kHz harmonic of fundamental.
    Returns stereo array (~1.5s).
    """
    # Find nearest harmonic of fundamental to 1 kHz
    harmonic_num = round(1000 / fundamental)
    freq = fundamental * harmonic_num
    duration = 1.5
    signal = sine_wave(freq, duration)
    signal = apply_envelope(signal, attack_s=0.01, decay_s=1.49)
    signal *= db_to_amp(-18)

    stereo = np.column_stack([signal, signal])
    return stereo


def generate_discovery_swell(fundamental, rng):
    """
    discovery_reveal cue: rich harmonic swell.
    Fundamental + perfect fifth + shimmer partials.
    Returns stereo array (~4s).
    """
    duration = 4.0
    fifth = fundamental * 1.5

    signal = sine_wave(fundamental, duration)
    signal += sine_wave(fifth, duration) * 0.8

    # Shimmer partials (3-5 kHz)
    for _ in range(3):
        freq = rng.uniform(3000, 5000)
        signal += sine_wave(freq, duration) * 0.15

    signal = apply_envelope(signal, attack_s=1.5, decay_s=1.5, sustain_s=1.0)
    signal *= db_to_amp(-6)
    signal /= np.max(np.abs(signal)) + 1e-10
    signal *= db_to_amp(-6)

    stereo = np.column_stack([signal, signal])
    return stereo


# ── Scene-level shaping ───────────────────────────────────────────────────────

def apply_scene_shaping(stereo, timing):
    """Apply per-scene amplitude and pitch shaping to the drone."""
    total_samples = stereo.shape[0]
    envelope = np.ones(total_samples)

    for scene in timing["scenes"]:
        start_s = int(scene["start"] * SR)
        end_s = int(scene["end"] * SR)

        if scene["id"] == "S1_Hook":
            # Fade in from silence over first 3s
            fade_n = int(3.0 * SR)
            fade_end = min(start_s + fade_n, total_samples)
            envelope[:start_s] = 0.0
            envelope[start_s:fade_end] = np.linspace(0, 1, fade_end - start_s)

        elif scene["id"] == "S4_Journey":
            # Slightly reduced level (-3dB)
            s = max(start_s, 0)
            e = min(end_s, total_samples)
            envelope[s:e] *= db_to_amp(-3)

        elif scene["id"] == "S6_Close":
            # Fade to silence over final 5s
            fade_n = int(5.0 * SR)
            fade_start = max(end_s - fade_n, start_s)
            e = min(end_s, total_samples)
            if fade_start < e:
                envelope[fade_start:e] *= np.linspace(1, 0, e - fade_start)
            if e < total_samples:
                envelope[e:] = 0.0

    stereo[:, 0] *= envelope
    stereo[:, 1] *= envelope
    return stereo


def apply_s3_pitch_bend(stereo, timing):
    """
    S3 Alignment: subtle pitch modulation during cloud rotation (~82-90s).
    We simulate this by adding a slightly detuned copy during that window.
    """
    s3 = next(s for s in timing["scenes"] if s["id"] == "S3_Alignment")
    bend_start = int(82.0 * SR)
    bend_end = int(90.0 * SR)

    if bend_end > stereo.shape[0]:
        return stereo

    n = bend_end - bend_start
    t = np.arange(n) / SR
    # Add a subtle +5 Hz pitch-shifted sine (tension)
    tension = 0.1 * np.sin(2 * np.pi * 70.0 * t)  # 65+5 Hz
    # Envelope it smoothly
    env = np.sin(np.pi * np.arange(n) / n)  # smooth hump
    tension *= env

    stereo[bend_start:bend_end, 0] += tension
    stereo[bend_start:bend_end, 1] += tension
    return stereo


# ── Main assembly ─────────────────────────────────────────────────────────────

def overlay(base, layer, start_time):
    """Overlay a stereo layer onto the base array at the given start time."""
    start_sample = int(start_time * SR)
    end_sample = start_sample + layer.shape[0]

    # Ensure layer is stereo
    if layer.ndim == 1:
        layer = np.column_stack([layer, layer])

    # Clip to base length
    if end_sample > base.shape[0]:
        layer = layer[:base.shape[0] - start_sample]
        end_sample = base.shape[0]

    if start_sample < base.shape[0]:
        base[start_sample:end_sample] += layer

    return base


def main():
    timing = load_timing()

    # Determine total duration
    last_scene = timing["scenes"][-1]
    total_duration = last_scene["end"] + 1.0  # 1s padding
    total_samples = int(total_duration * SR)

    rng = np.random.default_rng(42)

    # Phase 1: Base drone
    print("Phase 1: Generating base drone...")
    drone = generate_base_drone(total_samples, rng)

    # Phase 2: Apply scene-level shaping
    print("Phase 2: Applying scene-level shaping...")
    drone = apply_scene_shaping(drone, timing)
    drone = apply_s3_pitch_bend(drone, timing)

    # Phase 3: Overlay cue layers
    print("Phase 3: Generating reactive cues...")

    # Track discovery index for ascending key shifts
    discovery_fundamentals = [65, 77, 92, 109]  # minor third ascent
    discovery_idx = 0

    for scene in timing["scenes"]:
        for cue in scene["cues"]:
            t = cue["time"]

            if cue["type"] == "scene_start":
                print(f"  [shimmer] scene_start at {t:.1f}s")
                shimmer = generate_shimmer_swell(rng)
                drone = overlay(drone, shimmer, t)

            elif cue["type"] == "subtitle_drop":
                print(f"  [ping] subtitle_drop at {t:.1f}s")
                ping = generate_tonal_ping(fundamental=65.0, rng=rng)
                drone = overlay(drone, ping, t)

            elif cue["type"] == "discovery_reveal":
                fund = discovery_fundamentals[min(discovery_idx, len(discovery_fundamentals) - 1)]
                print(f"  [swell] discovery_reveal at {t:.1f}s (fundamental={fund} Hz)")
                swell = generate_discovery_swell(fund, rng)
                drone = overlay(drone, swell, t)
                discovery_idx += 1

    # Phase 4: Final normalization
    print("Phase 4: Normalizing...")
    peak = np.max(np.abs(drone)) + 1e-10
    drone /= peak
    drone *= 0.9  # Leave some headroom

    # Write output
    DRONE_DIR.mkdir(parents=True, exist_ok=True)
    output_16 = np.clip(drone * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(OUTPUT_FILE), SR, output_16)

    print(f"\nDone! Drone score: {OUTPUT_FILE}")
    print(f"  Duration: {total_samples / SR:.1f}s")
    print(f"  Sample rate: {SR} Hz")
    print(f"  Channels: stereo")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `generate_drone.py` and verify output**

```bash
cd /Users/crashy/Development/heiroglyphy
python docs/audio/generate_drone.py
```

Expected output:
- `docs/audio/drone/drone_full.wav` (~200s stereo WAV at 44100 Hz)
- Console showing each cue being generated

Verify:
```bash
python -c "
from scipy.io import wavfile
sr, data = wavfile.read('docs/audio/drone/drone_full.wav')
print(f'Sample rate: {sr}')
print(f'Duration: {len(data)/sr:.1f}s')
print(f'Shape: {data.shape}')
print(f'Channels: {data.shape[1] if data.ndim > 1 else 1}')
print(f'Dtype: {data.dtype}')
"
```

Expected: Sample rate 44100, duration ~201s, shape (N, 2), 2 channels, dtype int16.

Listen to the output to check:
- Warm drone fades in over the first 3s
- Shimmer swells at scene transitions
- Subtle pings at subtitle drops
- Discovery reveals have ascending harmonic swells
- Drone fades out at the end

- [ ] **Step 3: Commit**

```bash
git add docs/audio/generate_drone.py
git commit -m "feat: add generative drone score with reactive audio cues"
```

---

## Chunk 4: Mix, Merge & Integration

### Task 4: Create `mix_audio.py`

**Files:**
- Create: `docs/audio/mix_audio.py`

**Context:** This script loads both audio tracks, applies voice-activated ducking to the drone, mixes them, and merges with the rendered video via FFmpeg. It probes the video duration via `ffprobe` for authoritative length.

- [ ] **Step 1: Write `mix_audio.py`**

Create `docs/audio/mix_audio.py`:

```python
"""
Audio mixer and video merger for Heiroglyphy video.

Loads voice_full.wav and drone_full.wav, applies voice-activated
ducking to the drone, mixes them, and merges with the rendered video.

Usage:
    cd docs/audio
    python mix_audio.py

Requires: ffmpeg and ffprobe on PATH.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

AUDIO_DIR = Path(__file__).resolve().parent
DOCS_DIR = AUDIO_DIR.parent
VOICE_FILE = AUDIO_DIR / "voice_full.wav"
DRONE_FILE = AUDIO_DIR / "drone" / "drone_full.wav"
OUTPUT_FILE = AUDIO_DIR / "mixed_audio.wav"
VIDEO_DIR = DOCS_DIR / "media" / "videos" / "heiroglyphy_video" / "1080p60"
VIDEO_FILE = VIDEO_DIR / "HeiroglyphyVideo.mp4"
FINAL_VIDEO = DOCS_DIR / "media" / "HeiroglyphyVideo_final.mp4"

SR = 44100


def db_to_amp(db):
    return 10.0 ** (db / 20.0)


def get_video_duration(video_path):
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not probe video duration: {e}")
        print("Falling back to audio track length.")
        return None


def read_wav_float(path):
    """Read WAV, return (sample_rate, float64 array)."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    return sr, data


def resample_if_needed(data, src_sr, target_sr):
    """Resample audio if sample rates differ."""
    if src_sr == target_sr:
        return data
    from math import gcd
    g = gcd(target_sr, src_sr)
    up = target_sr // g
    down = src_sr // g
    if data.ndim == 1:
        return resample_poly(data, up, down)
    else:
        # Resample each channel
        channels = [resample_poly(data[:, ch], up, down) for ch in range(data.shape[1])]
        return np.column_stack(channels)


def pad_or_trim(data, target_samples):
    """Pad with zeros or trim to target length."""
    if data.ndim == 1:
        if len(data) >= target_samples:
            return data[:target_samples]
        return np.pad(data, (0, target_samples - len(data)))
    else:
        if data.shape[0] >= target_samples:
            return data[:target_samples]
        padding = np.zeros((target_samples - data.shape[0], data.shape[1]))
        return np.vstack([data, padding])


def compute_voice_envelope(voice, window_ms=100):
    """Compute RMS envelope of voice signal for ducking."""
    window_samples = int(window_ms / 1000.0 * SR)
    if voice.ndim > 1:
        mono = voice.mean(axis=1)
    else:
        mono = voice

    # RMS in sliding window
    squared = mono ** 2
    window = np.ones(window_samples) / window_samples
    rms = np.sqrt(np.convolve(squared, window, mode="same"))
    return rms


def apply_ducking(drone, voice_envelope, threshold_db=-40, duck_db=-3,
                  attack_ms=200, release_ms=200):
    """
    Apply voice-activated ducking to the drone.
    When voice is active (above threshold), attenuate drone.
    """
    threshold = db_to_amp(threshold_db)
    duck_amount = db_to_amp(duck_db)

    # Create ducking envelope: 1.0 when no voice, duck_amount when voice active
    active = (voice_envelope > threshold).astype(np.float64)

    # Smooth with IIR filter (vectorized attack/release)
    attack_coeff = 1.0 / max(int(attack_ms / 1000.0 * SR), 1)
    release_coeff = 1.0 / max(int(release_ms / 1000.0 * SR), 1)

    # Use scipy.signal.lfilter for one-pole smoothing
    # Approximate: use a single coefficient (attack for rising, release for falling)
    # Simple approach: two-pass with different coefficients
    from scipy.signal import lfilter
    # Single-pole lowpass as attack envelope
    coeff = attack_coeff  # use faster of the two for responsiveness
    b = [coeff]
    a = [1.0, -(1.0 - coeff)]
    smoothed = lfilter(b, a, active)
    smoothed = np.clip(smoothed, 0.0, 1.0)

    # Convert to gain: 1.0 when no voice, duck_amount when voice
    gain = 1.0 - smoothed * (1.0 - duck_amount)

    if drone.ndim == 1:
        drone *= gain
    else:
        drone[:, 0] *= gain
        drone[:, 1] *= gain

    return drone


def mono_to_stereo(data):
    """Convert mono to stereo by duplicating."""
    if data.ndim == 1:
        return np.column_stack([data, data])
    return data


def main():
    # Check inputs exist
    if not VOICE_FILE.exists():
        print(f"Error: {VOICE_FILE} not found. Run generate_voice.py first.")
        sys.exit(1)
    if not DRONE_FILE.exists():
        print(f"Error: {DRONE_FILE} not found. Run generate_drone.py first.")
        sys.exit(1)

    # Load audio
    print("Loading audio tracks...")
    voice_sr, voice = read_wav_float(VOICE_FILE)
    drone_sr, drone = read_wav_float(DRONE_FILE)

    # Resample if needed
    voice = resample_if_needed(voice, voice_sr, SR)
    drone = resample_if_needed(drone, drone_sr, SR)

    # Ensure both are stereo
    voice = mono_to_stereo(voice)
    drone = mono_to_stereo(drone)

    # Determine target duration
    video_duration = None
    if VIDEO_FILE.exists():
        video_duration = get_video_duration(VIDEO_FILE)
        if video_duration:
            print(f"Video duration: {video_duration:.1f}s")

    if video_duration:
        target_samples = int(video_duration * SR)
    else:
        target_samples = max(voice.shape[0], drone.shape[0])
        print(f"Using audio length: {target_samples / SR:.1f}s")

    # Pad/trim to target length
    voice = pad_or_trim(voice, target_samples)
    drone = pad_or_trim(drone, target_samples)

    # Apply voice-activated ducking to drone
    print("Applying voice-activated ducking...")
    voice_envelope = compute_voice_envelope(voice)
    drone = apply_ducking(drone, voice_envelope)

    # Mix: voice at 0dB, drone at -18dB
    print("Mixing...")
    drone_level = db_to_amp(-18)
    mixed = voice + drone * drone_level

    # Normalize to -1dB peak
    peak = np.max(np.abs(mixed)) + 1e-10
    target_peak = db_to_amp(-1)
    mixed = mixed * (target_peak / peak)

    # Write mixed audio
    output_16 = np.clip(mixed * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(str(OUTPUT_FILE), SR, output_16)
    print(f"Mixed audio: {OUTPUT_FILE}")
    print(f"  Duration: {target_samples / SR:.1f}s")

    # Merge with video if available
    if VIDEO_FILE.exists():
        print(f"\nMerging with video...")
        FINAL_VIDEO.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(VIDEO_FILE),
            "-i", str(OUTPUT_FILE),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(FINAL_VIDEO),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Final video: {FINAL_VIDEO}")
        else:
            print(f"FFmpeg error:\n{result.stderr}")
            sys.exit(1)
    else:
        print(f"\nNote: Video not found at {VIDEO_FILE}")
        print("Run 'manim -pqh heiroglyphy_video.py HeiroglyphyVideo' first,")
        print("then re-run this script to merge audio with video.")

    print("\nDone!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `mix_audio.py` and verify output**

```bash
cd /Users/crashy/Development/heiroglyphy
python docs/audio/mix_audio.py
```

Expected output:
- `docs/audio/mixed_audio.wav` (stereo, 44100 Hz, matches video duration)
- `docs/media/HeiroglyphyVideo_final.mp4` (if video exists)
- Console showing mix progress and FFmpeg merge

Verify mixed audio:
```bash
python -c "
from scipy.io import wavfile
import numpy as np
sr, data = wavfile.read('docs/audio/mixed_audio.wav')
peak_db = 20 * np.log10(np.max(np.abs(data / 32768.0)) + 1e-10)
print(f'Sample rate: {sr}')
print(f'Duration: {len(data)/sr:.1f}s')
print(f'Channels: {data.shape[1] if data.ndim > 1 else 1}')
print(f'Peak level: {peak_db:.1f} dB')
"
```

Expected: Sample rate 44100, stereo, peak level ~-1.0 dB.

If final video was produced, play it to verify audio syncs with visuals.

- [ ] **Step 3: Add generated audio files to .gitignore**

Check if `.gitignore` already covers `*.wav` and `*.mp4` in the audio/media dirs. If not, add:

```
# Generated audio files
docs/audio/voice/*.wav
docs/audio/drone/*.wav
docs/audio/voice_full.wav
docs/audio/mixed_audio.wav
docs/media/HeiroglyphyVideo_final.mp4
```

- [ ] **Step 4: Commit**

```bash
git add docs/audio/mix_audio.py docs/audio/generate_voice.py docs/audio/generate_drone.py docs/audio/audio_timing.json .gitignore
git commit -m "feat: complete audio pipeline — voice, drone, mix, and video merge"
```

### Task 5: Full pipeline integration test

- [ ] **Step 1: Run the complete pipeline end-to-end**

```bash
cd /Users/crashy/Development/heiroglyphy

# Step 1: Generate voice (requires OPENAI_API_KEY)
python docs/audio/generate_voice.py

# Step 2: Generate drone (no API needed)
python docs/audio/generate_drone.py

# Step 3: Mix and merge
python docs/audio/mix_audio.py
```

- [ ] **Step 2: Verify all outputs exist**

```bash
ls -lh docs/audio/voice/voice_*.wav
ls -lh docs/audio/drone/drone_full.wav
ls -lh docs/audio/voice_full.wav
ls -lh docs/audio/mixed_audio.wav
ls -lh docs/media/HeiroglyphyVideo_final.mp4
```

- [ ] **Step 3: Listen to the final video**

Open `docs/media/HeiroglyphyVideo_final.mp4` and verify:
- Voice narration plays at the right times (matches on-screen text/animations)
- Drone fades in at the start, sustains throughout, fades out at the end
- Shimmer swells are audible at scene transitions
- Subtle pings occur at subtitle drops
- Discovery reveals have richer harmonic swells with ascending pitch
- Voice is clearly audible above the drone at all times
- No clipping, no silence gaps where there should be audio

- [ ] **Step 4: Adjust timing if needed**

If narration doesn't align with visuals, edit `docs/audio/audio_timing.json` segment start times and re-run the pipeline. Only `mix_audio.py` needs re-running if just adjusting placement (voice segments are cached). If timing changes affect segment windows, re-run `generate_voice.py` too.

- [ ] **Step 5: Final commit**

```bash
git add -A docs/audio/*.py docs/audio/audio_timing.json .gitignore
git commit -m "feat: audio pipeline complete — voice narration + drone score"
```
