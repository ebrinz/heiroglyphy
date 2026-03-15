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


# -- Pink Noise (Voss-McCartney algorithm) ------------------------------------

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


# -- Synthesis primitives -----------------------------------------------------

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


# -- Base drone layer ---------------------------------------------------------

def generate_base_drone(total_samples, rng):
    """
    Base drone: 65 Hz fundamental + harmonics + pink noise + LFO.
    Returns stereo array (N, 2).
    """
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


# -- Cue layers ---------------------------------------------------------------

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


# -- Scene-level shaping ------------------------------------------------------

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


# -- Main assembly ------------------------------------------------------------

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
