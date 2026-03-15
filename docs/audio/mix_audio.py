"""
Audio mixer and video merger for Heiroglyphy video.

Loads voice_full.wav and drone_full.wav, applies voice-activated
ducking to the drone, mixes them, and merges with the rendered video.

Usage:
    cd docs/audio
    python mix_audio.py

Requires: ffmpeg and ffprobe on PATH.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, lfilter

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

    # Smooth with IIR filter (vectorized one-pole lowpass)
    attack_coeff = 1.0 / max(int(attack_ms / 1000.0 * SR), 1)
    coeff = attack_coeff
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
