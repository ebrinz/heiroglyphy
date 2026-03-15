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
