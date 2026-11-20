"""Usage: python -m examples.run_audio_engine path/to/input.wav"""
import argparse
from pathlib import Path
from backend.dsp.audio_engine import load_audio, apply_gain, save_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--gain", type=float, default=-6.0)
    parser.add_argument("--output", type=Path, default=Path("audio/output/processed.wav"))
    args = parser.parse_args()
    audio = load_audio(args.input)
    print(f"Input: {audio.sample_rate} Hz, {audio.channels} channels, {audio.duration:.2f} s")
    print(f"Peak: {audio.peak:.4f}; RMS: {audio.rms:.4f}")
    processed = apply_gain(audio, args.gain)
    save_audio(processed, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
