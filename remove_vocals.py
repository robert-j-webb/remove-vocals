#!/usr/bin/env python3
"""Remove vocals from an audio file (MP3, WAV, FLAC, M4A...).

Setup:  pip install demucs soundfile   (and have ffmpeg installed)
Usage:  python remove_vocals.py song.mp3 [-o out.wav] [--vocals-too] [--model htdemucs]
"""
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Strip the vocal track from a song.")
    p.add_argument("input", help="Input audio file")
    p.add_argument("-o", "--output", help="Output file (default: <name>_instrumental.wav)")
    p.add_argument("--model", default="htdemucs", help="Demucs model (htdemucs, htdemucs_ft, mdx_extra)")
    p.add_argument("--vocals-too", action="store_true", help="Also save the isolated vocals")
    p.add_argument("--mp3", action="store_true", help="Write MP3 instead of WAV")
    a = p.parse_args()

    src = Path(a.input).resolve()
    if not src.exists():
        sys.exit(f"File not found: {src}")
    ext = ".mp3" if a.mp3 else ".wav"
    out = Path(a.output) if a.output else src.with_name(f"{src.stem}_instrumental{ext}")

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
               "-n", a.model, "-o", tmp, str(src)]
        if a.mp3:
            cmd.insert(3, "--mp3")
        print("Separating (first run downloads the model)...")
        subprocess.run(cmd, check=True)

        stem_dir = Path(tmp) / a.model / src.stem
        shutil.move(str(stem_dir / f"no_vocals{ext}"), out)
        print(f"Instrumental: {out}")
        if a.vocals_too:
            v = out.with_name(f"{src.stem}_vocals{ext}")
            shutil.move(str(stem_dir / f"vocals{ext}"), v)
            print(f"Vocals: {v}")


if __name__ == "__main__":
    main()
