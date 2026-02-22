#!/usr/bin/env python3
"""
hide_in_wav.py
CLI tool: text -> image -> spectrogram-audio WAV, then deletes the image.

Open the resulting WAV in Audacity:
  Track dropdown -> Spectrogram
  Increase Max Frequency (e.g., 8000-12000), Window size 2048/4096, Range ~80 dB
"""

import argparse
import os
import sys
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def pick_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TTF font; fall back to Pillow default if not available."""
    if font_path:
        return ImageFont.truetype(font_path, font_size)

    # Try some common Linux fonts
    common = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in common:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass

    return ImageFont.load_default()


def make_text_image(message: str, out_png: str, width: int, height: int, font_size: int, font_path: str | None):
    """Create black background with white text centered."""
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    font = pick_font(font_path, font_size)

    # Simple word-wrapping for long messages
    max_chars = max(10, int(width / (font_size * 0.6)))
    words = message.split()
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # Measure and center lines
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    total_h = int(sum(line_heights) + (len(lines) - 1) * (font_size * 0.35))
    y = (height - total_h) // 2

    for i, line in enumerate(lines):
        lw = line_widths[i]
        lh = line_heights[i]
        x = (width - lw) // 2
        draw.text((x, y), line, fill="white", font=font)
        y += int(lh + font_size * 0.35)

    img.save(out_png)


def _flip_vertical(img: Image.Image) -> Image.Image:
    """Flip image vertically so top becomes high frequencies in spectrogram."""
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def image_to_wav_spectrogram(
    png_path: str,
    wav_path: str,
    *,
    sample_rate: int = 44100,
    f_min: float = 300.0,
    f_max: float = 8000.0,
    pixels_per_second: float = 120.0,
    gain: float = 0.9,
):
    """
    Convert image brightness into tones across frequency bins so the image appears in a spectrogram.

    - width determines duration: duration = width / pixels_per_second
    - height maps to frequencies [f_min, f_max]
    - each column x becomes a short time slice
    """
    img = Image.open(png_path).convert("L")
    img = _flip_vertical(img)
    A = np.asarray(img, dtype=np.float32) / 255.0  # shape: (H, W)
    H, W = A.shape

    hop = int(sample_rate / pixels_per_second)
    hop = max(hop, 64)  # keep a sane minimum

    duration_samples = W * hop
    audio = np.zeros(duration_samples, dtype=np.float32)

    freqs = f_min + (np.arange(H, dtype=np.float32) / max(1, H - 1)) * (f_max - f_min)
    t = np.arange(hop, dtype=np.float32) / sample_rate
    window = np.hanning(hop).astype(np.float32)

    for x in range(W):
        column = A[:, x]
        if column.max() < 0.01:
            continue

        slice_sig = np.zeros(hop, dtype=np.float32)
        for y in range(H):
            amp = column[y]
            if amp < 0.02:
                continue
            slice_sig += amp * np.sin(2.0 * np.pi * freqs[y] * t)

        slice_sig *= window
        start = x * hop
        audio[start : start + hop] += slice_sig

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = (audio / peak) * gain

    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser(description="Hide text as spectrogram art inside a WAV file.")
    ap.add_argument("-o", "--out", default="hidden.wav", help="Output WAV path (default: hidden.wav)")
    ap.add_argument("--width", type=int, default=1200, help="Generated image width in pixels (default: 1200)")
    ap.add_argument("--height", type=int, default=300, help="Generated image height in pixels (default: 300)")
    ap.add_argument("--font-size", type=int, default=110, help="Font size (default: 110)")
    ap.add_argument("--font", default=None, help="Optional path to a .ttf font")
    ap.add_argument("--fmin", type=float, default=300.0, help="Min frequency for spectrogram (default: 300)")
    ap.add_argument("--fmax", type=float, default=8000.0, help="Max frequency for spectrogram (default: 8000)")
    ap.add_argument("--pps", type=float, default=120.0, help="Pixels per second (controls duration) (default: 120)")
    args = ap.parse_args()

    try:
        message = input("what do u wanna hide? > ").strip()
    except KeyboardInterrupt:
        print("\nbye.")
        sys.exit(1)

    if not message:
        print("no message entered. exiting.")
        sys.exit(1)

    tmp_png = None
    try:
        fd, tmp_png = tempfile.mkstemp(suffix=".png", prefix="spectro_msg_")
        os.close(fd)

        make_text_image(
            message=message,
            out_png=tmp_png,
            width=args.width,
            height=args.height,
            font_size=args.font_size,
            font_path=args.font,
        )

        image_to_wav_spectrogram(
            png_path=tmp_png,
            wav_path=args.out,
            f_min=args.fmin,
            f_max=args.fmax,
            pixels_per_second=args.pps,
        )

        print(f"done. created: {args.out}")
        print("audacity tip: view -> spectrogram; try window size 2048/4096, max freq 8k-12k, range ~80 dB.")
    finally:
        if tmp_png and os.path.exists(tmp_png):
            try:
                os.remove(tmp_png)
            except Exception:
                pass


if __name__ == "__main__":
    main()
