#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an image to ASCII art and save as a .txt file.
Usage:
  python ascii_image.py input.jpg output.txt --width 960 --charset " .'`^\",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
Notes:
- Turn OFF "Word Wrap" in Notepad.
- Use zoom 10–30% and a monospaced font (Consolas recommended).
"""
import argparse
from PIL import Image, ImageOps
import numpy as np

def to_ascii(image: Image.Image, width: int = 960, charset: str = None, contrast: float = 1.0, invert: bool = False):
    # Character set from light to dark (default)
    if charset is None:
        charset = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        #charset = "1234567890-^qwertyuiop@[asdfghjkl;:]zxcvbnm,./!\"#$%&'()=~|QWERTYUIOP`{ASDFGHJKL+*}ZXCVBNM<>?_ "
        charset=np.array(list(charset[::-1]))
    # Aspect compensation: characters are taller than they are wide.
    # 0.46–0.55 works for many fonts when viewing in Notepad at small zoom.
    aspect = 0.5

    # Convert to grayscale and resize to target character grid
    img = image.convert("L")
    if invert:
        img = ImageOps.invert(img)
    # optional contrast scaling
    if contrast != 1.0:
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = arr.mean()
        arr = (arr - mean) * contrast + mean
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")

    w, h = img.size
    new_w = width
    new_h = max(1, int(h / w * new_w * aspect))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    # Map brightness to chars (0=black -> last char)
    chars = np.array(list(charset))
    idx = (arr * (len(chars) - 1)).round().astype(int)
    mapped = chars[idx]

    # Join into lines
    lines = ["".join(row) for row in mapped]
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="input image file (jpg, png, etc.)")
    p.add_argument("output", help="output text file")
    p.add_argument("--width", type=int, default=960, help="output width in characters (default: 960)")
    p.add_argument("--charset", type=str, default=None, help="characters from light to dark")
    p.add_argument("--contrast", type=float, default=1.0, help="contrast multiplier (e.g., 1.2)")
    p.add_argument("--invert", action="store_true", help="invert brightness mapping")
    args = p.parse_args()

    img = Image.open(args.input)
    art = to_ascii(img, width=args.width, charset=args.charset, contrast=args.contrast, invert=args.invert)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(art)

if __name__ == "__main__":
    main()
