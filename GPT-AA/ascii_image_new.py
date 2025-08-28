#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

def to_ascii_gray(image, width=120, charset=None, contrast=1.0, invert=False, brightness=1.0):
    """グレースケールASCIIアート生成"""
    if charset is None:
        charset = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    aspect = 0.5  # 文字縦横比補正

    img = image.convert("L")
    if invert:
        img = ImageOps.invert(img)

    # 明度補正
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    # コントラスト補正
    if contrast != 1.0:
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = arr.mean()
        arr = (arr - mean) * contrast + mean
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    w, h = img.size
    new_w = width
    new_h = max(1, int(h / w * new_w * aspect))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    chars = np.array(list(charset))
    idx = (arr * (len(chars) - 1)).round().astype(int)
    mapped = chars[idx]
    lines = ["".join(row) for row in mapped]
    return "\n".join(lines)

def to_ascii_color(image, width=120, char_colors=None, brightness=1.0):
    """カラー画像 → 指定色に近い文字でASCII化"""
    aspect = 0.5
    if char_colors is None:
        # デフォルト: 白黒文字
        char_colors = [(c, (i,i,i)) for i, c in enumerate(
            " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")]

    # RGBA画像の場合もRGBに変換
    img = image.convert("RGB")

    # 明度補正
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    w, h = img.size
    new_w = width
    new_h = max(1, int(h / w * new_w * aspect))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)

    lines = []
    char_array, color_array = zip(*char_colors)
    color_array = np.array(color_array, dtype=np.float32)

    for row in arr:
        line = ""
        for px in row:
            # ピクセル色と文字色の距離を計算
            dists = np.linalg.norm(color_array - px, axis=1)
            idx = np.argmin(dists)
            line += char_array[idx]
        lines.append(line)
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="input image")
    p.add_argument("output", help="output text file")
    p.add_argument("--width", type=int, default=120, help="output width in characters")
    p.add_argument("--charset", type=str, default=None, help="characters from light to dark (grayscale only)")
    p.add_argument("--contrast", type=float, default=1.0, help="contrast multiplier")
    p.add_argument("--invert", action="store_true", help="invert brightness mapping")
    p.add_argument("--color", action="store_true", help="enable color-based ASCII")
    p.add_argument("--brightness", type=float, default=1.0, help="brightness multiplier (default=1.0)")
    args = p.parse_args()

    img = Image.open(args.input)

    if args.color:
        # 文字と色のマッピング例（自由に追加可能）
        A= '1234567890-^qwertyuiop@[asdfghjkl\\;:]zxcvbnm,./!"#$%&\'()=~|QWERTYUIOP`{ASDFGHJKL+*}ZXCVBNM<>?_'
        char_colors = [
            (" ", (255,255,255)),("1", (219,219,219)),("2", (219,219,219)),("3", (200,200,219)),
            ("4", (182,191,191)),("5", (200,200,200)),("6", (200,200,200)),("7", (200,219,219)),
            ("8", (181,181,181)),("9", (200,200,181)),("0", (200,200,200)),("-", (213,213,213)),
            ("^", (237,237,237)),("q", (199,199,218)),("w", (178,178,178)),("e", (200,200,200)),
            ("r", (237,218,218)),("t", (237,218,237)),("y", (218,218,218)),("u", (218,218,218)),
            ("i", (237,237,219)),("o", (199,199,199)),("p", (218,218,218)),("@", (160,163,160)),
            ('[', (219,219,219)),('a', (199,199,199)),('s', (199,218,199)),('d', (181,181,200)),
            ('f', (219,219,219)),('g', (199,199,199)),('h', (200,299,200)),('j', (236,236,219)),
            ('k', (181,200,200)),('l', (237,219,219)),('\\', (163,181,163)),(';', (255,255,255)),
            (':', (255,255,255)),(']', (219,219,219)),("z", (237,218,237)),("x", (218,218,199)),
            ("c", (237,218,218)),("v", (218,218,218)),("b", (200,200,200)),("n", (218,218,218)),
            ("m", (199,199,199)),(",", (255,255,255)),(".", (255,255,255)),("/", (219,219,219)),
            ("!", (219,219,200)),("\"", (237,237,218)),("#", (163,163,163)),("$", (181,163,181)),
            ("&", (181,181,181)),("'", (237,237,237)),("(", (219,219,219)),(")", (219,219,219)),
            ("=", (199,218,199)),("~", (218,237,237)),("|", (219,219,219)),("Q", (160,160,160)),
            ("W", (255,255,255)),("E", (219,200,182)),('R', (181,163,163)),('S', (181,181,181)),
            ('T', (219,219,219)),('Y', (182,182,182)),('U', (182,200,182)),('I', (200,200,182)),
            ('O', (160,181,160)),('P', (200,182,182)),('`', (255,219,219)),('{', (219,219,219)),
            ('A', (181,181,181)),('S', (181,181,181)),('D', (200,182,182)),('F', (182,182,182)),
            ('G', (181,181,200)),('H', (182,200,182)),('J', (200,200,200)),('K', (182,182,182)),
            ('L', (219,219,219)),('+', (199,218,199)),('*', (199,199,199)),('}', (219,219,219)),
            ('Z', (200,200,219)),('X', (182,182,182)),('C', (200,200,200)),('V', (182,200,182)),
            ('B', (200,182,182)),('N', (163,181,163)),('M', (255,255,255)),('<', (219,219,200)),
            ('>', (200,219,219)),('?', (219,219,219)),('_', (255,255,255)),("█", (29,29,0)),
        ]
        art = to_ascii_color(img, width=args.width, char_colors=char_colors, brightness=args.brightness)
    else:
        art = to_ascii_gray(img, width=args.width, charset=args.charset,
                            contrast=args.contrast, invert=args.invert, brightness=args.brightness)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(art)

if __name__ == "__main__":
    main()
