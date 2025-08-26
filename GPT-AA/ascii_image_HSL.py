#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
画像をASCIIアートに変換（グレースケール or HSLカラー距離）
"""

import argparse
from PIL import Image, ImageOps
import numpy as np
import colorsys


# ---------------- グレースケール ASCII ----------------
def to_ascii_gray(image, width=120, charset=None, contrast=1.0, invert=False):
    if charset is None:
        charset = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    aspect = 0.5

    img = image.convert("L")
    if invert:
        img = ImageOps.invert(img)

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
    return "\n".join("".join(row) for row in mapped)


# ---------------- RGB→HSL変換（NumPy高速版） ----------------
def rgb_to_hsl_np(rgb_arr):
    """RGB配列 (H, W, 3) → HSL配列 (H, W, 3)"""
    rgb_arr = rgb_arr / 255.0
    r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]

    maxc = np.max(rgb_arr, axis=-1)
    minc = np.min(rgb_arr, axis=-1)
    l = (minc + maxc) / 2.0

    s = np.zeros_like(l)
    mask = maxc != minc
    s[mask] = np.where(
        l[mask] <= 0.5,
        (maxc[mask] - minc[mask]) / (maxc[mask] + minc[mask]),
        (maxc[mask] - minc[mask]) / (2.0 - maxc[mask] - minc[mask]),
    )

    h = np.zeros_like(l)
    rc = (maxc - r) / (maxc - minc + 1e-10)
    gc = (maxc - g) / (maxc - minc + 1e-10)
    bc = (maxc - b) / (maxc - minc + 1e-10)

    mask_r = (maxc == r) & mask
    mask_g = (maxc == g) & mask
    mask_b = (maxc == b) & mask

    h[mask_r] = (bc - gc)[mask_r]
    h[mask_g] = 2.0 + (rc - bc)[mask_g]
    h[mask_b] = 4.0 + (gc - rc)[mask_b]

    h = (h / 6.0) % 1.0

    return np.stack([h, l, s], axis=-1)


# ---------------- HSLカラー ASCII（ベクトル化） ----------------
def to_ascii_color_hsl(image, width=120, char_colors=None):
    aspect = 0.5
    if char_colors is None:
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
            ("!", (219,219,200)),('"', (237,237,218)),("#", (163,163,163)),("$", (181,163,181)),
            ("&", (181,181,181)),("\'", (237,237,237)),("(", (219,219,219)),(")", (219,219,219)),
            ("=", (199,218,199)),("~", (218,237,237)),("|", (219,219,219)),("Q", (160,160,160)),
            ("W", (255,255,255)),("E", (219,200,182)),('R', (181,163,163)),('S', (181,181,181)),#Wは144,144,144だけど色がつよすぎる
            ('T', (219,219,219)),('Y', (182,182,182)),('U', (182,200,182)),('I', (200,200,182)),
            ('O', (160,181,160)),('P', (200,182,182)),('`', (255,219,219)),('{', (219,219,219)),
            ('A', (181,181,181)),('S', (181,181,181)),('D', (200,182,182)),('F', (182,182,182)),
            ('G', (181,181,200)),('H', (182,200,182)),('J', (200,200,200)),('K', (182,182,182)),
            ('L', (219,219,219)),('+', (199,218,199)),('*', (199,199,199)),('}', (219,219,219)),
            ('Z', (200,200,219)),('X', (182,182,182)),('C', (200,200,200)),('V', (182,200,182)),
            ('B', (200,182,182)),('N', (163,181,163)),('M', (255,255,255)),('<', (219,219,200)),#Mは142,142,142
            ('>', (200,219,219)),('?', (219,219,219)),('_', (255,255,255)),("█", (29,29,0)),
        ]

    img = image.convert("RGB")
    w, h = img.size
    new_w = width
    new_h = max(1, int(h / w * new_w * aspect))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)

    # 画像全体を HSL に変換
    hsl_arr = rgb_to_hsl_np(arr)

    # char_colors を HSLに変換
    char_array, rgb_colors = zip(*char_colors)
    hsl_colors = np.array([rgb_to_hsl_np(np.array([[c]], dtype=np.float32))[0, 0] for c in rgb_colors])

    # 重み付け（Hue=5, Lightness=4, Saturation=1）
    weights = np.array([5.0, 4.0, 1.0], dtype=np.float32)

    # Hue距離は円環を考慮
    h_diff = np.abs(hsl_arr[..., 0, None] - hsl_colors[None, None, :, 0])
    h_diff = np.minimum(h_diff, 1 - h_diff)

    l_diff = np.abs(hsl_arr[..., 1, None] - hsl_colors[None, None, :, 1])
    s_diff = np.abs(hsl_arr[..., 2, None] - hsl_colors[None, None, :, 2])

    dists = (weights[0] * h_diff) ** 2 + (weights[1] * l_diff) ** 2 + (weights[2] * s_diff) ** 2
    idx = np.argmin(dists, axis=-1)

    mapped = np.array(char_array)[idx]
    return "\n".join("".join(row) for row in mapped)


# ---------------- メイン ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="input image")
    p.add_argument("output", help="output text file")
    p.add_argument("--width", type=int, default=120, help="output width in characters")
    p.add_argument("--charset", type=str, default=None, help="characters from light to dark (grayscale only)")
    p.add_argument("--contrast", type=float, default=1.0, help="contrast multiplier")
    p.add_argument("--invert", action="store_true", help="invert brightness mapping")
    p.add_argument("--color", action="store_true", help="enable HSL color-based ASCII")
    args = p.parse_args()

    img = Image.open(args.input)

    if args.color:
        art = to_ascii_color_hsl(img, width=args.width)
    else:
        art = to_ascii_gray(img, width=args.width, charset=args.charset,
                            contrast=args.contrast, invert=args.invert)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(art)


if __name__ == "__main__":
    main()
