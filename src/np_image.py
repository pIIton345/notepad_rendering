import argparse
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import sys

def apply_auto_rgb(img, gamma=0.7, exposure=0.5):
    """
    カラー画像用のアウトオート（暗部持ち上げ + ハイライト圧縮）
    - gamma < 1 : 暗部を持ち上げる
    - exposure (0..1) : 値が小さいほどハイライト圧縮が強くなる（0.5 程度が無難）
    処理は HSV の V チャネルに対して行う。
    """
    hsv = img.convert("HSV")
    arr = np.asarray(hsv).astype(np.float32)
    h = arr[..., 0]
    s = arr[..., 1]
    v = arr[..., 2] / 255.0

    # ガンマで暗部を持ち上げ
    v = np.power(v, gamma)

    # ソフトトーンマッピング（ハイライトの伸びを抑える）
    # v -> v/(v + exposure) (exposure は 0.1..1.0 の範囲が有効)
    v = v / (v + exposure)
    v = np.clip(v, 0.0, 1.0)

    arr[..., 2] = (v * 255.0).astype(np.float32)
    out = Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")
    return out

def apply_auto_gray(img, gamma=0.7, exposure=0.5):
    """
    グレースケール画像用のアウトオート
    入力は 'L' モードの Image を想定
    """
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = arr / (arr + exposure)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8)).convert("L")

def to_ascii_gray(image, width=120, charset=None, contrast=1.0, invert=False, brightness=1.0, aspect=0.5):
    """グレースケールASCIIアート生成"""
    if charset is None:
        charset = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    img = image.convert("L")
    if invert:
        img = ImageOps.invert(img)

    # 明度補正
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    # コントラスト補正（ピクセル単位で平均を保つ方式）
    if contrast != 1.0:
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = arr.mean()
        arr = (arr - mean) * contrast + mean
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")

    w, h = img.size
    new_w = max(1, int(width))
    # 文字の縦横比補正を使用
    new_h = max(1, int(h / w * new_w * aspect))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    chars = np.array(list(charset))
    idx = (arr * (len(chars) - 1)).round().astype(int)
    mapped = chars[idx]
    lines = ["".join(row) for row in mapped]
    return "\n".join(lines)

def to_ascii_color(image, width=120, char_colors=None, brightness=1.0, aspect=0.5):
    """カラー画像 → 指定色に近い文字でASCII化"""
    if char_colors is None:
        char_colors = build_default_char_colors()

    img = image.convert("RGB")

    # 明度補正
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    w, h = img.size
    new_w = max(1, int(width))
    new_h = max(1, int(h / w * new_w * aspect))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)

    lines = []
    # char_colors は (char, (r,g,b)) のリストを想定
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

def build_default_char_colors():
    # 元のリストを軽く修正（不正な値を修正）
    return [
        (' ', (255, 255, 255)), ('1', (219, 219, 219)), ('3', (200, 200, 219)),
        ('4', (182, 191, 191)), ('5', (200, 200, 200)), ('7', (200, 219, 219)),
        ('8', (181, 181, 181)), ('9', (200, 200, 181)), ('-', (213, 213, 213)),
        ('^', (237, 237, 237)), ('q', (199, 199, 218)), ('w', (178, 178, 178)),
        ('r', (237, 218, 218)), ('t', (237, 218, 237)), ('y', (218, 218, 218)),
        ('i', (237, 237, 219)), ('o', (199, 199, 199)), ('@', (160, 163, 160)),
        ('s', (199, 218, 199)), ('d', (181, 181, 200)), ('j', (236, 236, 219)),
        ('k', (181, 200, 200)), ('l', (237, 219, 219)), ('\\', (163, 181, 163)),
        ('!', (219, 219, 200)), ('"', (237, 237, 218)), ('#', (163, 163, 163)),
        ('$', (181, 163, 181)), ('&', (181, 181, 181)), ('=', (199, 218, 199)),
        ('~', (218, 237, 237)), ('Q', (160, 160, 160)), ('E', (219, 200, 182)),
        ('R', (181, 163, 163)), ('S', (181, 181, 181)), ('Y', (182, 182, 182)),
        ('U', (182, 200, 182)), ('I', (200, 200, 182)), ('D', (200, 182, 182)),
        ('`', (255, 219, 219)), ('█', (29, 29, 0)),
    ]
def main():
    p = argparse.ArgumentParser(description="Image -> ASCII (grayscale or color) with --auto shadow/highlight auto-correct")
    p.add_argument("--input",required=True, help="input image")
    p.add_argument("--output",required=True, help="output text file")
    p.add_argument("--width", type=int, default=120, help="output width in characters")
    p.add_argument("--charset", type=str, default=None, help="characters from light to dark (grayscale only)")
    p.add_argument("--contrast", type=float, default=1.0, help="contrast multiplier")
    p.add_argument("--invert", action="store_true", help="invert brightness mapping")
    p.add_argument("--color", action="store_true", help="enable color-based ASCII")
    p.add_argument("--brightness", type=float, default=1.0, help="brightness multiplier (default=1.0)")
    p.add_argument("--auto", action="store_true", help="auto adjust shadows/highlights (avoid white clipping)")
    p.add_argument("--gamma", type=float, default=0.7, help="gamma used by --auto (default 0.7, <1 lifts shadows)")
    p.add_argument("--exposure", type=float, default=0.5, help="exposure-like parameter for --auto tone-mapping (default 0.5)")
    p.add_argument("--aspect", type=float, default=0.5,
                   help="character aspect ratio (height/width). >0.5 makes characters taller, <0.5 squashes vertically. default=0.5")
    args = p.parse_args()

    try:
        img = Image.open(args.input)
    except Exception as e:
        print("Error: cannot open input:", e, file=sys.stderr)
        sys.exit(1)

    # --- auto が指定されたら適用（カラー/グレース別） ---
    if args.auto:
        if args.color:
            img = apply_auto_rgb(img.convert("RGB"), gamma=args.gamma, exposure=args.exposure)
        else:
            # グレースケール版: convert してから処理
            gray = img.convert("L")
            gray = apply_auto_gray(gray, gamma=args.gamma, exposure=args.exposure)
            img = gray  # 'L' モードの Image

    # 以降は brightness/contrast 等の既存フローに渡す
    if args.color:
        char_colors = build_default_char_colors()
        art = to_ascii_color(img, width=args.width, char_colors=char_colors,
                             brightness=args.brightness, aspect=args.aspect)
    else:
        art = to_ascii_gray(img, width=args.width, charset=args.charset,
                            contrast=args.contrast, invert=args.invert,
                            brightness=args.brightness, aspect=args.aspect)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(art)
    except Exception as e:
        print("Error: cannot write output:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
