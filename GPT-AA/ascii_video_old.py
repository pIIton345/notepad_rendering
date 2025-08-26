import os
import time
import cv2
import numpy as np
from PIL import Image
import argparse

# 画像をASCII文字列に変換
def image_to_ascii(image: Image.Image, width: int, charset: str, contrast: float, invert: bool) -> str:
    aspect_ratio = 0.5  # 文字縦横補正
    orig_width, orig_height = image.size
    new_height = max(1, int(orig_height * (width / orig_width) * aspect_ratio))

    image = image.resize((width, new_height))
    image = image.convert("L")

    pixels = np.array(image, dtype=np.float32)
    pixels = ((pixels - 128) * contrast) + 128
    pixels = np.clip(pixels, 0, 255)

    if invert:
        pixels = 255 - pixels

    ascii_str = ""
    num_chars = len(charset)
    for row in pixels:
        for pixel in row:
            idx = int(pixel / 255 * (num_chars - 1))
            ascii_str += charset[idx]
        ascii_str += "\n"
    return ascii_str

def frame_to_ascii(frame, width: int, charset: str, contrast: float, invert: bool) -> str:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    return image_to_ascii(pil_img, width, charset, contrast, invert)

def parse_args():
    parser = argparse.ArgumentParser(description="Video to ASCII in Notepad")
    parser.add_argument("--input", required=True, help="Input video file path or camera index (0, 1, etc.)")
    parser.add_argument("--output", required=True, help="Output text file path")
    parser.add_argument("--width", type=int, default=960, help="ASCII art width in characters")
    parser.add_argument("--charset", type=str,
                        default="$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
                        help="Characters from dark to light")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast adjustment factor")
    parser.add_argument("--invert", action="store_true", help="Invert brightness mapping")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS (default: video FPS)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 入力が数字ならカメラ扱い
    try:
        cam_index = int(args.input)
        cap = cv2.VideoCapture(cam_index)
    except ValueError:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"Error: Unable to open input {args.input}")
        return

    charset = args.charset
    out_width = args.width

    # FPS設定
    fps = args.fps
    if fps is None:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = video_fps if video_fps > 0 else 15.0
    frame_interval = 1.0 / fps

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break  # 動画終端

            ascii_str = frame_to_ascii(frame, out_width, charset, args.contrast, args.invert)
            tmp_path = args.output + ".tmp"

            # 一時ファイルに書き込み
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(ascii_str)

            # Windowsメモ帳対応
            try:
                os.remove(args.output)
            except FileNotFoundError:
                pass
            except PermissionError:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(ascii_str)
                elapsed = time.time() - start_time
                time.sleep(max(0, frame_interval - elapsed))
                continue

            os.replace(tmp_path, args.output)

            # フレーム処理時間を考慮して sleep
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_interval - elapsed))

    finally:
        cap.release()
        print("\n[Done] 動画の変換が完了しました！")

if __name__ == "__main__":
    main()
