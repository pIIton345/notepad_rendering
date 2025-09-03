#!/usr/bin/env python3
# ascii_mixed_color_fast.py
# Fast mixed Camera/Video -> ASCII (grayscale or color)
# Requires: pip install opencv-python numpy
# Notes:
#  - Use smaller --width for faster results.
#  - Default resize algo is bilinear (faster than bicubic).
#  - Color mapping uses squared-distance and vectorized argmin.

import argparse
import threading
import queue
import time
import os
import sys
import cv2
import numpy as np

SENTINEL = object()

def parse_args():
    p = argparse.ArgumentParser(description="Fast Mixed Camera/Video -> ASCII (grayscale or color)")
    p.add_argument("--input", required=True, help="Video file path or camera index (0,1,...)")
    p.add_argument("--output", required=True, help="Output text file path")
    p.add_argument("--width", type=int, default=480, help="ASCII width in characters (smaller -> faster)")
    p.add_argument("--workers", type=int, default=4, help="Number of worker threads for conversion (video only)")
    p.add_argument("--buffer", type=int, default=8, help="Max frames to buffer (reader->workers)")
    p.add_argument("--charset", type=str,
                   default="$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
                   help="Characters from dark to light (grayscale only)")
    p.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier (grayscale only)")
    p.add_argument("--invert", action="store_true", help="Invert brightness mapping (grayscale only)")
    p.add_argument("--fps", type=float, default=None, help="Target FPS (default: use video FPS)")
    p.add_argument("--drop-late", type=float, default=2.0, help="Drop frame if late (video only)")
    p.add_argument("--color", action="store_true", help="Enable color-based ASCII")
    p.add_argument("--resize-algo", choices=["bilinear", "bicubic", "nearest"], default="bilinear",
                   help="Resize algorithm (bilinear faster than bicubic)")
    p.add_argument("--brightness", type=float, default=1.0, help="Input brightness multiplier (applied before conversion)")
    # New options for white-preserving adaptive brightness
    p.add_argument("--auto", action="store_true", help="Preserve highlights when increasing brightness (adaptive per-pixel; prevents white clipping)")
    p.add_argument("--auto-gamma", type=float, default=1.0, help="Gamma for --auto curve (0.5..2.0 typical). Lower -> stronger lift for darkest pixels")
    p.add_argument("--aspect", type=float, default=0.5,
                   help="character aspect ratio (height/width). >0.5 makes characters taller, <0.5 squashes vertically. default=0.5")
    return p.parse_args()

# ---- helpers for cv2 <-> ascii ----
CV_INTERPOL = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}

def build_char_palette(char_colors):
    # char_array: numpy array of dtype '<U1' (one unicode char each)
    chars, cols = zip(*char_colors)
    # ensure each char is length 1 in codepoints; if multi-codepoint entries exist they'd still be preserved
    char_array = np.array([c for c in chars], dtype=np.dtype('U'))  # variable-length unicode
    color_array = np.array(cols, dtype=np.float32)  # shape (n_colors, 3)
    return char_array, color_array

def gray_array_to_ascii_lines(arr_uint8, charset, contrast, invert):
    # arr_uint8: 2D uint8 grayscale image (h, w)
    arr = arr_uint8.astype(np.float32)
    if contrast != 1.0:
        arr = ((arr - 128.0) * contrast) + 128.0
    arr = np.clip(arr, 0, 255)
    if invert:
        arr = 255.0 - arr
    N = len(charset)
    # index 0..N-1
    idx = np.floor((arr / 255.0) * (N - 1)).astype(np.int32)
    # vectorized mapping to chars
    charset_arr = np.array(list(charset), dtype=np.dtype('U'))
    mapped = charset_arr[idx]  # shape (h, w)
    # join rows to strings
    lines = ["".join(row.tolist()) for row in mapped]
    return "\n".join(lines)

def color_array_to_ascii_lines(img_rgb_float, char_array, color_array):
    # img_rgb_float: (h, w, 3) float32, 0..255
    # color_array: (n, 3) float32
    # Compute squared distances vectorized across image:
    # result shape -> (h, w, n)
    # To reduce temporary peak memory, we compute with broadcasting but be aware of memory for very large sizes.
    h, w, _ = img_rgb_float.shape
    # Broadcasting: (h, w, 1, 3) - (1, 1, n, 3) -> (h, w, n, 3)
    diff = img_rgb_float[:, :, None, :] - color_array[None, None, :, :]  # (h, w, n, 3)
    dist2 = np.einsum('hwna,hwna->hwn', diff, diff)  # squared distances, shape (h, w, n)
    idx = np.argmin(dist2, axis=2)  # (h, w)
    # map idx to characters
    mapped = char_array[idx]  # (h, w) dtype '<U'
    lines = ["".join(row.tolist()) for row in mapped]
    return "\n".join(lines)

# ---- conversion functions that use cv2 (fast) ----
def frame_bgr_to_ascii_fast(frame_bgr, width, charset, contrast, invert, use_color,
                            char_array=None, color_array=None, resize_inter=cv2.INTER_LINEAR,
                            brightness=1.0, auto=False, auto_gamma=1.0, aspect=0.5):
    # frame_bgr: cv2 BGR image (H, W, 3) uint8
    # brightness: multiplier applied to input frame before any conversion
    # auto: if True and brightness>1.0, apply per-pixel adaptive factor to protect highlights
    # aspect: character height/width ratio
    h0, w0 = frame_bgr.shape[:2]

    # --- 明度補正（白飛び抑制オプション --auto） ---
    if brightness != 1.0:
        try:
            if auto and brightness > 1.0:
                # Lab の L チャネルを使って画素ごとに倍率を計算（ハイライト保護）
                # OpenCV Lab: L in 0..255
                lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                L = lab[:, :, 0]  # 0..255
                L_norm = np.clip(L / 255.0, 0.0, 1.0)
                # factor: 1.0 .. brightness  (L=255 -> 1, L=0 -> brightness)
                # 使用式: factor = 1 + (brightness-1) * (1 - L_norm^gamma)
                factor = 1.0 + (float(brightness) - 1.0) * (1.0 - np.power(L_norm, float(auto_gamma)))
                # expand and apply per-channel
                factor = factor[:, :, None]  # (H,W,1)
                frame_bgr = np.clip(frame_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            else:
                # 通常の一括倍率（既存の挙動）
                frame_bgr = np.clip(frame_bgr.astype(np.float32) * float(brightness), 0, 255).astype(np.uint8)
        except Exception:
            # 安全フォールバック
            frame_bgr = cv2.convertScaleAbs(frame_bgr, alpha=float(brightness), beta=0)

    # use provided aspect (character height/width)
    nh = max(1, int(h0 * (width / w0) * float(aspect)))
    # resize using cv2 (fast)
    small = cv2.resize(frame_bgr, (width, nh), interpolation=resize_inter)
    if use_color:
        # convert to RGB float32
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        return color_array_to_ascii_lines(rgb, char_array, color_array)
    else:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)  # uint8
        return gray_array_to_ascii_lines(gray, charset, contrast, invert)

# ---- reader / worker / writer for video files ----
def reader_thread(cap, frame_queue: queue.Queue, stop_event, read_info):
    frame_id = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((frame_id, frame))
            frame_id += 1
    finally:
        read_info['total_frames'] = frame_id
        read_info['done'] = True
        frame_queue.put(SENTINEL)

def worker_thread(frame_queue: queue.Queue, ascii_buffer: dict, buf_lock: threading.Lock, buf_cond: threading.Condition,
                  width, charset, contrast, invert, stop_event, use_color, char_array, color_array, resize_inter, brightness, auto, auto_gamma, aspect):
    while not stop_event.is_set():
        item = frame_queue.get()
        try:
            if item is SENTINEL:
                frame_queue.put(SENTINEL)
                return
            frame_id, frame = item
            try:
                ascii_str = frame_bgr_to_ascii_fast(frame, width, charset, contrast, invert,
                                                    use_color, char_array, color_array, resize_inter,
                                                    brightness, auto, auto_gamma, aspect)
            except Exception as e:
                print(f"[worker error] frame {frame_id}: {e}", file=sys.stderr)
                ascii_str = f"[ERROR converting frame {frame_id}: {e}]"
            with buf_lock:
                ascii_buffer[frame_id] = ascii_str
                buf_cond.notify_all()
        finally:
            frame_queue.task_done()

def writer_thread(output_path, ascii_buffer: dict, buf_lock: threading.Lock, buf_cond: threading.Condition,
                  start_time, frame_interval, read_info, stop_event, drop_late_factor):
    expected_id = 0
    drop_threshold = frame_interval * drop_late_factor
    while not stop_event.is_set():
        with buf_lock:
            if expected_id in ascii_buffer:
                ascii_str = ascii_buffer.pop(expected_id)
            else:
                if read_info.get('done', False):
                    total = read_info.get('total_frames', 0)
                    if expected_id >= total:
                        break
                buf_cond.wait(timeout=0.05)
                continue

        target_time = start_time + expected_id * frame_interval
        now = time.time()
        if now < target_time:
            time.sleep(max(0, target_time - now))

        now = time.time()
        lateness = now - target_time
        if lateness > drop_threshold:
            # skip but keep going
            print(f"[writer] frame {expected_id} skipped due to lateness {lateness:.3f}")
            expected_id += 1
            continue

        tmp_path = output_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(ascii_str)
            try:
                os.remove(output_path)
            except FileNotFoundError:
                pass
            except PermissionError:
                # fallback: overwrite file (some editors/OS may lock replace)
                with open(output_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(ascii_str)
                expected_id += 1
                continue
            os.replace(tmp_path, output_path)
        except Exception as e:
            print(f"[writer error] {e}", file=sys.stderr)

        expected_id += 1

    stop_event.set()

# ---- camera realtime loop ----
def camera_loop(cap, args, charset, out_width, frame_interval, char_array, color_array, resize_inter, aspect):
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            ascii_str = frame_bgr_to_ascii_fast(frame, out_width, charset, args.contrast, args.invert,
                                               args.color, char_array, color_array, resize_inter,
                                               args.brightness, args.auto, args.auto_gamma, aspect)
            tmp_path = args.output + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(ascii_str)
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
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_interval - elapsed))
    finally:
        cap.release()

def main():
    args = parse_args()

    # input open
    is_camera = False
    cap = None
    try:
        idx = int(args.input)
        cap = cv2.VideoCapture(idx)
        is_camera = True
    except ValueError:
        cap = cv2.VideoCapture(args.input)

    if not cap or not cap.isOpened():
        print(f"Error: cannot open input {args.input}", file=sys.stderr)
        return

    fps = args.fps
    if fps is None:
        vfps = cap.get(cv2.CAP_PROP_FPS)
        fps = vfps if (vfps and vfps > 0) else 15.0
    frame_interval = 1.0 / fps

    charset = args.charset
    out_width = args.width
    resize_inter = CV_INTERPOL.get(args.resize_algo, cv2.INTER_LINEAR)

    # color mapping table (user can modify)
    char_colors = [
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

    char_array, color_array = build_char_palette(char_colors)

    if is_camera:
        print("[mode] camera realtime")
        try:
            camera_loop(cap, args, charset, out_width, frame_interval, char_array, color_array, resize_inter, args.aspect)
        except KeyboardInterrupt:
            pass
        print("\n[Done] camera stopped.")
        return

    print("[mode] video fast pipeline")
    frame_queue = queue.Queue(maxsize=max(2, args.buffer))
    ascii_buffer = {}
    buf_lock = threading.Lock()
    buf_cond = threading.Condition(buf_lock)
    stop_event = threading.Event()
    read_info = {'done': False, 'total_frames': None}

    reader = threading.Thread(target=reader_thread, args=(cap, frame_queue, stop_event, read_info), daemon=True)
    reader.start()

    workers = []
    for i in range(max(1, args.workers)):
        t = threading.Thread(target=worker_thread,
                             args=(frame_queue, ascii_buffer, buf_lock, buf_cond,
                                   out_width, charset, args.contrast, args.invert,
                                   stop_event, args.color, char_array, color_array, resize_inter,
                                   args.brightness, args.auto, args.auto_gamma, args.aspect),
                             daemon=True)
        t.start()
        workers.append(t)

    start_time = time.time() + 0.01
    writer = threading.Thread(target=writer_thread,
                              args=(args.output, ascii_buffer, buf_lock, buf_cond,
                                    start_time, frame_interval, read_info, stop_event, args.drop_late),
                              daemon=True)
    writer.start()

    try:
        while reader.is_alive() or any(w.is_alive() for w in workers) or writer.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] stopping...")
        stop_event.set()
    finally:
        stop_event.set()
        cap.release()
        reader.join(timeout=1.0)
        for w in workers:
            w.join(timeout=1.0)
        writer.join(timeout=1.0)

    print("\n[Done] video processing finished.")

if __name__ == "__main__":
    main()
