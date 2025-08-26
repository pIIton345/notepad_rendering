#!/usr/bin/env python3
# ascii_video_realtime.py
# Video -> ASCII realtime (reader + worker pool + timed writer)
# Requires: pip install opencv-python pillow numpy

import argparse
import threading
import queue
import time
import os
import sys
import cv2
import numpy as np
from PIL import Image

SENTINEL = object()

def parse_args():
    p = argparse.ArgumentParser(description="Realtime Video->ASCII with worker pool (for Notepad/VSCode)")
    p.add_argument("--input", required=True, help="Video file path or camera index (0,1,...)")
    p.add_argument("--output", required=True, help="Output text file path")
    p.add_argument("--width", type=int, default=960, help="ASCII width in characters (try 800 if slow)")
    p.add_argument("--workers", type=int, default=4, help="Number of worker threads for conversion")
    p.add_argument("--buffer", type=int, default=8, help="Max frames to buffer (reader->workers)")
    p.add_argument("--charset", type=str,
                   default="$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
                   help="Characters from dark to light")
    p.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier")
    p.add_argument("--invert", action="store_true", help="Invert brightness mapping")
    p.add_argument("--fps", type=float, default=None, help="Target FPS (default: use video FPS)")
    p.add_argument("--drop-late", type=float, default=2.0, help="Drop frame if it's this many frame_intervals late")
    return p.parse_args()

# Fast image->ascii conversion
def image_to_ascii_str(pil_img: Image.Image, width: int, charset: str, contrast: float, invert: bool) -> str:
    aspect_ratio = 0.5
    ow, oh = pil_img.size
    nh = max(1, int(oh * (width / ow) * aspect_ratio))
    img = pil_img.resize((width, nh), Image.BICUBIC).convert("L")

    arr = np.asarray(img, dtype=np.float32)
    if contrast != 1.0:
        arr = ((arr - 128.0) * contrast) + 128.0
    arr = np.clip(arr, 0, 255)
    if invert:
        arr = 255.0 - arr

    N = len(charset)
    idx = np.floor((arr / 255.0) * (N - 1)).astype(np.int32)

    charset_list = list(charset)
    lines = ["".join(charset_list[i] for i in r) for r in idx]
    return "\n".join(lines)

def frame_bgr_to_ascii(frame_bgr, width, charset, contrast, invert):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return image_to_ascii_str(pil, width, charset, contrast, invert)

# Reader thread
def reader_thread(cap, frame_queue: queue.Queue, max_buffer, stop_event, read_info):
    frame_id = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((frame_id, frame))
            frame_id += 1
    finally:
        frame_queue.put(SENTINEL)
        read_info['total_frames'] = frame_id

# Worker thread
def worker_thread(frame_queue: queue.Queue, ascii_buffer: dict, buf_lock: threading.Lock, buf_cond: threading.Condition,
                  width, charset, contrast, invert, stop_event):
    while not stop_event.is_set():
        item = frame_queue.get()
        try:
            if item is SENTINEL:
                frame_queue.put(SENTINEL)
                return
            frame_id, frame = item
            try:
                ascii_str = frame_bgr_to_ascii(frame, width, charset, contrast, invert)
            except Exception as e:
                ascii_str = f"[ERROR converting frame {frame_id}: {e}]"
            with buf_lock:
                ascii_buffer[frame_id] = ascii_str
                buf_cond.notify_all()
        finally:
            frame_queue.task_done()

# Writer thread
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
                    if expected_id >= read_info.get('total_frames', 0):
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
            expected_id += 1
            continue

        tmp_path = output_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(ascii_str)
            try:
                try:
                    os.remove(output_path)
                except FileNotFoundError:
                    pass
                except PermissionError:
                    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(ascii_str)
                    expected_id += 1
                    continue
                os.replace(tmp_path, output_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[writer error] {e}", file=sys.stderr)

        expected_id += 1

    stop_event.set()

def main():
    args = parse_args()

    try:
        idx = int(args.input)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"Error: cannot open input {args.input}", file=sys.stderr)
        return

    fps = args.fps
    if fps is None:
        vfps = cap.get(cv2.CAP_PROP_FPS)
        fps = vfps if vfps and vfps > 0 else 15.0
    frame_interval = 1.0 / fps

    frame_queue = queue.Queue(maxsize=max(2, args.buffer))
    ascii_buffer = {}
    buf_lock = threading.Lock()
    buf_cond = threading.Condition(buf_lock)
    stop_event = threading.Event()
    read_info = {'done': False, 'total_frames': None}

    reader = threading.Thread(target=reader_thread, args=(cap, frame_queue, args.buffer, stop_event, read_info), daemon=True)
    reader.start()

    workers = []
    for i in range(max(1, args.workers)):
        t = threading.Thread(target=worker_thread,
                             args=(frame_queue, ascii_buffer, buf_lock, buf_cond,
                                   args.width, args.charset, args.contrast, args.invert, stop_event), daemon=True)
        t.start()
        workers.append(t)

    start_time = time.time() + 0.1
    writer = threading.Thread(target=writer_thread,
                              args=(args.output, ascii_buffer, buf_lock, buf_cond,
                                    start_time, frame_interval, read_info, stop_event, args.drop_late), daemon=True)
    writer.start()

    try:
        # メインループ：Ctrl+C ですぐ抜ける
        while reader.is_alive() or any(w.is_alive() for w in workers) or writer.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 停止要求を受けました。終了します...")
        stop_event.set()
    finally:
        cap.release()
        reader.join(timeout=1.0)
        for w in workers:
            w.join(timeout=1.0)
        writer.join(timeout=1.0)

    print("\n[Done] 終了しました。")

if __name__ == "__main__":
    main()
