import argparse
import threading
import queue
import time
import os
import sys
import cv2
import numpy as np
import random

# VizDoom import check
try:
    import vizdoom as vzd
    HAS_VIZDOOM = True
except ImportError:
    HAS_VIZDOOM = False

SENTINEL = object()

def parse_args():
    p = argparse.ArgumentParser(description="Fast Mixed Doom -> ASCII")
    p.add_argument("--wad", required=True,help="input WAD file path")
    p.add_argument("--output", required=True, help="output text file path")
    p.add_argument("--width", type=int, default=480, help="output width in characters (smaller -> faster)")
    p.add_argument("--workers", type=int, default=4, help="Number of worker threads for conversion (video only)")
    p.add_argument("--buffer", type=int, default=8, help="Max frames to buffer (reader->workers)")
    p.add_argument("--charset", type=str,
                   default="$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
                   help="Characters from dark to light (grayscale only)")
    p.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier")
    p.add_argument("--invert", action="store_true", help="Invert brightness mapping")
    p.add_argument("--fps", type=float, default=None, help="Target FPS (default: use doom FPS)")
    p.add_argument("--drop-late", type=float, default=2.0, help="Drop frame if late")
    p.add_argument("--color", action="store_true", help="Enable color mode")
    p.add_argument("--resize-algo", choices=["bilinear", "bicubic", "nearest"], default="bilinear")
    p.add_argument("--brightness", type=float, default=1.0, help="brightness multiplier (default=1.0)")
    p.add_argument("--auto", action="store_true", help="Preserve highlights when increasing brightness (adaptive per-pixel; prevents white clipping)")
    p.add_argument("--auto-gamma", type=float, default=1.0, help="Gamma for --auto curve (0.5..2.0 typical). Lower -> stronger lift for darkest pixels")
    p.add_argument("--aspect", type=float, default=0.5, help="character aspect ratio (height/width). >0.5 makes characters taller, <0.5 squashes vertically. default=0.5")
    return p.parse_args()

# ---- helpers for cv2 <-> ascii ----
CV_INTERPOL = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}

def build_char_palette(char_colors):
    chars, cols = zip(*char_colors)
    char_array = np.array([c for c in chars], dtype=np.dtype('U'))
    color_array = np.array(cols, dtype=np.float32)
    return char_array, color_array

def gray_array_to_ascii_lines(arr_uint8, charset, contrast, invert):
    arr = arr_uint8.astype(np.float32)
    if contrast != 1.0:
        arr = ((arr - 128.0) * contrast) + 128.0
    arr = np.clip(arr, 0, 255)
    if invert:
        arr = 255.0 - arr
    N = len(charset)
    idx = np.floor((arr / 255.0) * (N - 1)).astype(np.int32)
    charset_arr = np.array(list(charset), dtype=np.dtype('U'))
    mapped = charset_arr[idx]
    lines = ["".join(row.tolist()) for row in mapped]
    return "\n".join(lines)

def color_array_to_ascii_lines(img_rgb_float, char_array, color_array):
    h, w, _ = img_rgb_float.shape
    diff = img_rgb_float[:, :, None, :] - color_array[None, None, :, :]
    dist2 = np.einsum('hwna,hwna->hwn', diff, diff)
    idx = np.argmin(dist2, axis=2)
    mapped = char_array[idx]
    lines = ["".join(row.tolist()) for row in mapped]
    return "\n".join(lines)

# ---- conversion functions ----
def frame_bgr_to_ascii_fast(frame_bgr, width, charset, contrast, invert, use_color,
                            char_array=None, color_array=None, resize_inter=cv2.INTER_LINEAR,
                            brightness=1.0, auto=False, auto_gamma=1.0, aspect=0.5):
    h0, w0 = frame_bgr.shape[:2]

    if brightness != 1.0:
        try:
            if auto and brightness > 1.0:
                lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                L = lab[:, :, 0]
                L_norm = np.clip(L / 255.0, 0.0, 1.0)
                factor = 1.0 + (float(brightness) - 1.0) * (1.0 - np.power(L_norm, float(auto_gamma)))
                factor = factor[:, :, None]
                frame_bgr = np.clip(frame_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            else:
                frame_bgr = np.clip(frame_bgr.astype(np.float32) * float(brightness), 0, 255).astype(np.uint8)
        except Exception:
            frame_bgr = cv2.convertScaleAbs(frame_bgr, alpha=float(brightness), beta=0)

    nh = max(1, int(h0 * (width / w0) * float(aspect)))
    small = cv2.resize(frame_bgr, (width, nh), interpolation=resize_inter)
    
    if use_color:
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        return color_array_to_ascii_lines(rgb, char_array, color_array)
    else:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray_array_to_ascii_lines(gray, charset, contrast, invert)

# ---- Video/Camera Threads (Existing Logic) ----
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
                ascii_str = ""
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
                pass
            os.replace(tmp_path, output_path)
        except Exception:
            pass
        expected_id += 1
    stop_event.set()

# ---- Camera Loop ----
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
                os.replace(tmp_path, args.output)
            except OSError:
                pass
            
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_interval - elapsed))
    finally:
        cap.release()

# ---- Doom Loop (New) ----
def init_doom(wad_path):
    print(f"[Doom] Initializing VizDoom with {wad_path}...")
    game = vzd.DoomGame()
        # 変更箇所：init_doom の冒頭付近
    # wad_path が IWAD か PWAD かを判定して適切な API を呼ぶ
    with open(wad_path, "rb") as f:
        magic = f.read(4).decode("ascii", errors="ignore").upper()

    if magic == "IWAD":
        # 本体 IWAD を指定（DOOM / DOOM2 / etc）
        game.set_doom_game_path(wad_path)
    elif magic == "PWAD":
        # シナリオ / mod 用の指定
        game.set_doom_scenario_path(wad_path)
    else:
        # 不明なら従来互換でシナリオとして渡す（もしくはエラーにする）
        game.set_doom_scenario_path(wad_path)


    # wad_type 判定の後にマップを設定（上の IWAD 判定の直後あたり）
    if magic == "IWAD":
        lower_name = os.path.basename(wad_path).lower()
        # 簡易判定：ファイル名に 'doom2' が含まれるなら Doom2
        if "doom2" in lower_name:
            game.set_doom_map("MAP01")   # Doom II の 1 面
        else:
            # Doom 1 系（DOOM.WAD 等）は E1M1 を指定
            game.set_doom_map("E1M1")
    else:    
    # PWAD 等は map 名を固定しない・または config から渡す
    # game.set_doom_map("MAP01")
        pass

    # Set resolution low for performance (we resize anyway)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    # Use BGR24 to match OpenCV format
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    
    # Enable rendering features
    game.set_render_hud(True)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(True)
    game.set_render_particles(True)
    
    # Buttons available in the window
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)
    game.add_available_button(vzd.Button.USE)

    # Show window to capture input? 
    # VizDoom default window is needed to capture keyboard/mouse easily.
    # If false, you need to implement your own input listener (e.g. keyboard lib).
    game.set_window_visible(True) 
    
    # Set to SPECTATOR to allow human input via window
    game.set_mode(vzd.Mode.SPECTATOR)
    
    game.init()
    return game

def doom_loop(game, args, charset, out_width, frame_interval, char_array, color_array, resize_inter, aspect):
    print("[Doom] Game started. Use the VizDoom window for input, watch the ASCII file for output.")
    try:
        while not game.is_episode_finished():
            start_time = time.time()
            
            # Advance game state (input is handled by VizDoom internal window automatically in SPECTATOR mode)
            game.advance_action()
            
            if game.is_player_dead():
                game.respawn_player()

            state = game.get_state()
            if state is not None:
                # Buffer is (Height, Width, 3) in BGR
                frame = state.screen_buffer
                
                ascii_str = frame_bgr_to_ascii_fast(frame, out_width, charset, args.contrast, args.invert,
                                                    args.color, char_array, color_array, resize_inter,
                                                    args.brightness, args.auto, args.auto_gamma, aspect)
                
                tmp_path = args.output + ".tmp"
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(ascii_str)
                    os.replace(tmp_path, args.output)
                except OSError:
                    pass

            elapsed = time.time() - start_time
            # Cap FPS
            time.sleep(max(0, frame_interval - elapsed))
            
    except Exception as e:
        print(f"Doom Error: {e}")
    finally:
        game.close()

def main():
    args = parse_args()

    # Color palette definition
    char_colors = [
        (' ', (20, 20, 20)), # Dark BG
        ('.', (60, 60, 60)),
        (':', (100, 100, 100)),
        ('-', (120, 120, 120)),
        ('=', (150, 150, 150)),
        ('+', (180, 180, 180)),
        ('*', (200, 200, 200)),
        ('#', (220, 220, 220)),
        ('%', (240, 240, 240)),
        ('@', (255, 255, 255)),
        # Specific Doom-ish colors (Red for blood/imps, Green for toxic/armor, Blue for health)
        ('▒', (50, 50, 180)),   # Reddish
        ('▓', (50, 180, 50)),   # Greenish
        ('█', (180, 50, 50)),   # Blueish (BGR)
    ]
    # NOTE: For better color ASCII, the palette above should be expanded significantly 
    # or generated dynamically, but we use the script's default method.
    if args.color:
         # Fallback to a richer default palette if user didn't customize
         pass 

    # Re-use existing palette builder from script
    # (Simulated here for brevity, using the one from original script logic)
    full_char_colors = [
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
    char_array, color_array = build_char_palette(full_char_colors)

    resize_inter = CV_INTERPOL.get(args.resize_algo, cv2.INTER_LINEAR)
    
    # ---- Mode Selection ----
    if args.wad:
        if not HAS_VIZDOOM:
            print("Error: vizdoom module not found. Install with 'pip install vizdoom'", file=sys.stderr)
            return
        game = init_doom(args.wad)
        fps = args.fps if args.fps else 35.0
        frame_interval = 1.0 / fps
        try:
            doom_loop(game, args, args.charset, args.width, frame_interval, char_array, color_array, resize_inter, args.aspect)
        except KeyboardInterrupt:
            pass
        print("\n[Done] Doom stopped.")
        return

    # Default to Video/Camera logic if no --wad
    if not args.input:
        print("Error: Must provide --input (video/camera) OR --wad (doom)", file=sys.stderr)
        return

    # ... (Existing Camera/Video logic continues below, unchanged mostly) ...
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

    if is_camera:
        print("[mode] camera realtime")
        try:
            camera_loop(cap, args, args.charset, args.width, frame_interval, char_array, color_array, resize_inter, args.aspect)
        except KeyboardInterrupt:
            pass
        return

    # Video File Mode
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
                                   args.width, args.charset, args.contrast, args.invert,
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