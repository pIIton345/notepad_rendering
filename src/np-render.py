import os
import time
import win32gui
import win32con
import argparse

# 引数処理
parser = argparse.ArgumentParser()
parser.add_argument("file", help="Path to the text file")
parser.add_argument("--fps", type=float, default=30.0,
                    help="監視間隔 (1秒あたりの更新チェック回数, default=30)")
args = parser.parse_args()

FILE = args.file
TITLE_HINT = os.path.basename(FILE)

def find_edit_child(parent_hwnd):
    h = win32gui.FindWindowEx(parent_hwnd, 0, "Edit", None)
    if h:
        return h

    target = []
    def enum_child(hwnd, _):
        cls = win32gui.GetClassName(hwnd)
        if cls in ("Edit", "RichEditD2DPT"):
            target.append(hwnd)
    win32gui.EnumChildWindows(parent_hwnd, enum_child, None)
    return target[0] if target else None

def find_notepad_edit_for(title_hint):
    edit_hwnd = None
    def enum_top(hwnd, _):
        nonlocal edit_hwnd
        if edit_hwnd is not None:
            return
        title = win32gui.GetWindowText(hwnd)
        if title_hint.lower() in title.lower():
            e = find_edit_child(hwnd)
            if e:
                edit_hwnd = e
    win32gui.EnumWindows(enum_top, None)
    return edit_hwnd

def push_text_to_notepad(edit_hwnd, text):
    win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, text)

def main():
    edit = find_notepad_edit_for(TITLE_HINT)
    if not edit:
        print("Notepadでそのファイルを開いてから実行してください。該当ウィンドウが見つかりません。")
        return

    last_mtime = None
    interval = 1.0 / args.fps if args.fps > 0 else 0.1  # fps=0回避
    while True:
        try:
            mtime = os.path.getmtime(FILE)
            if mtime != last_mtime:
                last_mtime = mtime
                with open(FILE, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                push_text_to_notepad(edit, text)
        except Exception as e:
            print("error:", e)
        time.sleep(interval)

if __name__ == "__main__":
    main()
