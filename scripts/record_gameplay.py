"""Record Geometry Dash gameplay: screen capture + keyboard input + death detection.

Reads game memory to auto-detect death and split episodes automatically.
Just press F5 once to start, play the game, and episodes are saved on each death.

Usage:
    python scripts/record_gameplay.py --monitor-top 0 --monitor-left 0

Controls:
    F5  — start/stop recording
    F6  — manual episode split (saves current, starts fresh)
    ESC — quit (saves current episode if recording)

Output:
    data/episodes/ep_NNNN/
        frames.npy     (T, 64, 64) uint8 — Sobel edge maps
        actions.npy    (T,) uint8 — 0=idle, 1=jump
        metadata.json  {fps_target, fps_actual, timestamp, num_frames}
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import ctypes
import cv2
import keyboard
import mss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deepdash.gd_mem import GDReader


import ctypes.wintypes as wt

# Win32 always-on-top helper with proper 64-bit type annotations
_user32 = ctypes.windll.user32
_user32.FindWindowW.argtypes = [wt.LPCWSTR, wt.LPCWSTR]
_user32.FindWindowW.restype = wt.HWND
_user32.SetWindowPos.argtypes = [
    wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_uint,
]
_user32.SetWindowPos.restype = wt.BOOL
_HWND_TOPMOST = wt.HWND(-1)
_SWP_FLAGS = 0x0002 | 0x0001 | 0x0010  # NOMOVE | NOSIZE | NOACTIVATE
_cached_hwnd = None


def _force_topmost(window_title: str):
    """Force a window to stay on top using Win32 SetWindowPos."""
    global _cached_hwnd
    if _cached_hwnd is None:
        _cached_hwnd = _user32.FindWindowW(None, window_title)
    if _cached_hwnd:
        _user32.SetWindowPos(
            _cached_hwnd, _HWND_TOPMOST, 0, 0, 0, 0, _SWP_FLAGS)


def preprocess_frame(bgra: np.ndarray, crop_x: int, crop_y: int,
                     crop_size: int, target_size: int) -> np.ndarray:
    """BGRA screenshot -> 64x64 Sobel edge map (uint8).

    1. Crop crop_size x crop_size at (crop_x, crop_y)
    2. Sobel edge detection at full resolution (1032x1032 for 1080p)
    3. Resize to target_size x target_size with INTER_AREA
    """
    bgr = bgra[:, :, :3]
    cropped = bgr[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))
    resized = cv2.resize(edges, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)
    return resized


def save_episode(episode_dir: Path, frames: list, actions: list,
                 fps_target: int, t_start: float, t_end: float):
    """Save a single episode to disk."""
    if not frames:
        return
    episode_dir.mkdir(parents=True, exist_ok=True)
    np.save(episode_dir / "frames.npy", np.array(frames, dtype=np.uint8))
    np.save(episode_dir / "actions.npy", np.array(actions, dtype=np.uint8))
    num_frames = len(frames)
    duration = t_end - t_start
    fps_actual = num_frames / duration if duration > 0 else 0
    metadata = {
        "fps_target": fps_target,
        "fps_actual": round(fps_actual, 2),
        "num_frames": num_frames,
        "duration_s": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
    }
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {episode_dir.name}: {num_frames} frames, "
          f"{duration:.1f}s, {fps_actual:.1f} FPS actual")


def next_episode_id(episodes_dir: Path) -> int:
    """Find the next available episode number."""
    existing = sorted(episodes_dir.glob("ep_*"))
    if not existing:
        return 1
    last = existing[-1].name
    return int(last.split("_")[1]) + 1


def main():
    parser = argparse.ArgumentParser(
        description="Record Geometry Dash gameplay for training data")
    parser.add_argument("--monitor-top", type=int, default=0,
                        help="Top-left Y of the game window on screen")
    parser.add_argument("--monitor-left", type=int, default=0,
                        help="Top-left X of the game window on screen")
    parser.add_argument("--window-width", type=int, default=1920,
                        help="Game window width (default: 1920)")
    parser.add_argument("--window-height", type=int, default=1080,
                        help="Game window height (default: 1080)")
    parser.add_argument("--crop-x", type=int, default=660,
                        help="Crop X offset within captured window (default: 660)")
    parser.add_argument("--crop-y", type=int, default=48,
                        help="Crop Y offset within captured window (default: 48)")
    parser.add_argument("--crop-size", type=int, default=1032,
                        help="Crop square size (default: 1032)")
    parser.add_argument("--target-size", type=int, default=64,
                        help="Output frame size (default: 64)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Target capture FPS (default: 60)")
    parser.add_argument("--jump-key", default="space",
                        help="Key used for jumping (default: space)")
    parser.add_argument("--output-dir", default="data/episodes",
                        help="Output directory for episodes")
    args = parser.parse_args()

    # Connect to GD process for death detection
    print("Connecting to Geometry Dash process...")
    try:
        gd = GDReader()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Make sure Geometry Dash is running.")
        return

    episodes_dir = Path(args.output_dir)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_id = next_episode_id(episodes_dir)

    monitor = {
        "top": args.monitor_top,
        "left": args.monitor_left,
        "width": args.window_width,
        "height": args.window_height,
    }

    frame_interval = 1.0 / args.fps
    recording = False
    frames = []
    actions = []
    t_episode_start = 0.0
    was_dead = False

    print(f"Capture region: {monitor}")
    print(f"Target FPS: {args.fps}")
    print(f"Jump key: {args.jump_key}")
    print(f"Episodes dir: {episodes_dir}")
    print()
    print("Controls:")
    print("  F5  — start/stop recording")
    print("  F6  — manual episode split")
    print("  ESC — quit")
    print()
    print("Death detection: ON (auto-splits episodes on death)")
    print("Preview window open — verify capture region is correct.")

    with mss.mss() as sct:
        while True:
            t_loop_start = time.perf_counter()

            # Capture screen
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)

            # Preprocess
            edge_frame = preprocess_frame(
                img, args.crop_x, args.crop_y,
                args.crop_size, args.target_size)

            # Show preview
            preview = cv2.resize(edge_frame, (256, 256),
                                 interpolation=cv2.INTER_NEAREST)
            status = "REC" if recording else "IDLE"
            color = (0, 0, 255) if recording else (128, 128, 128)
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            cv2.putText(preview_bgr, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if recording:
                cv2.putText(preview_bgr,
                            f"ep_{episode_id:04d}  {len(frames)} frames",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
            cv2.imshow("DeepDash Recorder", preview_bgr)

            # Force always-on-top every frame via Win32 API
            # (cv2.setWindowProperty alone doesn't beat fullscreen games)
            _force_topmost("DeepDash Recorder")

            # Record frame + action if recording
            if recording:
                jump = keyboard.is_pressed(args.jump_key)
                frames.append(edge_frame)
                actions.append(1 if jump else 0)

            # Death detection — auto-split episodes
            if recording:
                state = gd.get_state()
                is_dead = state["is_dead"]
                if is_dead and not was_dead and frames:
                    # Player just died — save episode
                    t_end = time.perf_counter()
                    ep_dir = episodes_dir / f"ep_{episode_id:04d}"
                    save_episode(ep_dir, frames, actions,
                                 args.fps, t_episode_start, t_end)
                    episode_id += 1
                    frames = []
                    actions = []
                    t_episode_start = time.perf_counter()
                    print(f">> Death detected. New episode: ep_{episode_id:04d}")
                was_dead = is_dead

            # Handle OpenCV key events (1ms wait)
            key = cv2.waitKey(1) & 0xFF

            # Hotkeys
            if keyboard.is_pressed("f5"):
                if not recording:
                    recording = True
                    frames = []
                    actions = []
                    was_dead = False
                    t_episode_start = time.perf_counter()
                    print(f"\n>> Recording ep_{episode_id:04d}...")
                else:
                    recording = False
                    t_end = time.perf_counter()
                    ep_dir = episodes_dir / f"ep_{episode_id:04d}"
                    save_episode(ep_dir, frames, actions,
                                 args.fps, t_episode_start, t_end)
                    episode_id += 1
                    frames = []
                    actions = []
                    print(">> Stopped recording.")
                while keyboard.is_pressed("f5"):
                    time.sleep(0.01)

            if keyboard.is_pressed("f6"):
                if recording and frames:
                    t_end = time.perf_counter()
                    ep_dir = episodes_dir / f"ep_{episode_id:04d}"
                    save_episode(ep_dir, frames, actions,
                                 args.fps, t_episode_start, t_end)
                    episode_id += 1
                    frames = []
                    actions = []
                    t_episode_start = time.perf_counter()
                    print(f"\n>> New episode: ep_{episode_id:04d}...")
                while keyboard.is_pressed("f6"):
                    time.sleep(0.01)

            if keyboard.is_pressed("escape") or key == 27:
                if recording and frames:
                    t_end = time.perf_counter()
                    ep_dir = episodes_dir / f"ep_{episode_id:04d}"
                    save_episode(ep_dir, frames, actions,
                                 args.fps, t_episode_start, t_end)
                    print(">> Saved final episode on exit.")
                break

            # Frame rate limiting
            elapsed = time.perf_counter() - t_loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    gd.close()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
