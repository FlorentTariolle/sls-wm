"""Pre-compute spatially shifted episode variants for data augmentation.

Creates shifted copies of episode frames with edge padding.
Actions are symlinked (or copied on Windows) from the base episode.

Usage:
    python scripts/shift_episodes.py
    python scripts/shift_episodes.py --episodes-dir data/death_episodes --shifts-v -4 -2 0 2 4
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np


def _shift_frames(frames, dx, dy):
    """Shift frames by (dx, dy) pixels with edge padding.

    dx>0 shifts content right (new pixels appear on left edge).
    dy>0 shifts content down (new pixels appear on top edge).
    """
    if dx == 0 and dy == 0:
        return frames
    shifted = np.roll(np.roll(frames, dx, axis=-1), dy, axis=-2)
    if dx > 0:
        shifted[..., :, :dx] = frames[..., :, :1]
    elif dx < 0:
        shifted[..., :, dx:] = frames[..., :, -1:]
    if dy > 0:
        shifted[..., :dy, :] = frames[..., :1, :]
    elif dy < 0:
        shifted[..., dy:, :] = frames[..., -1:, :]
    return shifted


def process_directory(episodes_dir, shifts_v):
    """Process a single episodes directory, creating shift variants."""
    episodes_dir = Path(episodes_dir)
    if not episodes_dir.exists():
        print(f"  Skipping {episodes_dir} (does not exist)")
        return 0, 0, 0

    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")

    # Find base episodes (no shift suffix)
    base_episodes = sorted(
        ep for ep in episodes_dir.glob("*")
        if ep.is_dir() and (ep / "frames.npy").exists()
        and not shift_re.search(ep.name)
    )

    if not base_episodes:
        print(f"  No base episodes in {episodes_dir}")
        return 0, 0, 0

    # Delete ALL existing shift directories first
    deleted = 0
    for ep in episodes_dir.glob("*"):
        if ep.is_dir() and shift_re.search(ep.name):
            shutil.rmtree(ep)
            deleted += 1
    if deleted:
        print(f"  Deleted {deleted} existing shift directories")

    # Build shift combos: horizontal is always 0, vertical from --shifts-v
    # Skip (0, 0) since that's the base episode
    aug_shifts = [(0, dy) for dy in shifts_v if dy != 0]

    created = 0
    for ep in base_episodes:
        frames = np.load(ep / "frames.npy")

        for dx, dy in aug_shifts:
            shifted = _shift_frames(frames, dx, dy)
            aug_name = f"{ep.name}_s{dx:+d}_{dy:+d}"
            aug_dir = episodes_dir / aug_name
            aug_dir.mkdir(exist_ok=True)

            np.save(aug_dir / "frames.npy", shifted)

            # Link actions.npy from base episode
            dst = aug_dir / "actions.npy"
            if not dst.exists():
                src = (ep / "actions.npy").resolve()
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

            created += 1

    return len(base_episodes), created, deleted


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute shifted episode variants for augmentation")
    parser.add_argument("--episodes-dir", default="data/death_episodes",
                        help="Death episodes directory")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes",
                        help="Expert episodes directory")
    parser.add_argument("--shifts-v", type=int, nargs="+", default=[-4, -2, 0, 2, 4],
                        help="Vertical pixel shifts (default: -4 -2 0 2 4)")
    args = parser.parse_args()

    total_base = 0
    total_created = 0

    for ep_dir in [args.episodes_dir, args.expert_episodes_dir]:
        print(f"Processing {ep_dir}:")
        n_base, n_created, _ = process_directory(ep_dir, args.shifts_v)
        total_base += n_base
        total_created += n_created

    print(f"\nDone. {total_base} base episodes, {total_created} shift variants created.")


if __name__ == "__main__":
    main()
