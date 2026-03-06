"""Flush, extract, and split dataset in one go."""

import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(description="Flush, extract and split frames")
parser.add_argument("--video-dir", default="data/videos/Standard")
parser.add_argument("--output-dir", default="data/frames")
parser.add_argument("--every-n", type=int, default=5)
parser.add_argument("--levels", type=int, nargs="+", default=None, help="e.g. --levels 1 2 3")
parser.add_argument("--train-ratio", type=float, default=0.9)
args = parser.parse_args()

extract_cmd = [
    "python", "scripts/extract_frames.py",
    "--video-dir", args.video_dir,
    "--output-dir", args.output_dir,
    "--every-n", str(args.every_n),
]
if args.levels:
    extract_cmd += ["--levels"] + [str(l) for l in args.levels]

split_cmd = [
    "python", "scripts/split_dataset.py",
    "--frames-dir", args.output_dir,
    "--train-ratio", str(args.train_ratio),
]

for cmd in [["python", "scripts/flush_data.py"], extract_cmd, split_cmd]:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
