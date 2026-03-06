"""Open eval output samples in a grid using matplotlib."""

import glob
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="eval_output")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.output_dir}/sample_*.png"))
if not files:
    print(f"No samples found in {args.output_dir}/")
    raise SystemExit(1)

n = len(files)
cols = min(4, n)
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = [axes] if n == 1 else axes.flat

for ax, f in zip(axes, files):
    ax.imshow(mpimg.imread(f))
    ax.set_title(f.split("/")[-1].split("\\")[-1], fontsize=8)
    ax.axis("off")

for ax in list(axes)[n:]:
    ax.axis("off")

plt.tight_layout()
plt.show()
