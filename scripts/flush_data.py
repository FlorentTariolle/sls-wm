"""Delete all extracted frames and train/val splits."""

import shutil
from pathlib import Path

for d in ["data/frames", "data/train", "data/val"]:
    shutil.rmtree(d, ignore_errors=True)
    Path(d).mkdir(parents=True)
    print(f"Cleared {d}/")
