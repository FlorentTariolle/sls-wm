"""Fix W&B PPO run: rescale local iterations to A100-equivalent.

Local training used 128 episodes/iter vs A100's 512.
Step 1 (--fetch): download history to JSON.
Step 2 (--upload): create new run with iteration/4 as step.

Usage:
    python scripts/fix_wandb_ppo_run.py --fetch
    python scripts/fix_wandb_ppo_run.py --upload
"""

import argparse
import json
import math
from pathlib import Path

ENTITY = "florent-tariolle-insa-rouen-normandie"
PROJECT = "deepdash"
OLD_RUN_ID = "4p8krn7k"
SCALE = 4  # 512 / 128
CACHE = Path("checkpoints/wandb_ppo_history.json")


def fetch():
    import wandb
    api = wandb.Api()
    old_run = api.run(f"{ENTITY}/{PROJECT}/{OLD_RUN_ID}")
    print(f"Fetching from: {old_run.name} ({old_run.id})")

    rows = []
    for row in old_run.scan_history(page_size=10000):
        it = row.get("iteration")
        if it is not None and it % SCALE == 0:
            clean = {}
            for k, v in row.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                clean[k] = v
            clean["iteration"] = it // SCALE
            rows.append(clean)

    CACHE.write_text(json.dumps(rows))
    print(f"Saved {len(rows)} rows to {CACHE}")
    print(f"Old run config: {json.dumps(dict(old_run.config), indent=2)[:500]}")
    # Save config too
    Path("checkpoints/wandb_ppo_config.json").write_text(json.dumps(dict(old_run.config)))


def upload():
    import wandb
    rows = json.loads(CACHE.read_text())
    config = json.loads(Path("checkpoints/wandb_ppo_config.json").read_text())
    print(f"Uploading {len(rows)} rows...")

    run = wandb.init(
        project=PROJECT,
        name="ppo-512d",
        config=config,
        notes=f"Rescaled from {OLD_RUN_ID} (local 128ep -> A100-equiv 512ep)",
        mode="offline",
    )
    for i, row in enumerate(rows):
        wandb.log(row)
        if (i + 1) % 1000 == 0:
            print(f"  logged {i + 1}/{len(rows)}")
    wandb.finish()
    print(f"\nOffline run saved to: {run.dir}")
    print(f"Sync with: wandb sync {run.dir}")
    print(f"Then delete old run: https://wandb.ai/{ENTITY}/{PROJECT}/runs/{OLD_RUN_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    elif args.upload:
        upload()
    else:
        print("Use --fetch or --upload")
