"""Encode episode frames through frozen tokenizer to produce token sequences.

Supports both VQ-VAE (6x6, 36 tokens) and FSQ-VAE (8x8, 64 tokens).
Tokenizes all episodes in the directory (base + pre-shifted).

Usage:
    python scripts/tokenize_episodes.py --model fsq --checkpoint checkpoints/fsq_best.pt
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _tokenize_frames(model, frames, batch_size, tokens_per_frame, device):
    """Encode frames through the model and return flat token array."""
    all_tokens = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        x = torch.from_numpy(batch).float().unsqueeze(1).to(device) / 255.0
        indices = model.encode(x)  # (B, grid, grid)
        all_tokens.append(indices.cpu().reshape(-1, tokens_per_frame).numpy())
    return np.concatenate(all_tokens, axis=0).astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Tokenize episodes with frozen tokenizer")
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--model", choices=["vqvae", "fsq"], default="fsq")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: checkpoints/{model}_best.pt)")
    parser.add_argument("--batch-size", type=int, default=128)
    # VQ-VAE specific
    parser.add_argument("--num-embeddings", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=8)
    # FSQ specific
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.model}_best.pt" if args.model == "fsq" \
            else "checkpoints/vqvae_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "fsq":
        from deepdash.fsq import FSQVAE
        model = FSQVAE(levels=args.levels).to(device)
        tokens_per_frame = 64
    else:
        from deepdash.vqvae import VQVAE
        model = VQVAE(num_embeddings=args.num_embeddings,
                      embedding_dim=args.embedding_dim).to(device)
        tokens_per_frame = 36

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.model.upper()} from {args.checkpoint}")

    episodes_dir = Path(args.episodes_dir)
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")

    # Verify shift directories exist (created by shift_episodes.py)
    has_shifts = any(
        ep.is_dir() and shift_re.search(ep.name)
        for ep in episodes_dir.glob("*")
    )
    if not has_shifts:
        print("ERROR: No shift augmentation found. "
              "Run scripts/shift_episodes.py first.")
        sys.exit(1)

    # Tokenize all episodes (base + pre-shifted)
    episodes = sorted(ep for ep in episodes_dir.glob("*")
                      if ep.is_dir() and (ep / "frames.npy").exists())
    print(f"Found {len(episodes)} episodes (base + shifted)")

    def _tokens_valid(path):
        """Check if tokens.npy exists and is not corrupt."""
        if not path.exists():
            return False
        try:
            np.load(path)
            return True
        except (EOFError, ValueError):
            path.unlink()
            print(f"  Deleted corrupt {path}")
            return False

    total_frames = 0
    skipped = 0

    with torch.no_grad():
        for ep in episodes:
            if _tokens_valid(ep / "tokens.npy"):
                skipped += 1
                continue

            frames = np.load(ep / "frames.npy")  # (T, 64, 64) uint8
            total_frames += len(frames)
            tokens = _tokenize_frames(model, frames, args.batch_size,
                                      tokens_per_frame, device)
            np.save(ep / "tokens.npy", tokens)
            print(f"  {ep.name}: {len(frames)} frames -> {tokens.shape} tokens")

    if skipped:
        print(f"Skipped {skipped} already-tokenized episodes.")
    print(f"\nDone. Tokenized {total_frames} new frames across "
          f"{len(episodes) - skipped} episodes.")


if __name__ == "__main__":
    main()
