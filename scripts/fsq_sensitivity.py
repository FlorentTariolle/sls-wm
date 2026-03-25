"""FSQ codebook sensitivity analysis.

Measures how much reconstruction degrades when perturbing quantized codes
by 1 or 2 steps in various FSQ dimensions. Used to calibrate the structured
label smoothing sigma for the transformer.

Usage:
    python scripts/fsq_sensitivity.py
    python scripts/fsq_sensitivity.py --checkpoint checkpoints/fsq_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepdash.fsq import FSQVAE
from deepdash.data_split import get_val_episodes, is_val_episode


def load_val_frames(episodes_dir, expert_episodes_dir, max_frames=10000, seed=42):
    """Load a sample of validation frames."""
    val_set = get_val_episodes(episodes_dir, expert_episodes_dir)
    all_frames = []
    for ep_dir in [episodes_dir, expert_episodes_dir]:
        p = Path(ep_dir)
        if not p.exists():
            continue
        for ep in sorted(p.glob("*")):
            fp = ep / "frames.npy"
            if not fp.exists():
                continue
            if is_val_episode(ep.name, val_set):
                all_frames.append(np.load(fp))
    data = np.concatenate(all_frames, axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=min(max_frames, len(data)), replace=False)
    return data[indices]


def main():
    parser = argparse.ArgumentParser(description="FSQ codebook sensitivity analysis")
    parser.add_argument("--checkpoint", default="checkpoints/fsq_best.pt")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    parser.add_argument("--episodes-dir", default="data/death_episodes")
    parser.add_argument("--expert-episodes-dir", default="data/expert_episodes")
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FSQVAE(levels=args.levels)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load data
    print("Loading validation frames...")
    frames = load_val_frames(
        args.episodes_dir, args.expert_episodes_dir, args.max_frames
    )
    print(f"Loaded {len(frames)} frames")

    # Encode all frames to get quantized codes
    all_z_q = []  # (B, D, 8, 8) quantized codes
    all_imgs = []  # (B, 1, 64, 64) original images

    with torch.no_grad():
        for i in range(0, len(frames), args.batch_size):
            batch = torch.from_numpy(
                frames[i:i + args.batch_size].astype(np.float32) / 255.0
            ).unsqueeze(1).to(device)
            z_e = model.encoder(batch)
            z_q, _ = model.fsq(z_e)
            all_z_q.append(z_q.cpu())
            all_imgs.append(batch.cpu())

    all_z_q = torch.cat(all_z_q)  # (N, D, 8, 8)
    all_imgs = torch.cat(all_imgs)  # (N, 1, 64, 64)

    n_dims = len(args.levels)
    half_levels = torch.tensor([L // 2 for L in args.levels], dtype=torch.float32)

    # Baseline reconstruction MSE
    with torch.no_grad():
        baseline_mse = 0.0
        for i in range(0, len(all_z_q), args.batch_size):
            z_q = all_z_q[i:i + args.batch_size].to(device)
            recon = model.decoder(z_q)
            orig = all_imgs[i:i + args.batch_size].to(device)
            baseline_mse += ((recon - orig) ** 2).sum().item()
        baseline_mse /= len(all_z_q)

    print(f"\nBaseline reconstruction MSE: {baseline_mse:.6f}")
    print()

    # Define perturbation experiments
    experiments = []

    # Single dimension perturbations: +1 and +2 in each dim
    for d in range(n_dims):
        for step in [1, 2]:
            experiments.append((f"dim{d}(L={args.levels[d]}) +{step}", [(d, step)]))

    # Two-dimension perturbations: +1 in two dims
    for d1 in range(n_dims):
        for d2 in range(d1 + 1, n_dims):
            experiments.append(
                (f"dim{d1}+dim{d2} +1 each", [(d1, 1), (d2, 1)])
            )

    print(f"{'Perturbation':<30} {'MSE':>10} {'Delta':>10} {'Ratio':>8} {'Valid%':>8}")
    print("-" * 70)

    results = []
    for name, perturbations in experiments:
        perturbed_z_q = all_z_q.clone()

        # Apply perturbations and clamp to valid range
        valid_mask = torch.ones(len(all_z_q), dtype=torch.bool)
        for d, step in perturbations:
            half = half_levels[d]
            max_val = half - (1.0 - args.levels[d] % 2)
            min_val = -half
            # Randomly perturb up or down
            N = perturbed_z_q.shape[0]
            signs = (torch.randint(0, 2, (N,)) * 2 - 1).float()  # (N,)
            delta = signs.view(N, 1, 1) * step  # (N, 1, 1)
            perturbed_z_q[:, d] = perturbed_z_q[:, d] + delta
            # Track which samples stay in valid range
            out_of_range = (
                (perturbed_z_q[:, d] > max_val) |
                (perturbed_z_q[:, d] < min_val)
            ).any(dim=-1).any(dim=-1)
            # Clamp to valid range
            perturbed_z_q[:, d] = perturbed_z_q[:, d].clamp(min_val, max_val)
            valid_mask &= ~out_of_range

        valid_pct = valid_mask.float().mean().item() * 100

        # Compute perturbed reconstruction MSE
        with torch.no_grad():
            perturbed_mse = 0.0
            for i in range(0, len(perturbed_z_q), args.batch_size):
                z_q = perturbed_z_q[i:i + args.batch_size].to(device)
                recon = model.decoder(z_q)
                orig = all_imgs[i:i + args.batch_size].to(device)
                perturbed_mse += ((recon - orig) ** 2).sum().item()
            perturbed_mse /= len(perturbed_z_q)

        delta = perturbed_mse - baseline_mse
        ratio = perturbed_mse / baseline_mse
        results.append((name, perturbed_mse, delta, ratio, valid_pct))
        print(f"{name:<30} {perturbed_mse:10.6f} {delta:+10.6f} {ratio:8.2f}x {valid_pct:7.1f}%")

    # Summary
    print()
    print("Interpretation for label smoothing sigma:")
    print("  - If +1 step causes ~1.5-2x MSE: sigma=0.9 is reasonable")
    print("  - If +1 step causes ~3-5x MSE: sigma is too high, codes are very distinct")
    print("  - If +1 step causes <1.2x MSE: sigma could be higher, codes are similar")
    print()
    print("Compare these results against the old FSQ checkpoint to see if")
    print("neighbor similarity changed with the new training.")


if __name__ == "__main__":
    main()
