"""Generate the SLS kernel visualization for the paper (Figure 2).

Two panels:
  (a) 2D heatmap slice of SLS weights for a center token
  (b) Sorted SLS vs uniform distribution comparison (log scale)

Usage: python scripts/gen_sls_figure.py [--output paper/figures/sls_kernel.pdf]
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def build_sls_row(levels, target_idx, sigma=0.9, smoothing=0.1, dim_weights=None):
    """Build the SLS target distribution for a single token (CPU-only)."""
    vocab_size = math.prod(levels)
    n_dims = len(levels)

    # Mixed-radix decomposition
    divisors = []
    acc = 1
    for L in reversed(levels):
        divisors.append(acc)
        acc *= L
    divisors.reverse()

    coords = torch.zeros(vocab_size, n_dims)
    for idx in range(vocab_size):
        remainder = idx
        for d in range(n_dims):
            coords[idx, d] = remainder // divisors[d]
            remainder = remainder % divisors[d]

    # Weighted squared distance from target
    target_coords = coords[target_idx].unsqueeze(0)  # (1, D)
    diff = coords - target_coords  # (V, D)
    if dim_weights is not None:
        w = torch.tensor(dim_weights, dtype=torch.float32)
        diff = diff * w
    sq_dist = (diff ** 2).sum(dim=-1)  # (V,)

    # Gaussian kernel
    weights = torch.exp(-sq_dist / (2 * sigma ** 2))
    weights[target_idx] = 0.0

    # Normalize and apply smoothing
    row_sum = weights.sum().clamp(min=1e-8)
    weights = smoothing * weights / row_sum
    weights[target_idx] = 1.0 - smoothing

    return weights.numpy(), coords.numpy(), divisors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/figures/sls_kernel.pdf")
    args = parser.parse_args()

    levels = [8, 5, 5, 5]
    sigma = 0.9
    smoothing = 0.1
    dim_weights = [1.29, 0.85, 0.97, 0.89]

    # Target token: center of lattice (3, 2, 2, 2) -> index 437
    target_idx = 3 * 125 + 2 * 25 + 2 * 5 + 2  # = 437
    sls_row, coords, divisors = build_sls_row(
        levels, target_idx, sigma, smoothing, dim_weights
    )

    # --- Paper-quality settings ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Panel (a): 2D heatmap slice ---
    # Fix dim0=3, dim3=2 (matching target), vary dim1 (5) x dim2 (5)
    target_d0, target_d3 = 3, 2
    target_d1, target_d2 = 2, 2
    grid = np.full((levels[1], levels[2]), np.nan)
    for d1 in range(levels[1]):
        for d2 in range(levels[2]):
            idx = target_d0 * divisors[0] + d1 * divisors[1] + d2 * divisors[2] + target_d3
            if (d1, d2) != (target_d1, target_d2):
                grid[d1, d2] = sls_row[idx]

    im = ax1.imshow(grid, cmap="viridis", origin="lower", aspect="equal")
    # Mark target cell with a red square
    ax1.plot(target_d2, target_d1, "s", markeredgecolor="red", markerfacecolor="red",
             markersize=12, markeredgewidth=2)
    ax1.set_xlabel("Dim 2 (5 levels)")
    ax1.set_ylabel("Dim 1 (5 levels)")
    ax1.set_title("(a) SLS weights (2D slice)")
    ax1.set_xticks(range(levels[2]))
    ax1.set_yticks(range(levels[1]))
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    # --- Panel (b): Sorted distribution comparison ---
    # Remove target from distribution
    off_diag = np.delete(sls_row, target_idx)
    sorted_sls = np.sort(off_diag)[::-1]
    uniform_val = smoothing / (math.prod(levels) - 1)

    ax2.semilogy(sorted_sls, color="#2196F3", linewidth=1.2, label="SLS (Gaussian)")
    ax2.axhline(uniform_val, color="#FF9800", linewidth=1.2, linestyle="--", label="Uniform")
    ax2.set_xlabel("Token rank (by SLS weight)")
    ax2.set_ylabel("Smoothing probability")
    ax2.set_title("(b) SLS vs. uniform smoothing")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(0, len(sorted_sls))

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
