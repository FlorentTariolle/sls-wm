"""Plot FSQ-VAE training curves from CSV log.

Usage:
    python scripts/plot_fsq_training.py
    python scripts/plot_fsq_training.py --log checkpoints/fsq_log.csv
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot FSQ-VAE training curves")
    parser.add_argument("--log", default="checkpoints/fsq_log.csv")
    parser.add_argument("--output", default="plots/fsq_training.png")
    args = parser.parse_args()

    df = pd.read_csv(args.log)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("FSQ-VAE Training", fontsize=14, fontweight="bold")

    # Reconstruction loss
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_recon"], label="Train", color="royalblue")
    ax.plot(df["epoch"], df["val_recon"], label="Val", color="orange")
    ax.set_title("Reconstruction Loss")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.2)

    # Learning rate
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["lr"].astype(float), color="crimson", linewidth=2)
    ax.set_title("Learning Rate")
    ax.set_ylabel("LR")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.grid(alpha=0.2)

    # GRWM slowness loss
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["train_slow"], color="green")
    ax.set_title("GRWM Slowness Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.2)

    # GRWM uniformity loss
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["train_uniform"], color="purple")
    ax.set_title("GRWM Uniformity Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.2)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
