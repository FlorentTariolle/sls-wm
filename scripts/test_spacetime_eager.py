"""Eager-mode sanity test for v5-spacetime AdaLNSpaceTimeBlock.

Runs forward + backward with no torch.compile, no cudagraphs -- just the
raw model. If this passes, the model code is correct and all the crashes
you've been seeing are torch.compile / cudagraph issues, not model bugs.

Pass --compile to also try torch.compile(mode=default).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from deepdash.world_model import WorldModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Mirror configs/v5-spacetime.yaml model section
    model = WorldModel(
        vocab_size=625,
        n_actions=2,
        embed_dim=512,
        n_heads=8,
        n_layers=8,
        context_frames=4,
        tokens_per_frame=64,
        dropout=0.15,
        adaln=True,
        attention_pattern="space_time_sst_t",
    ).to(device)
    model.train()

    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={device}")
    print(f"attention_pattern={model.attention_pattern} n_layers={len(model.blocks)}")
    print(f"block axes: {[b.axis for b in model.blocks]}")
    print(f"seq_len={model.seq_len} block_size={model.block_size} vocab={model.full_vocab_size}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    if args.compile:
        print(f"torch.compile mode={args.mode}")
        model = torch.compile(model, mode=args.mode)

    B = 4
    K = 4
    block_size = model.block_size  # 65
    vocab = model.full_vocab_size

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(3):
        frame_tokens = torch.randint(0, 625, (B, K + 1, block_size), device=device, dtype=torch.long)
        frame_tokens[:, :, -1] = model.ALIVE_TOKEN  # all alive for simplicity
        actions = torch.randint(0, 2, (B, K), device=device, dtype=torch.long)
        target = frame_tokens[:, K].clone()

        opt.zero_grad(set_to_none=True)
        logits, cpc_loss = model(frame_tokens, actions)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = ce + cpc_loss
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"step {step} loss={loss.item():.4f} ce={ce.item():.4f} cpc={cpc_loss.item():.4f} grad_norm={grad_norm.item():.3f}")

    print("OK -- eager forward+backward works. Model code is fine.")


if __name__ == "__main__":
    main()
