"""Benchmark inference latency for real-time play.

Two modes:
  - Prefill only: context → h_t (what real-time play actually needs)
  - Full predict_next_frame: prefill + 65-step autoregressive decode (dream rollouts)

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel


def prefill_only(model, ctx_s, actions):
    """Run only the prefill pass to extract h_t (no autoregressive decode)."""
    K = model.context_frames
    BS = model.block_size

    parts = []
    for i in range(K):
        parts.append(model.token_embed(ctx_s[:, i]))
        act = model.action_embed(actions[:, i])
        parts.append(act.unsqueeze(1))
    x = torch.cat(parts, dim=1)

    ctx_len = K * (BS + 1)
    ctx_mask = model.attn_mask[:ctx_len, :ctx_len]
    rope_cos = model.rope_cos[:ctx_len]
    rope_sin = model.rope_sin[:ctx_len]

    for block in model.blocks:
        x, _ = block(x, ctx_mask, rope_cos, rope_sin)
    x = model.ln_f(x)
    h_t = x[:, -1]  # hidden state at last context position
    return h_t


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_best.pt")
    parser.add_argument("--episodes-dir", default="data/episodes")
    parser.add_argument("--device", default=None, help="Force device (cpu/cuda)")
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model = WorldModel(
        vocab_size=1000, embed_dim=128, n_heads=4, n_layers=6,
        context_frames=4, tokens_per_frame=64,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Load one episode for context
    ep = next(ep for ep in sorted(Path(args.episodes_dir).glob("*"))
              if (ep / "tokens.npy").exists())
    t = np.load(ep / "tokens.npy").astype(np.int64)
    ctx = torch.from_numpy(t[:4]).unsqueeze(0).to(device)
    status = torch.full((1, 4, 1), model.ALIVE_TOKEN,
                        dtype=torch.long, device=device)
    ctx_s = torch.cat([ctx, status], dim=2)
    actions = torch.zeros(1, 4, dtype=torch.long, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def bench(fn, label):
        with torch.no_grad():
            for _ in range(args.warmup):
                fn()
            sync()
            start = time.perf_counter()
            for _ in range(args.n_runs):
                fn()
            sync()
            ms = (time.perf_counter() - start) / args.n_runs * 1000
        print(f"  {label}: {ms:.2f} ms", end="")
        if ms < 33.3:
            print(f"  -> OK ({33.3 / ms:.1f}x margin)")
        else:
            print(f"  -> TOO SLOW ({ms / 33.3:.1f}x over budget)")
        return ms

    print(f"\nBudget: 33.3 ms (30 FPS)\n")

    bench(lambda: prefill_only(model, ctx_s, actions),
          "Prefill only (real-time play)")
    bench(lambda: model.predict_next_frame(ctx_s, actions, return_hidden=True),
          "Full predict_next_frame (dream rollouts)")


if __name__ == "__main__":
    main()
