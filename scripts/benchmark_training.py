"""Benchmark training throughput under various optimization combos.

Tests: eager vs compile, fp32 vs fp16 vs bf16, compile modes.
Reports time/step and throughput for each configuration.

Usage:
    python scripts/benchmark_training.py
    python scripts/benchmark_training.py --config configs/v5.yaml --warmup 3 --steps 10
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel


def make_synthetic_batch(vocab_size, context_frames, tokens_per_frame, batch_size, device):
    """Create a synthetic training batch matching real data shapes."""
    block_size = tokens_per_frame + 1  # tokens + status
    B, K, BS = batch_size, context_frames, block_size
    tokens = torch.randint(0, vocab_size, (B, K + 1, tokens_per_frame), device=device)
    status = torch.zeros(B, K + 1, 1, dtype=torch.long, device=device)
    frames = torch.cat([tokens, status], dim=2)
    actions = torch.randint(0, 2, (B, K), device=device)
    target = torch.randint(0, vocab_size, (B, tokens_per_frame), device=device)
    return frames, actions, target


def benchmark_config(name, model, optimizer, frames, actions, target,
                     amp_dtype, use_scaler, warmup, steps):
    """Run warmup + timed steps, return median ms/step."""
    device = frames.device
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, cpc_loss = model(frames, actions)
            pred = logits[:, :target.shape[1]]
            loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
            total = loss + cpc_loss

        optimizer.zero_grad()
        if scaler:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            times.append(dt)

    med = sorted(times)[len(times) // 2]
    mean = sum(times) / len(times)
    print(f"  {name:45s}  median={med:7.1f}ms  mean={mean:7.1f}ms  "
          f"throughput={frames.shape[0] / (med / 1000):6.0f} samples/s")
    return med


def main():
    parser = argparse.ArgumentParser(description="Benchmark training optimizations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for benchmark (default: 64)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    from deepdash.config import apply_config
    # Just need model params
    parser2 = argparse.ArgumentParser()
    for k in ["vocab_size", "embed_dim", "n_heads", "n_layers",
              "tokens_per_frame", "context_frames", "dropout"]:
        parser2.add_argument(f"--{k.replace('_', '-')}", default=None)
    parser2.add_argument("--config", default=args.config)
    parser2.add_argument("--adaln", default=None)
    margs = parser2.parse_args([])
    if args.config:
        margs.config = args.config
    apply_config(margs, section="transformer")

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: {margs.embed_dim}d / {margs.n_layers}L / {margs.n_heads}H")
    print(f"Batch size: {args.batch_size}, Warmup: {args.warmup}, Steps: {args.steps}")
    print()

    # Synthetic data
    frames, actions, target = make_synthetic_batch(
        int(margs.vocab_size), int(margs.context_frames),
        int(margs.tokens_per_frame), args.batch_size, device)

    # Check Flash Attention availability
    print("Flash Attention (SDPA):", torch.backends.cuda.flash_sdp_enabled())
    print()

    # (name, amp_dtype, use_scaler, use_compile, compile_mode, tf32, fused_optim)
    configs = [
        ("eager + fp32",                       None,           False, False, None,              False, False),
        ("eager + fp32 + tf32",                None,           False, False, None,              True,  False),
        ("eager + fp16 + GradScaler",          torch.float16,  True,  False, None,              False, False),
        ("eager + fp16 + GradScaler + tf32",   torch.float16,  True,  False, None,              True,  False),
        ("eager + bf16",                       torch.bfloat16, False, False, None,              False, False),
        ("eager + bf16 + tf32",                torch.bfloat16, False, False, None,              True,  False),
        ("eager + bf16 + tf32 + fused optim",  torch.bfloat16, False, False, None,              True,  True),
        ("compile + fp16 + tf32",              torch.float16,  True,  True,  "default",         True,  True),
        ("compile + bf16 + tf32",              torch.bfloat16, False, True,  "default",         True,  True),
        ("compile(reduce-overhead) + bf16 + tf32", torch.bfloat16, False, True, "reduce-overhead", True, True),
        ("compile(max-autotune) + bf16 + tf32",    torch.bfloat16, False, True, "max-autotune",    True, True),
    ]

    results = []
    for name, amp_dtype, use_scaler, use_compile, compile_mode, tf32, fused_optim in configs:
        # TF32 matmul precision (A100 tensor cores, ~3x faster matmuls)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

        # Fresh model each time to avoid compile state leaking
        model = WorldModel(
            vocab_size=int(margs.vocab_size), embed_dim=int(margs.embed_dim),
            n_heads=int(margs.n_heads), n_layers=int(margs.n_layers),
            context_frames=int(margs.context_frames), dropout=float(margs.dropout),
            tokens_per_frame=int(margs.tokens_per_frame),
            adaln=bool(getattr(margs, 'adaln', False)),
        ).to(device)

        if use_compile:
            try:
                model = torch.compile(model, mode=compile_mode)
                # Warmup compile
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    _ = model(frames, actions)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  {name:45s}  SKIPPED: {e}")
                continue

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                       fused=fused_optim)

        try:
            med = benchmark_config(name, model, optimizer, frames, actions, target,
                                   amp_dtype, use_scaler, args.warmup, args.steps)
            results.append((name, med))
        except Exception as e:
            print(f"  {name:40s}  FAILED: {e}")

        del model, optimizer
        torch.cuda.empty_cache()

    print("\n=== Summary (sorted by speed) ===")
    baseline = next((m for n, m in results if "eager + fp32" in n and "tf32" not in n), results[0][1])
    for name, med in sorted(results, key=lambda x: x[1]):
        speedup = baseline / med
        print(f"  {name:45s}  {med:7.1f}ms  {speedup:.2f}x vs eager fp32")


if __name__ == "__main__":
    main()
