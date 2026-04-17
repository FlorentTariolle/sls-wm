"""Benchmark FSQ-VAE training throughput under optimization combos.

Mirrors scripts/benchmark_training.py but targets the FSQ-VAE (single-channel
64x64 encoder/decoder + slowness/uniformity losses) instead of the Transformer.

Each config runs in a subprocess for clean CUDA state.

Usage:
    python scripts/benchmark_fsq_training.py
    python scripts/benchmark_fsq_training.py --config configs/v5b-local.yaml \\
        --batch-size 512 --warmup 3 --steps 10
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# (name, amp_dtype, use_scaler, use_compile, compile_mode, tf32, fused_optim, channels_last)
CONFIGS = [
    ("eager + fp32",                          "none",     False, False, "none",            False, False, False),
    ("eager + fp32 + tf32",                   "none",     False, False, "none",            True,  False, False),
    ("eager + fp32 + channels_last",          "none",     False, False, "none",            False, False, True),
    ("eager + fp16 + GradScaler",             "float16",  True,  False, "none",            False, False, False),
    ("eager + fp16 + GradScaler + tf32",      "float16",  True,  False, "none",            True,  False, False),
    ("eager + fp16 + channels_last",          "float16",  True,  False, "none",            False, False, True),
    ("eager + bf16",                          "bfloat16", False, False, "none",            False, False, False),
    ("eager + bf16 + tf32 + fused optim",     "bfloat16", False, False, "none",            True,  True,  False),
    ("compile(default) + fp32",               "none",     False, True,  "default",         False, False, False),
    ("compile(default) + fp16",               "float16",  True,  True,  "default",         True,  True,  False),
    ("compile(default) + bf16",               "bfloat16", False, True,  "default",         True,  True,  False),
    ("compile(default) + fp32 + channels_last", "none",   False, True,  "default",         False, False, True),
    ("compile(reduce-overhead) + fp16",       "float16",  True,  True,  "reduce-overhead", True,  True,  False),
    ("compile(max-autotune) + fp16",          "float16",  True,  True,  "max-autotune",    True,  True,  False),
]


def run_single_config(config_json, model_json, batch_size, warmup, steps):
    """Run one config in an isolated subprocess."""
    from deepdash.fsq import FSQVAE, fsqvae_loss, grwm_slowness, grwm_uniformity

    cfg = json.loads(config_json)
    mp = json.loads(model_json)
    amp_dtype_str = cfg["amp_dtype"]
    use_scaler = cfg["use_scaler"]
    use_compile = cfg["use_compile"]
    compile_mode = cfg["compile_mode"]
    tf32 = cfg["tf32"]
    fused_optim = cfg["fused_optim"]
    channels_last = cfg["channels_last"]

    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "none": None}[amp_dtype_str]

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = True

    # Inductor has channels_last layout heuristics that break on some Turing
    # conv kernels; disable when we're NOT explicitly opting in
    if use_compile and not channels_last:
        try:
            import torch._inductor.config as ind_cfg
            ind_cfg.layout_optimization = False
        except Exception:
            pass

    levels = mp["levels"]
    model = FSQVAE(levels=levels).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if use_compile:
        model = torch.compile(model, mode=compile_mode)

    # Synthetic input: 64x64 single-channel frames in [0,1]
    B = batch_size
    ft = torch.rand(B, 1, 64, 64, device=device)
    ft1 = (ft + 0.05 * torch.randn_like(ft)).clamp(0.0, 1.0)
    if channels_last:
        ft = ft.contiguous(memory_format=torch.channels_last)
        ft1 = ft1.contiguous(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, fused=fused_optim)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # Optional compile warmup pass (not timed)
    if use_compile:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            _ = model(ft)
        torch.cuda.synchronize()

    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            recon_t, z_e_t, _ = model(ft)
            recon_t1, z_e_t1, _ = model(ft1)
            recon_loss = (fsqvae_loss(recon_t, ft) + fsqvae_loss(recon_t1, ft1)) / 2
            slow_loss = grwm_slowness(z_e_t, z_e_t1)
            uniform_loss = grwm_uniformity(z_e_t)
            loss = recon_loss + 0.1 * slow_loss + 0.01 * uniform_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(dt)

    med = sorted(times)[len(times) // 2]
    mean = sum(times) / len(times)
    print(json.dumps({
        "name": cfg["name"], "median": med, "mean": mean,
        "throughput": batch_size / (med / 1000),
    }))


def main():
    parser = argparse.ArgumentParser(description="Benchmark FSQ-VAE training optimizations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--_run-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_config-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_model-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._run_single:
        run_single_config(args._config_json, args._model_json,
                          args.batch_size, args.warmup, args.steps)
        return

    # Parent process: load levels from config and dispatch subprocesses
    from deepdash.config import apply_config
    parser2 = argparse.ArgumentParser()
    parser2.add_argument("--levels", nargs="+", type=int, default=None)
    parser2.add_argument("--config", default=args.config)
    margs = parser2.parse_args([])
    if args.config:
        margs.config = args.config
    apply_config(margs, section="fsq")
    # fsq section doesn't have levels; grab from model section
    from deepdash.config import load_config
    model_cfg = load_config(margs.config, section="model") if args.config else {}
    levels = model_cfg.get("levels") or margs.levels or [5, 5, 5, 5]
    model_json = json.dumps({"levels": levels})

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"FSQ levels: {levels}  (codebook={1 if not levels else __import__('math').prod(levels)})")
    print(f"Batch size: {args.batch_size}, Warmup: {args.warmup}, Steps: {args.steps}")
    print()

    results = []
    for name, amp_dtype, use_scaler, use_compile, compile_mode, tf32, fused_optim, channels_last in CONFIGS:
        cfg = json.dumps({
            "name": name, "amp_dtype": amp_dtype, "use_scaler": use_scaler,
            "use_compile": use_compile, "compile_mode": compile_mode,
            "tf32": tf32, "fused_optim": fused_optim, "channels_last": channels_last,
        })

        cmd = [
            sys.executable, __file__,
            "--_run-single",
            "--_config-json", cfg,
            "--_model-json", model_json,
            "--batch-size", str(args.batch_size),
            "--warmup", str(args.warmup),
            "--steps", str(args.steps),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                err = (result.stderr.strip().split("\n") or [""])[-1][:140]
                print(f"  {name:45s}  FAILED: {err}")
                continue

            data = json.loads(result.stdout.strip().split("\n")[-1])
            print(f"  {name:45s}  median={data['median']:7.1f}ms  "
                  f"mean={data['mean']:7.1f}ms  "
                  f"throughput={data['throughput']:7.0f} samples/s")
            results.append((name, data["median"]))
        except subprocess.TimeoutExpired:
            print(f"  {name:45s}  TIMEOUT (>600s)")
        except Exception as e:
            print(f"  {name:45s}  FAILED: {e}")

    if results:
        print("\n=== Summary (sorted by speed) ===")
        baseline = next((m for n, m in results if n == "eager + fp32"), results[0][1])
        for name, med in sorted(results, key=lambda x: x[1]):
            speedup = baseline / med
            print(f"  {name:45s}  {med:7.1f}ms  {speedup:.2f}x vs eager fp32")


if __name__ == "__main__":
    main()
