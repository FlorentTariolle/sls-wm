#!/bin/bash
#SBATCH -J "bench_train"
#SBATCH -o slurm/logs/benchmark_training.out
#SBATCH -e slurm/logs/benchmark_training.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=01:00:00

# Benchmark training throughput under various optimizations.
# Tests eager/compile x fp32/fp16/bf16 combos.
# ~20 min total (8 configs x ~2 min each).
#
# Submit:  sbatch slurm/benchmark_training.sl
# Results: cat slurm/logs/benchmark_training.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6

echo "=== Benchmark 512d (batch_size=512, matching real training) ==="
python -u scripts/benchmark_training.py \
    --config configs/v4.yaml \
    --batch-size 512 \
    --warmup 3 \
    --steps 10
