#!/bin/bash
#SBATCH -J "regen_data"
#SBATCH -o slurm/logs/regen_data.out
#SBATCH -e slurm/logs/regen_data.err
#SBATCH -p ar_mig
#SBATCH --gres=gpu:a100_2g.20gb:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 32G
#SBATCH --time=02:00:00

# Regenerate shift-augmented episodes and re-tokenize.
# Use this after a tree rename/move breaks the actions.npy symlinks
# in shift variants. Sized for an A100 MIG 3g.20gb slice.
#
# Submit:  sbatch slurm/regen_data.sl
# Monitor: tail -f slurm/logs/regen_data.out

module purge
module load aidl/pytorch/2.6.0-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user wandb 2>/dev/null

# 1. Rebuild shift-augmented episode dirs with relative symlinks.
#    Deletes existing _s*_* dirs and recreates frames.npy + actions.npy.
python -u scripts/shift_episodes.py \
    --episodes-dir data/death_episodes \
    --expert-episodes-dir data/expert_episodes

# 2. Re-tokenize all episodes (shift dirs have no tokens.npy after step 1).
python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/death_episodes \
    --batch-size 256 \
    --levels 8 5 5 5

python -u scripts/tokenize_episodes.py \
    --model fsq \
    --checkpoint checkpoints/fsq_best.pt \
    --episodes-dir data/expert_episodes \
    --batch-size 256 \
    --levels 8 5 5 5
