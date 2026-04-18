#!/bin/bash
#SBATCH -J "train_ctrl_bc"
#SBATCH -o slurm/logs/train_controller_bc.out
#SBATCH -e slurm/logs/train_controller_bc.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=01:00:00

# Behavioral cloning: pretrain controller on expert episodes.
# Should be fast (~33K samples, 50 epochs).
#
# Submit:  sbatch slurm/train_controller_bc.sl
# Monitor: tail -f slurm/logs/train_controller_bc.out

module purge
module load aidl/pytorch/2.10.0-py3.12-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user --upgrade wandb "protobuf>=6.32" 2>/dev/null

echo "=== Train Controller (BC) ==="
python -u scripts/train_controller_bc.py \
    --expert-episodes-dir data/expert_episodes \
    --transformer-checkpoint checkpoints/transformer_best.pt \
    --epochs 50 \
    --batch-size 512 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --val-ratio 0.1 \
    --checkpoint-dir checkpoints \
    --seed 42
