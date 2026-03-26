#!/bin/bash
#SBATCH --account=def-vianeylb_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-03:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "$SCRATCH/Player-decision-simulation" || exit 1
source "$HOME/myenv/bin/activate"

python code/train_pass_selection.py \
    --data_root data/leverkusen_data \
    --train_match_ids 3895348,3895340,3895333 \
    --epochs 1 \
    --batch_size 30 \
    --lr 1e-3 \
    --device cuda \
    --compute_velocities \
    --val_split 0.15 \
    --embed_team "Bayer Leverkusen" \
    --auto_select_players 6 \
    --auto_goalkeepers 1 \
    --loss ce \
    --out_ckpt checkpoints/fir_context_fail_testing.pt
