#!/bin/zsh
cd "$(dirname "$0")/.."

python code/train_pass_selection.py \
    --data_root data/leverkusen_data \
    --holdout_match_id 3895348 \
    --epochs 1 \
    --batch_size 30 \
    --lr 1e-3 \
    --device cpu \
    --compute_velocities \
    --val_split 0.15 \
    --embed_team "Bayer Leverkusen" \
    --out_ckpt checkpoints/testing.pt
