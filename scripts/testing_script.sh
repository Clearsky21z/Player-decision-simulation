#!/bin/zsh
cd "$(dirname "$0")/.."

python code/train_pass_selection.py \
    --data_root data/leverkusen_data \
    --train_match_ids 3895348,3895340,3895333 \
    --epochs 1 \
    --batch_size 100 \
    --lr 1e-3 \
    --device mps \
    --compute_velocities \
    --val_split 0.15 \
    --embed_team "Bayer Leverkusen" \
    --auto_select_players 6 \
    --auto_goalkeepers 1 \
    --loss ce \
    --out_ckpt checkpoints/testing_with_new_stuff3.pt
