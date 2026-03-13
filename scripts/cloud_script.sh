#!/bin/bash
#SBATCH --account=def-vianeylb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-03:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

cd $SCRATCH/Player-decision-simulation || exit 1
source ~/myenv/bin/activate

python ../code/train_pass_selection.py \
    --data_root ../data/leverkusen_data \
    --holdout_match_id 3895348 \
    --epochs 1 \
    --batch_size 30 \
    --lr 1e-3 \
    --device cpu \
    --compute_velocities \
    --val_split 0.15 \
    --embed_team "Bayer Leverkusen" \
    --out_ckpt ../checkpoints/testing.pt
