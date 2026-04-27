#!/bin/bash
# Phase 3 campaign: runs the four ablation experiments end-to-end on the pod.
# Each stage logs to /workspace/runs/<stage>.log. Stage names appear as
# "==== STAGE: <name> ====" markers so monitors can grep on them.
#
# Stages:
#   1. precompute_dataset   — resize TAS replays to 224×224
#   2. latent_cache         — encode all windows once for v3 reward head
#   3. reward_head_v3       — composite (Δx + Δscore + Δcoins + Δptype) + frozen baseline encoder
#   4. joint_v2_score       — joint finetune with composite reward, baseline encoder
#   5. eval_score           — autonomous eval of (3) and (4)
#   6. eval_horizon8        — re-eval baseline + score variants with CEM horizon=8
#   7. vit_small_phase1     — Phase 1 retrain with ViT-small (~22M)
#   8. vit_small_phase2     — reward head + joint + eval on the bigger encoder
#   9. ppo_train            — PPO expert training (2M steps)
#   10. ppo_rollout         — collect PPO rollouts as TAS-shaped npzs
#   11. combined_phase1     — Phase 1 retrain on TAS+PPO union (ViT-tiny)
#   12. combined_phase2     — reward head + joint + eval on combined-data ckpt
set -e
set -o pipefail
cd /workspace/repo
git pull -q

stage() {
  echo "==== STAGE: $1 ===="
}

# Common args
RAW=/workspace/data/tas_full
PRE=/workspace/data/tas_precomputed
BASE_CKPT=/workspace/ckpt/best.pt
JOINT_CKPT=/workspace/ckpt/joint_best.pt

# === 1. precompute ===
if [ ! -d "$PRE" ] || [ "$(ls "$PRE" 2>/dev/null | wc -l)" -lt 100 ]; then
  stage precompute_dataset
  python3 /workspace/lewm_mario_pkg/precompute_mario_dataset.py \
    --dataset-root "$RAW" \
    --output-dir "$PRE" \
    --image-size 224 2>&1 | tee /workspace/runs/precompute.log | tail -50
fi

# === 2. latent cache (one-shot — runs minimal v1 head training but produces the cache file) ===
if [ ! -f /workspace/runs/reward_head/lat_targets.pt ]; then
  stage latent_cache
  mkdir -p /workspace/runs/reward_head
  python3 finetune_reward_head.py \
    --ckpt "$BASE_CKPT" \
    --dataset-root "$PRE" \
    --raw-root "$RAW" \
    --cache /workspace/runs/reward_head/lat_targets.pt \
    --out /workspace/runs/reward_head/reward_head_smoke.pt \
    --epochs 1 2>&1 | tee /workspace/runs/latent_cache.log | tail -30
fi

# === 3. reward head v3 (composite reward, frozen baseline encoder) ===
stage reward_head_v3
python3 finetune_reward_head_v3.py \
  --ckpt "$BASE_CKPT" \
  --dataset-root "$PRE" \
  --raw-root "$RAW" \
  --latent-cache /workspace/runs/reward_head/lat_targets.pt \
  --out /workspace/runs/reward_head_v3/reward_head_v3.pt \
  --wandb-run-name reward-head-v3 \
  --epochs 50 2>&1 | tee /workspace/runs/reward_head_v3.log | tail -30

# === 4. joint v2 with composite reward (baseline encoder weights as init) ===
stage joint_v2_score
python3 joint_finetune_v2.py \
  --ckpt "$BASE_CKPT" \
  --dataset-root "$PRE" \
  --raw-root "$RAW" \
  --out-dir /workspace/runs/joint_v2_score \
  --wandb-run-name joint-v2-score \
  --epochs 12 2>&1 | tee /workspace/runs/joint_v2_score.log | tail -30

# === 5. eval autonomous play (v3 head + joint v2 score) ===
stage eval_score
python3 autonomous_eval_v2.py \
  --ckpt "$BASE_CKPT" \
  --reward-head /workspace/runs/reward_head_v3/reward_head_v3.pt \
  --out-dir /workspace/runs/eval/rh_v3 \
  --label rh_v3 2>&1 | tee /workspace/runs/eval_rh_v3.log | tail -30
python3 autonomous_eval_v2.py \
  --ckpt /workspace/runs/joint_v2_score/best.pt \
  --out-dir /workspace/runs/eval/joint_v2_score \
  --label joint_v2_score 2>&1 | tee /workspace/runs/eval_joint_v2_score.log | tail -30

# === 6. eval horizon=8 on the strongest ckpts ===
stage eval_horizon8
for ck_label in "joint_baseline:$JOINT_CKPT" "joint_v2_score:/workspace/runs/joint_v2_score/best.pt"; do
  label=${ck_label%%:*}; ck=${ck_label##*:}
  python3 autonomous_eval_v2.py \
    --ckpt "$ck" \
    --out-dir /workspace/runs/eval/h8_$label \
    --label "${label}_h8" \
    --horizon 8 --n-samples 256 \
    --max-videos 2 2>&1 | tee /workspace/runs/eval_h8_$label.log | tail -20
done

# === 7. ViT-small Phase 1 retrain ===
stage vit_small_phase1
python3 /workspace/lewm_mario_pkg/train_mario.py \
  --dataset-root "$RAW" \
  --precomputed-root "$PRE" \
  --output-dir /workspace/runs/vit_small \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-5 \
  --encoder-hidden-dim 384 \
  --encoder-heads 6 \
  --encoder-mlp-dim 1536 \
  --predictor-hidden-dim 384 \
  --predictor-output-dim 384 \
  --action-embed-dim 384 \
  --compile 2>&1 | tee /workspace/runs/vit_small_phase1.log | tail -50

# === 8. reward head + joint + eval on ViT-small ===
stage vit_small_phase2
# precompute latent cache for the new encoder
python3 finetune_reward_head.py \
  --ckpt /workspace/runs/vit_small/best.pt \
  --dataset-root "$PRE" \
  --raw-root "$RAW" \
  --cache /workspace/runs/vit_small/lat_targets.pt \
  --out /workspace/runs/vit_small/reward_head_smoke.pt \
  --epochs 1 2>&1 | tee /workspace/runs/vit_small_latent_cache.log | tail -10

python3 finetune_reward_head_v3.py \
  --ckpt /workspace/runs/vit_small/best.pt \
  --dataset-root "$PRE" \
  --raw-root "$RAW" \
  --latent-cache /workspace/runs/vit_small/lat_targets.pt \
  --out /workspace/runs/vit_small/reward_head_v3.pt \
  --wandb-run-name reward-head-v3-vit-small \
  --epochs 50 2>&1 | tee /workspace/runs/vit_small_reward_v3.log | tail -20

python3 joint_finetune_v2.py \
  --ckpt /workspace/runs/vit_small/best.pt \
  --dataset-root "$PRE" \
  --raw-root "$RAW" \
  --out-dir /workspace/runs/vit_small_joint \
  --wandb-run-name joint-v2-vit-small \
  --epochs 12 2>&1 | tee /workspace/runs/vit_small_joint.log | tail -30

python3 autonomous_eval_v2.py \
  --ckpt /workspace/runs/vit_small_joint/best.pt \
  --out-dir /workspace/runs/eval/vit_small_joint \
  --label vit_small_joint 2>&1 | tee /workspace/runs/eval_vit_small.log | tail -20

# === 9. PPO training ===
stage ppo_train
mkdir -p /workspace/runs/ppo
python3 ppo_expert.py train \
  --steps 2000000 \
  --n-envs 8 \
  --ckpt-out /workspace/runs/ppo/ppo.zip 2>&1 | tee /workspace/runs/ppo_train.log | tail -50

# === 10. PPO rollout capture ===
stage ppo_rollout
python3 ppo_expert.py rollout \
  --ckpt /workspace/runs/ppo/ppo.zip \
  --n-episodes 80 \
  --max-frames 2400 \
  --epsilon 0.05 \
  --out-dir /workspace/data/ppo_full 2>&1 | tee /workspace/runs/ppo_rollout.log | tail -50

# === 11. combined dataset Phase 1 retrain (ViT-tiny) ===
stage combined_phase1
mkdir -p /workspace/data/combined_full /workspace/data/combined_precomputed
# symlink raw episodes from both sources
for f in "$RAW"/*.npz /workspace/data/ppo_full/*.npz; do
  [ -e "$f" ] || continue
  ln -sf "$f" /workspace/data/combined_full/$(basename "$f")
done
python3 /workspace/lewm_mario_pkg/precompute_mario_dataset.py \
  --dataset-root /workspace/data/combined_full \
  --output-dir /workspace/data/combined_precomputed \
  --image-size 224 2>&1 | tee /workspace/runs/combined_precompute.log | tail -30

python3 /workspace/lewm_mario_pkg/train_mario.py \
  --dataset-root /workspace/data/combined_full \
  --precomputed-root /workspace/data/combined_precomputed \
  --output-dir /workspace/runs/combined_phase1 \
  --epochs 50 --batch-size 128 --lr 5e-5 --compile 2>&1 \
  | tee /workspace/runs/combined_phase1.log | tail -50

# === 12. combined Phase 2 (reward head + joint + eval) ===
stage combined_phase2
python3 finetune_reward_head.py \
  --ckpt /workspace/runs/combined_phase1/best.pt \
  --dataset-root /workspace/data/combined_precomputed \
  --raw-root /workspace/data/combined_full \
  --cache /workspace/runs/combined_phase1/lat_targets.pt \
  --out /workspace/runs/combined_phase1/reward_head_smoke.pt \
  --epochs 1 2>&1 | tee /workspace/runs/combined_latent_cache.log | tail -10

python3 joint_finetune_v2.py \
  --ckpt /workspace/runs/combined_phase1/best.pt \
  --dataset-root /workspace/data/combined_precomputed \
  --raw-root /workspace/data/combined_full \
  --out-dir /workspace/runs/combined_joint \
  --wandb-run-name joint-v2-combined \
  --epochs 12 2>&1 | tee /workspace/runs/combined_joint.log | tail -30

python3 autonomous_eval_v2.py \
  --ckpt /workspace/runs/combined_joint/best.pt \
  --out-dir /workspace/runs/eval/combined_joint \
  --label combined_joint 2>&1 | tee /workspace/runs/eval_combined.log | tail -20

echo "==== CAMPAIGN_DONE ===="
