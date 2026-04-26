# lewm-mario

Adaptation of [LeWorldModel](https://arxiv.org/abs/2603.19312) (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026) — a stable end-to-end Joint-Embedding Predictive Architecture — to **Super Mario Bros**.

This repo holds the data-generation, training, and evaluation scripts. It builds on:
- Upstream JEPA: [`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm)
- Mario fork: [`0xShug0/lewm_mario`](https://github.com/0xShug0/lewm_mario) — model code, dataset utilities, FM2 parser

## What this repo adds

| File | Purpose |
|---|---|
| `tas_replay.py` | Replays TAS movies (`.fm2`) through raw `nes_py.NESEnv` against the SMB ROM bundled with `gym-super-mario-bros`. Captures (frame, action) pairs from cold-boot through gameplay. Reads `x_pos`/`lives` directly from SMB1 RAM and detects desync (Mario dies or stops progressing) so partial trajectories before drift are still kept. |
| `madmario_rollout.py` | Alternative data source: rolls out [`yfeng997/MadMario`](https://github.com/yfeng997/MadMario)'s pretrained DDQN on `SuperMarioBros-1-1-v0`. Captures raw 240×256 RGB frames + 8-bit FM2 action vectors. Note: MadMario is a tutorial-grade policy and dies at the first Goomba most of the time — used here for pipeline smoke tests, not as the main expert. |
| `eval_lewm.py` | (1) Open-loop latent prediction MSE vs horizon. (2) Linear `x_pos` probe — predicts Mario's x position from the JEPA latent, demonstrating that the unsupervised representation encodes spatial structure. |
| `train_mario.py` | `lewm_mario/train_mario.py` patched with W&B logging hooks (`--wandb-project`, `--wandb-run-name`, `--wandb-entity`). Otherwise unchanged. |
| `eval_results.json` | Phase-1 evaluation results from the run described below. |

## Phase-1 results

Trained 50 epochs on a single Targon H100 (~25 min), batch 128, bf16, `--compile`.

- **Dataset:** 120/121 TAS files replayed → 156k action frames → 31k blocked transitions (frame_skip=5).
- **Convergence:** val `pred_loss` 0.36 → 0.127 (epoch 45 best).
- **Open-loop latent MSE** (mean) at horizons 1/2/4/8/16: 0.12 / 0.24 / 0.55 / 1.15 / 1.83.
- **`x_pos` linear probe (192-d latent → x px, range 0–1571):** test **R² = 0.83**, MAE 65 px.

The probe result is the LeWM paper's "physics emerges unsupervised" signature reproduced on Mario: the encoder was never trained against `x_pos` labels, but the latent recovers Mario's horizontal position with high accuracy.

Pre-trained checkpoint: [`obamaTeo/lewm-mario`](https://huggingface.co/obamaTeo/lewm-mario) (HF, private).

## Reproducing

Tested on Targon `h100-small` ($1.79/h) with image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.

```bash
# system deps
apt-get install -y git build-essential cmake ffmpeg

# python deps (numpy and scikit-image must both be pinned, otherwise
# scikit-image upgrades numpy>=2 and breaks nes_py with a uint8 OverflowError)
pip install "setuptools<60" wheel "numpy<2"
pip install nes-py gym-super-mario-bros einops opencv-python-headless wandb tensorboard
pip install "scikit-image<0.22" "gym==0.25.2"
pip install scikit-learn  # for the eval probe

# clone the upstream model code
git clone https://github.com/0xShug0/lewm_mario.git
git clone https://github.com/yfeng997/MadMario.git  # only needed for madmario_rollout.py

# 1. generate data via TAS replay
python tas_replay.py --traces-dir lewm_mario/traces --output-dir data/raw \
    --max-frames 20000 --stuck-window 300

# 2. block + precompute (upstream scripts)
python lewm_mario/build_lewm_mario_dataset.py \
    --dataset-root data/raw --output-dir data/blocked --frame-skip 5
python lewm_mario/precompute_mario_dataset.py \
    --dataset-root data/blocked --output-dir data/precomputed --image-size 224

# 3. train
python train_mario.py \
    --dataset-root data/blocked \
    --precomputed-root data/precomputed \
    --output-dir runs/tas_v1 \
    --epochs 50 --batch-size 128 --num-workers 6 \
    --save-every 10 --npz-load-mode preload --max-cached-episodes 200 \
    --batching episode --log-every-steps 25 --compile \
    --wandb-project lewm-mario --wandb-run-name tas-v1-50ep

# 4. evaluate
python eval_lewm.py --ckpt runs/tas_v1/best.pt --n-starts 300
```

## Known limitations

- **TAS desync.** TAS files were recorded against FCEUX with the canonical SMB1 (JU) PRG0 iNES dump (sha256 `F61548FD…`). `gym-super-mario-bros` ships a normalized ROM (sha256 `ec299b9…`); replay through `nes_py` desyncs after ~200–3000 frames per TAS. Each TAS still contributes a clean prefix but full-game replays do not survive.
- **Goal-conditioned planning only.** The CEM planner in `lewm_mario/mario_lewm/planning.py` plans toward a goal *image*; for autonomous Mario play a reward head is needed. Not implemented yet.
- **Action space.** FM2 8-button rows blocked into 40-d vectors. CEM at planning time over this space requires constraining to observed action combinations.

## Next steps (Path B)

1. Add a small reward MLP `r̂(z)` trained jointly with stop-grad on the encoder, on `x_pos` delta + flag-get from the TAS RAM dumps.
2. Replace goal-MSE in `mario_lewm/planning.py` with `−Σ r̂(ẑ_t)` and run online MPC against `gym_super_mario_bros`.
3. Either source the F61548FD ROM dump for cleaner TAS replay, or train a strong PPO baseline on Targon and use its rollouts as expert data.

## License

Code licensed MIT (matching upstream `lewm_mario`). TAS movie files belong to their original creators.
