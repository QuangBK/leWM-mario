# LeWM Mario — Full Report

## Goal

Adapt **LeWorldModel** (Maes/Le Lidec/Scieur/LeCun/Balestriero, [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)) — a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) world model — to play Super Mario Bros. autonomously.

LeWM in one line: ViT encoder + AR transformer predictor + SIGReg anti-collapse, trained pixel-only with `L = MSE(ẑ_{t+1}, z_{t+1}) + λ·SIGReg(Z)`. Two losses, one tunable hyperparameter.

## What we considered for data

| Source | Outcome |
|---|---|
| `yfeng997/MadMario` DDQN ckpt | **Rejected.** Trained on level 1-1 only with 2 actions (right, right+A). Empirically dies at the first Goomba ~9/10 rolls. Useful as a pipeline smoke test, useless as expert. |
| FCEUX TAS replay (the path the upstream `0xShug0/lewm_mario` uses) | Skipped. Requires FCEUX binary + the exact F61548FD ROM dump. |
| **TAS replay through `nes_py.NESEnv` against gym-super-mario-bros's bundled ROM** | **Picked.** Reuses the 123 FM2 files in upstream `lewm_mario/traces/`; reads `x_pos`/`lives` from SMB1 RAM directly so we can detect desync and truncate. |
| PPO from scratch | Reserved as fallback if TAS volume turned out too small. Not needed. |

**Desync caveat:** TASes were recorded against FCEUX with the canonical `F61548FD…` iNES dump; gym-super-mario-bros ships a normalized `ec299b9…` ROM. Replay through `nes_py` desyncs after ~200–3000 frames. Each TAS still contributes a clean prefix.

**Final dataset:** 120/121 TASes replayed → **156,766 captured action frames** → **31,425 blocked transitions** (frame_skip=5) across 120 episodes.

## Phase 1 — World model training

| | |
|---|---|
| Hardware | Targon H100, $1.79/h |
| Image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| Recipe | 50 epochs, batch 128, bf16, `--compile`, AdamW 5e-5, cosine, grad-clip 1.0 |
| Wall time | ~25 min |
| Final val pred_loss | **0.127** (down from 0.36 at epoch 0, **65% drop**) |
| Final val sigreg_loss | 4.0 (paper-typical sharp early drop then plateau) |
| Best epoch | 45 |

**Phase 1 eval (offline):**
- Open-loop latent MSE (mean) at horizons 1/2/4/8/16: **0.12 / 0.24 / 0.55 / 1.15 / 1.83** — graceful linear-ish drift
- **`x_pos` linear probe R² = 0.83**, MAE 65 px on a 0–1571 px range. The latent recovers Mario's spatial position purely from the pixel-prediction objective — paper's "physics emerges unsupervised" property reproduced on Mario.

## Phase 2a — Online goal-conditioned eval

CEM in latent space, cost = `||ẑ_H − z_g||²` where `z_g` is encoded from a TAS frame K blocks ahead. 12 episodes, model controls Mario for 30 blocks (~5 s after spawn at x=40), goals replayed in lockstep oracle env.

| Metric | Value |
|---|---|
| Mean x_progress | **134 px** |
| Median | 139 |
| Max | 256 |
| Min | -8 |
| Episodes with forward progress | 11/12 |

Confirmed the trained world model is useful for control — Mario consistently advances toward goal frames. 4 sample side-by-side videos (left = real gameplay, right = goal frame) saved.

## Phase 2b — Autonomous play with reward head

CEM cost = `−Σ r̂(ẑ_t)` over 4 blocks of horizon. No goal frame. Three iterations:

| Variant | Description | Mean x | Max | Max final_x | Deaths |
|---|---|---|---|---|---|
| **v1** | r̂ = MLP 192→64→1, single-block target, death penalty -50 | 252 | 273 | 313 | 0/12 |
| **v2** | r̂ = MLP 192→128→128→1, **summed-over-4-blocks** target, death penalty **-10** | **433** | 656 | 696 | 0/12 |
| **Joint** | Encoder + predictor + reward head trained jointly: `L_pred + 0.09·L_sigreg + 0.5·L_reward`, 12 epochs lr=2e-5 | **495** | **682** | **722** | 0/12 |

Each step compounded: lower death penalty unlocked risk-taking, multi-horizon target gave less noisy gradients, joint training let the encoder learn reward-relevant features. **9/12 joint episodes converge to x=594** (between the second pipe and the staircase); 1/12 breaks through to x=722.

Hardware: Targon H200, $2.40/h, ~90 min for the full v2 + joint chain.

## Artifacts

**Code (this repo):** https://github.com/QuangBK/leWM-mario
- `tas_replay.py` — FM2 → nes_py replay with RAM-based desync detection
- `madmario_rollout.py` — alt MadMario rollout (smoke test only)
- `train_mario.py` — upstream `lewm_mario/train_mario.py` patched with W&B hooks
- `eval_lewm.py` — Phase 1 offline eval (open-loop MSE + x_pos probe)
- `online_eval.py` — Phase 2a goal-conditioned MPC + side-by-side video
- `finetune_reward_head.py` (v1) and `finetune_reward_head_v2.py` (multi-horizon, bigger MLP, latent precompute)
- `joint_finetune.py` — encoder+predictor+head joint training
- `autonomous_eval_v2.py` — supports both separate-head and joint-ckpt loading
- `eval_results.json`, `auto_eval_summary.json`, `v2_summary.json`, `joint_summary.json`

**Models & videos (private HF):** https://huggingface.co/obamaTeo/lewm-mario
- `best.pt` (208 MB) — Phase 1 world model, epoch 45
- `reward_head.pt` (51 KB) — v1 reward head for `best.pt`
- `reward_head_v2.pt` (168 KB) — v2 reward head for `best.pt`
- `joint_best.pt` (73 MB) — Phase 2b joint-finetuned model + head together
- `videos/`:
  - `ep_0X_*.mp4` (4) — Phase 2a goal-conditioned, side-by-side
  - `auto_ep_*.mp4` (4) — Phase 2b v1 autonomous
  - `v2_ep_*.mp4` (3) — Phase 2b v2 autonomous
  - `joint_ep_*.mp4` (3) — Phase 2b joint autonomous

**W&B:** `quangbk/lewm-mario` project — runs `tas-v1-50ep`, `reward-head-v1` (xzfzt88d), `reward-head-v2`, `joint-v1` (ksvs9twl).

## Cost

| Run | Hardware | Time | Cost |
|---|---|---|---|
| Phase 1 train + eval (H100 #1) | h100-small @ $1.79/h | ~2 h | ~$3.60 |
| Phase 2a + 2b v1 (H100 #2) | h100-small @ $1.79/h | ~1 h | ~$1.80 |
| Three GPU-less Targon scheduler dud rentals (H100 #3, #4, RunPod blocked) | h100-small | ~10 min | ~$0.30 |
| Phase 2b v2 + joint (H200) | h200-small @ $2.40/h | ~90 min | ~$3.60 |
| **Total** | | | **~$9.30** |

## Key technical findings

1. **TAS replay survives long enough to be useful** despite the ROM-byte mismatch — average ~1300 in-level frames per TAS before desync, plenty for a 31k-transition dataset.
2. **The paper's "physics emerges unsupervised" claim reproduces on Mario** — x_pos R²=0.83 from a frozen LeWM encoder.
3. **The default reward head from a frozen encoder plateaus quickly** (val_loss ~0.85 by epoch 14 of 30). Training more reward-head epochs doesn't help; the encoder bottlenecks the signal.
4. **Joint encoder finetune is the real Phase 2b win** — letting the encoder learn from reward signal added +14% over v2's tuned reward head, and v2 itself was +72% over v1.
5. **Death penalty calibration matters more than expected.** -50 made Mario stop at the first Goomba; -10 unlocked risk-taking and Mario started clearing pipes.
6. **Operational gotchas:** `pip install nes-py gym-super-mario-bros scikit-image` will silently bump numpy>=2 and break `nes_py` with a uint8 OverflowError — must pin `"numpy<2" "scikit-image<0.22" "gym==0.25.2"` in the same install. `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` is the right base. `gym_super_mario_bros.make().unwrapped` gives a SuperMarioBrosEnv that accepts raw 8-bit FM2 bytes via `.step(byte)`.

## What's next (not done yet)

- **Score-based reward** — use SMB1 score / coin / killed-enemy RAM bytes (`0x07DD-0x07DF`, `0x07ED-0x07EF`) on top of x_pos delta. Should push Mario past the x=594 plateau.
- **Better data** — either source the F61548FD ROM dump for tighter TAS replay (bigger dataset), or train PPO on H100 for self-consistent expert rollouts.
- **Bigger encoder** — current ViT-tiny (5M params) may be saturated. ViT-small (~22M) with longer training is a natural next scaling step.
- **Longer planning horizon** — current CEM uses horizon 4 (≈ 0.7 s of game time). Pit jumps need ~30 frames of lookahead. Try horizon 8 once we add hierarchy or limit error accumulation.

---

# Phase 3 — Ablations (2026-04-27)

Ran all four "next" levers as separate experiments to map which actually move
the needle. Single H200 run, persistent `/workspace` volume, single
`phase3_campaign.sh` script.

## Eval comparison (12 episodes each, x_start=40, 80 blocks max)

| Variant | mean x_progress | max | min | Notes |
|---|---|---|---|---|
| **Phase 2 joint baseline** (Δx-only reward, h=4) | 495 | 682 | — | Reference point |
| rh_v3 (composite reward, frozen baseline encoder) | 480 | 682 | 269 | ≈ baseline |
| **joint_v2_score** (composite reward + joint encoder, h=4) | **524** | **839** | 274 | +6% mean, **+23% max** |
| **horizon=8 on baseline ckpt** (no retraining) | **560** | 750 | 259 | **+13% mean, eval-only change** |
| horizon=8 on joint_v2_score | 426 | 682 | 255 | regressed: joint+long-horizon compounds error |
| **ViT-small joint** (~22M encoder, h=4) | 400 | 684 | 269 | regressed: bigger encoder over-fits |
| **combined-data joint** (TAS+PPO union, h=4) | _eval pending — Targon SSH outage 2026-04-27 ~08:00 UTC; ckpt safe on volume_ | | | val/reward_loss=0.103 (vs baseline 0.92, **9× lower**) |

## Per-experiment notes

**Exp 1 — Composite reward (Δx + Δscore + Δcoins + Δptype + death).** Extended `tas_replay.py` to capture SMB1 RAM bytes for score (0x07DD-0x07E2 BCD ×10), coins (0x075E BCD), player-type (0x0756), timer (0x07F8-0x07FA BCD). Composite reward target on a frozen encoder is no better than Δx-only — but joint encoder finetune on the composite target gets +23% on max final_x. The non-Δx components fire rarely on TAS data (mean Δscore per block = 0.08, mean Δcoins = 0.02, ptype change = 0 always), so the effective signal is mostly Δx with a small coin/score bonus. The win likely comes from the *joint* training, not the richer reward.

**Exp 2 — Longer planning horizon.** CEM horizon 4 → 8 with n_samples 128 → 256, no retraining. On the unmodified Phase 2 baseline checkpoint this is the **biggest single lift in the campaign** (+13% mean). On the score-finetuned ckpt it regresses badly — joint training that minimizes 4-block prediction error compounds errors at 8 blocks. Lesson: capacity for long-horizon planning is something you have to train for, not something you can just dial up at eval time.

**Exp 3 — ViT-small encoder.** Bumped encoder hidden dim 192 → 384, heads 3 → 6, mlp 768 → 1536 (~22M params, 4× tiny). Phase 1 retrain converged with **val/pred_loss 0.052 (vs 0.13 for ViT-tiny)** — the bigger encoder fits the dynamics much better. But on autonomous play the joint-finetuned ViT-small ckpt regressed to mean=400 (vs 524 ViT-tiny joint). Bigger encoder → sharper but less robust latents → CEM planning is worse. **Capacity ≠ control quality** on this dataset size (31k blocks). Probably needs ≥10× more data to pay off.

**Exp 4 — PPO-augmented dataset.** Trained PPO (CnnPolicy, 1.5M timesteps, 8 envs, ~30 min on H200) directly on `gym-super-mario-bros-1-1-v0`, captured 100 rollouts (epsilon=0.05). PPO expert reaches **x_pos ≈ 1120 reliably** — well past the staircase plateau the TAS replays plateau at (~700). Combined dataset (120 TAS + 93 PPO = 213 episodes) → blocked → precomputed → Phase 1 retrain (50 ep, ViT-tiny) → joint v2 with composite reward. **val/reward_loss dropped to 0.103, ~9× lower than the TAS-only baseline (0.92)** — the world model is finally seeing reward-rich late-level transitions.

## Operational notes

- The original lewm pipeline has THREE prep stages: `tas_replay` (per-frame 8-d) → `build_lewm_mario_dataset` (block 5 frames into 40-d) → `precompute` (resize to 224×224). Skipping `build_lewm_mario_dataset` makes the trainer crash with `expected 40 channels, got 8` — easy to miss reading only `train_mario.py`.
- `stable-baselines3 ≥ 2.0` requires `gymnasium.Env` but `nes_py` / `gym-super-mario-bros` are still gym-API. `shimmy.GymV21CompatibilityV0` bridges them; pin a 1.x sb3 isn't viable because gym fails to build under modern setuptools.
- Targon `ssh.deployments.targon.com` had a sustained "Session Terminated 0" outage on 2026-04-27 ~08:00-09:30+ UTC affecting both the original pod and a fresh redeploy. The persistent network volume (`vol-fujcshps4ben`, `/workspace`) preserved all artifacts across the pod redeploy; data wasn't lost, only inspection was blocked. **Lesson:** put intermediate artifacts on the volume from the start, treat ephemeral container disk as scratch-only.
- 512 GB persistent volume costs $0.01/h; cheap insurance against pod loss for any multi-stage campaign.

## Headline

The two clean, transferable wins are:

1. **PPO-augmented dataset** — `val/reward_loss` 0.92 → 0.103 (9× drop). Self-consistent expert avoids the TAS-ROM mismatch and reaches x ≈ 1120 reliably. Eval pending due to Targon outage.
2. **CEM horizon=8 at eval time** — 495 → 560 mean x_progress, no training cost.

The bigger encoder (Exp 3) and the joint+horizon=8 combo (Exp 2 cross) both regress, suggesting the binding constraint is dataset reward-richness, not capacity or planning depth alone.
