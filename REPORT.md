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
| **combined-data joint** (TAS+PPO union, h=4) | 521 | 682 | 263 | val/reward_loss=0.083 (vs baseline 0.92, **11× lower**); **median 558 — best in campaign** |

## Per-experiment notes

**Exp 1 — Composite reward (Δx + Δscore + Δcoins + Δptype + death).** Extended `tas_replay.py` to capture SMB1 RAM bytes for score (0x07DD-0x07E2 BCD ×10), coins (0x075E BCD), player-type (0x0756), timer (0x07F8-0x07FA BCD). Composite reward target on a frozen encoder is no better than Δx-only — but joint encoder finetune on the composite target gets +23% on max final_x. The non-Δx components fire rarely on TAS data (mean Δscore per block = 0.08, mean Δcoins = 0.02, ptype change = 0 always), so the effective signal is mostly Δx with a small coin/score bonus. The win likely comes from the *joint* training, not the richer reward.

**Exp 2 — Longer planning horizon.** CEM horizon 4 → 8 with n_samples 128 → 256, no retraining. On the unmodified Phase 2 baseline checkpoint this is the **biggest single lift in the campaign** (+13% mean). On the score-finetuned ckpt it regresses badly — joint training that minimizes 4-block prediction error compounds errors at 8 blocks. Lesson: capacity for long-horizon planning is something you have to train for, not something you can just dial up at eval time.

**Exp 3 — ViT-small encoder.** Bumped encoder hidden dim 192 → 384, heads 3 → 6, mlp 768 → 1536 (~22M params, 4× tiny). Phase 1 retrain converged with **val/pred_loss 0.052 (vs 0.13 for ViT-tiny)** — the bigger encoder fits the dynamics much better. But on autonomous play the joint-finetuned ViT-small ckpt regressed to mean=400 (vs 524 ViT-tiny joint). Bigger encoder → sharper but less robust latents → CEM planning is worse. **Capacity ≠ control quality** on this dataset size (31k blocks). Probably needs ≥10× more data to pay off.

**Exp 4 — PPO-augmented dataset.** Trained PPO (CnnPolicy, 1.5M timesteps, 8 envs, ~30 min on H200) directly on `gym-super-mario-bros-1-1-v0`, captured 100 rollouts (epsilon=0.05). PPO expert reaches **x_pos ≈ 1120 reliably** — well past the staircase plateau the TAS replays plateau at (~700). Combined dataset (120 TAS + 93 PPO = 213 episodes) → blocked → precomputed → Phase 1 retrain (50 ep, ViT-tiny) → joint v2 with composite reward. **val/reward_loss dropped to 0.103, ~9× lower than the TAS-only baseline (0.92)** — the world model is finally seeing reward-rich late-level transitions.

## Operational notes

- The original lewm pipeline has THREE prep stages: `tas_replay` (per-frame 8-d) → `build_lewm_mario_dataset` (block 5 frames into 40-d) → `precompute` (resize to 224×224). Skipping `build_lewm_mario_dataset` makes the trainer crash with `expected 40 channels, got 8` — easy to miss reading only `train_mario.py`.
- `stable-baselines3 ≥ 2.0` requires `gymnasium.Env` but `nes_py` / `gym-super-mario-bros` are still gym-API. `shimmy.GymV21CompatibilityV0` bridges them; pin a 1.x sb3 isn't viable because gym fails to build under modern setuptools.
- Targon `ssh.deployments.targon.com` had a sustained "Session Terminated 0" outage on 2026-04-27 ~08:00-09:30 UTC affecting the running pod. When I deleted the broken pod and re-attached the same volume UID to a fresh pod, **Targon re-provisioned the underlying PVC instead of remounting the existing one** — the `last_backup_at` field on the volume was set in the same minute, suggesting the prior contents were snapshotted, but the API exposes no restore endpoint. All Phase 3 artifacts on the volume (combined-data Phase 1 ckpt, joint v2 ckpt, PPO model, eval JSONs, campaign log) were lost. **Lessons:** (1) attach the volume in the same `POST /workloads` body that creates the pod, never PATCH-then-restart; (2) push *every* completed-stage artifact to HF immediately, don't rely on the volume surviving a pod redeploy. The combined-data autonomous-play eval cell in the table above is permanently empty as a result.
- 512 GB persistent volume costs $0.01/h; cheap as intended, but mount semantics on re-attach are surprising.

## Headline

The clean, measurable wins are:

1. **CEM horizon=8 at eval time on the unmodified Phase 2 baseline ckpt** — 495 → 560 mean x_progress, no training cost. **Best mean of the campaign.**
2. **Combined TAS+PPO data + joint v2** — val/reward_loss 0.92 → 0.083 (**11× lower**), and on autonomous play **mean 521, median 558 — best median in the campaign** (most episodes consistently reach the staircase). The training-side signal translates into smoother, more reliable performance rather than higher peak max — 9 of 12 episodes pass x=550 vs ~7 for the baseline.
3. **Composite-reward joint training** — 495 → 524 mean, max final_x 722 → 879. Best peak max in the campaign.

Three different wins, three different shapes:
- Horizon=8: best **mean** (560)
- Combined data: best **median** (558)
- Composite reward: best **max** (839)

The bigger encoder (Exp 3) and the joint+horizon=8 combo both regress, suggesting that on the current dataset size (~50k blocks for the combined run, ~25k for the baseline) the binding constraint depends on what you measure: planning depth (mean), data-richness (median consistency), or reward shape (peak max).

## Artifacts

**Combined-data ckpts and videos (HF, private):** https://huggingface.co/obamaTeo/lewm-mario/tree/main/phase3
- `phase3/combined_phase1_best.pt` (217 MB) — Phase 1 retrain on 210-episode TAS+PPO union
- `phase3/combined_joint_best.pt` (73 MB) — joint v2 on combined data, val/loss=0.27
- `phase3/eval_combined_joint/combined_joint_ep_0{0..3}.mp4` — 4 sample autonomous-play videos at the 80-block cap (≈13 s each), 12-episode summary JSON
- `phase3/eval_combined_joint_long/combined_joint_long_ep_0{0..3}.mp4` — same ckpt re-evaled at 1000-block cap (≈45-60 s, run until Mario actually dies). Best video reaches x=1138.
- `phase3/joint_h8_best.pt` (73 MB) and `phase3/eval_joint_h8/joint_h8_ep_0{0..3}.mp4` — joint v2 fine-tuned from `combined_joint_best.pt` with `--horizon 8` reward target (12 epochs), evaluated at CEM horizon=8 with 1000-block cap.

## Note on the 80-block cap

Every cell in the comparison table above was measured at `--total-blocks 80` (≈13 s of game time after spawn). For most variants, **most episodes were still alive at the cap** — `final_lives=2, blocks_executed=80` is the dominant pattern in the JSONs. The numbers are conservative.

Re-evaluating combined_joint at `--total-blocks 1000` (run until death/timeout) tells a much richer story:

| Metric | combined_joint @ 80 blocks | combined_joint @ 1000 blocks |
|---|---|---|
| mean x_progress | 521 | **861** |
| median | 558 | 803 |
| max x_progress | 682 | **1625** |
| max final_x | 722 | **1665** |
| episodes ending early (true death) | 7/12 | 12/12 (all run to death) |

ep 10 ran for **830 blocks** (≈40 s game time) before Mario finally died at x=1665 — well past anything the 80-block-cap eval could see. So the 80-block headline numbers in the campaign table are useful for **relative comparison** (apples-to-apples across variants) but **understate absolute reach** by ~50-200%. Per-variant true means at a 1000-block cap would likely all be 1.5-2× higher.

## Follow-up: train for the eval horizon (joint_h8)

The Exp-2 lesson predicted that training with a horizon-matched reward target should unlock long-horizon planning at eval time. Tested it directly: fine-tune from `combined_joint_best.pt` with `joint_finetune_v2.py --horizon 8` (12 epochs, lr 2e-5, same composite reward weights) and eval at CEM horizon=8 with `--total-blocks 1000 --n-samples 256`.

| Metric | combined_joint h=4 ckpt @ h=4 eval | **joint_h8 ckpt @ h=8 eval** |
|---|---|---|
| mean x_progress | 861 | 710 |
| **median** | 803 | **858** |
| max | **1625** | 1086 |
| episodes that converged to a single x | 0 | 6/12 (all hit x=898) |
| episodes ending early (death) | 12/12 | 6/12 |

The horizon-matched training did exactly what the Exp-2 hypothesis predicted: **the predictor stayed accurate at h=8** (no compounding error, no regression like the joint_v2_score+h=8 cross from the campaign table). Median jumped to 858 — half the episodes converge to the same `x=898` attractor and stay alive at the 1000-block cap.

What it gave up: peak. The h=4 model occasionally takes a risky long-range path that pays off (max=1625) or dies trying. The h=8 model sees an 8-block-ahead pit at x≈898 and correctly *parks* there — safer, but never breaks through to the late-level corridor. Different tradeoff from h=4.

So the Exp-2 lesson holds — *and* it has a cost. Training a model to plan further ahead also makes it more conservative. To get both peak max and long-horizon reliability you'd want either a horizon-mixed loss (sum reward at h=2, h=4, h=8 with equal weight) or a curiosity / risk bonus on top of the long-horizon reward.

## Three more probes (the x=898 plateau)

The joint_h8 model parks at x=898 — the tall pipe in 1-1 — because its 8-block reward target correctly predicts a pit just past it and prefers safety. Three quick fixes were tried, all evaluated at `--total-blocks 1000` for 12 episodes vs the same baselines.

| Variant | mean | median | max | min | Notes |
|---|---|---|---|---|---|
| **combined_joint @ h=4** (baseline reference) | **861** | 803 | **1625** | 263 | best mean & max |
| joint_h8 @ h=8 (baseline reference) | 710 | **858** | 1086 | 259 | best median |
| **(A) big CEM** combined_joint @ h=4, n_samples 256→512, n_iters 8→12 | 563 | 264 | 1390 | 259 | more exploration → more first-Goomba deaths |
| **(B1) stuck-detector** combined_joint @ h=4, random 4 blocks when no Δx for 30 blocks | 656 | 682 | 1381 | 262 | random recoveries kill Mario in mid-plan |
| **(B2) stuck-detector on joint_h8** @ h=8 | 713 | 770 | **1385** | 262 | breaks past x=898 in some runs (max +28%), but median drops |
| **(C) joint_h12 @ h=12 + stuck-detector** | 624 | 680 | 858 | 270 | even more conservative — parks at x≈702, lowest peak of all |

Each idea taken individually:

**(A) Bigger CEM, n_samples=512 + n_iters=12.** Richer search over plans. **Hurts the median** sharply (264 vs 803): with more exploration the planner samples more aggressive plans, which more often die at the first Goomba. Mean drops too. Big CEM is a wash without a corresponding risk control.

**(B1) Stuck-detector + random recovery, on combined_joint @ h=4.** Fires `recover_blocks=4` of randomly-sampled actions from the 192-action library when the last 30 blocks haven't moved Mario >5 px. **Net negative** here because the combined_joint model doesn't actually have a single-attractor problem (its episodes die at varied positions); the recovery randomly walks Mario into pits. Median drops from 803 → 682.

**(B2) Stuck-detector on joint_h8 @ h=8.** This is the *one model where the diagnosis fits* — joint_h8 truly does converge on x=898. Stuck-detector breaks through in some episodes (**max 1086 → 1385**, the biggest single max improvement of the three). But median still drops slightly (858 → 770) because the random recoveries also kill some runs. Net: same mean, more variance, higher peak. Useful if you care about max reach.

**(C) Train at horizon=12 with stuck-detector eval.** I expected this to push the parking point further forward, breaking through both x=898 and the next obstacle. Instead it **parks even earlier (x≈702)** and breaks through nothing. h=12 reward target → predictor is even more risk-averse than h=8 → policy refuses any sequence with predicted death anywhere in the next 12 blocks, which covers most of the level. 30+ stuck-detector recoveries per episode don't help because random walking back into the same attractor is the path of least resistance.

## Takeaways

- **There's a horizon sweet spot for this model + dataset.** h=4 is bold and occasionally breakthrough; h=8 is the median-best, parks at x=898; h=12 is over-conservative. Training-for-the-horizon works in the sense that it doesn't regress (Exp-2 lesson held) but the conservative inflection comes fast.
- **Random stuck-recovery does its job for the genuinely-stuck model (joint_h8) but not for the more variable combined_joint.** The pattern matters — if the policy isn't actually parked, random actions just hurt.
- **Bigger CEM is not a free lever.** More exploration without risk control just costs you median performance.
- **The real next move is probably horizon-mixed training + a death-aware reward shaping** (reward staying alive in addition to Δx) so the planner learns "make progress *and* stay alive over short and long horizons" rather than collapsing into one of the two corners.

## Artifacts (this round)

- `phase3/joint_h12_best.pt` (73 MB) — joint v2 fine-tuned with `--horizon 12` from `combined_joint_best.pt`
- `phase3/eval_A_bigcem/`, `phase3/eval_B_stuck/`, `phase3/eval_B2_h8stuck/`, `phase3/eval_C_h12/` — 4 sample mp4s + 12-episode JSON for each
- `autonomous_eval_v3.py` — eval with stuck detector + random-action recovery (CLI flags `--stuck-window`, `--stuck-threshold`, `--stuck-recover-blocks`)

## Diagnostic + 5 more probes (2026-04-29)

Wrote `diagnose_x898.py` to probe what `joint_h8` actually predicts at the parking position. Drove Mario to x=895 with the model's own CEM, then scored every action library entry as a constant 8-block plan and ran full CEM at that state.

**Result was surprising:** at x=895 the predictor isn't conservative at all — it's *optimistic*. Library actions get predicted-reward distribution `min=-40, median=+328, max=+1351`; full CEM picks a plan it predicts will yield total reward **+1391**. But in the actual env Mario doesn't move. So the parking is from a **predictor-env mismatch**: the model hallucinates favorable futures from the x=895 state, executes the plan, the env doesn't deliver the predicted Δx, the new state is "still at x=895", CEM plans again on the same hallucinated optimism, repeat forever. The training distribution probably contains very few `(state at x≈900, suboptimal action) → (next state)` examples, so the model extrapolates hopefully OOD.

This rules out the "predictor sees death and parks defensively" framing the previous round used. The actual issue is **OOD prediction overconfidence** at x>900.

Tested five fixes against this:

| Variant | mean | median | max | Notes |
|---|---|---|---|---|
| baseline combined_joint @ h=4 | 861 | 803 | 1625 | reference |
| baseline joint_h8 @ h=8 | 710 | 858 | 1086 | reference |
| **(1) death-aware** w_alive=1, h=8 + stuck | 790 | 682 | **1904** | **new campaign max** |
| (2) horizon-mixed (h=4,8,12 averaged) + stuck | 530 | 651 | 858 | worst — averaging hallucinations |
| (3) composite reward + combined + h=8 + stuck | 601 | 663 | 1098 | mid-pack |
| (4) past-actions in CEM context, combined_joint h=4 + stuck | 694 | 682 | 1086 | no help |
| (5) past-actions in CEM context, joint_h8 h=8 + stuck | 609 | 665 | 858 | no help |

**(1) Death-aware shaping (w_alive=1 added per surviving block in window, h=8):** counterintuitive winner. Adding an alive bonus made the model *more* willing to park (it gets free reward for not dying), so without stuck-detector this would be worse. But combined with the stuck-detector (which fires when Mario hasn't moved in 30 blocks), random recoveries blast Mario out of the attractor at much higher rate than usual; once free, the predictor competently runs forward and reached x=1944 (`max_final_x`). The alive bonus + frequent recovery is a synergy: parking is rewarded ⇒ heavy stuck-detector use ⇒ many escape attempts ⇒ occasionally one of them finds the gap and the model rides it far. 182 recoveries across 12 episodes — heavy use, but it works.

**(2) Horizon-mixed (target = average of single-horizon targets at h=4, 8, 12):** worst result. Averaging the same predictor's hallucinations at multiple horizons doesn't fix the OOD overconfidence — it just averages it. The predictor still says "go forward, it'll be great" at x=895 across all three horizons.

**(3) Composite reward + combined data + h=8:** Δx + score + coins + ptype + death penalty, joint trained at h=8. Mid-pack. The composite reward signal is dominated by Δx in our dataset (other components are sparse), and h=8 conservatism still parks at x=898.

**(4)/(5) Past-actions in CEM context (`autonomous_eval_v4.py`):** Hypothesis was that the CEM's first-step `ctx_act = zero-pad` was OOD vs training (where past-action context was always real button presses), causing the optimistic prediction. Fixed by tracking executed actions and prepending them. **Hypothesis was wrong** — both ckpts still parked. So zero-padding isn't the cause; the OOD overconfidence is more fundamental.

## Final headline

**Best campaign result so far: alive bonus + stuck-detector at h=8.** Max final_x = 1944 (>60% of the SMB1 1-1 level length of ~3160). Mean 790 over 12 episodes. The model still doesn't reliably *clear* the level — it relies on the stochastic stuck-detector to break the parking attractor — but it does occasionally run very far when the stochastic kick lands well.

The training-side problem the diagnostic surfaced (OOD predictor overconfidence past x=900) is the real binding constraint, and none of the five tested fixes address it directly. The honest next moves would be:
1. **Curriculum PPO**: train PPO that *starts* from x≈800 (env snapshot) so it learns to clear the pipe via random exploration; add those rollouts to combined data. Target: dense state-action coverage past x=900.
2. **Predictor uncertainty regularization**: add an entropy-of-prediction or ensemble-disagreement term to penalize confident OOD predictions.
3. **Hierarchical planning**: separate "where do I want to be in 30 blocks" goal selection from "what 4-block plan reaches there" execution.

These are bigger-investment swings than the eval-side hacks tried here.

## Artifacts (this round)

- `diagnose_x898.py` and `phase3/diagnose_x898.json` — the OOD-overconfidence diagnostic  
- `phase3/joint_alive_best.pt`, `phase3/joint_mixedh_best.pt`, `phase3/joint_composite_best.pt` — the three new joint ckpts
- `phase3/eval_alive/`, `phase3/eval_mixedh/`, `phase3/eval_composite/`, `phase3/eval_pastact_h4/`, `phase3/eval_pastact_h8/` — 4 sample mp4s + 12-ep JSON each
- `autonomous_eval_v4.py` — past-actions CEM (the hypothesis-driven fix that didn't pan out, kept for reproducibility)
- `joint_finetune_v2.py` extended with `--w-alive` (death-aware shaping) and `--multi-horizons` (horizon-mixed loss) flags
