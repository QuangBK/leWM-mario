"""Joint train v3: adds RC-aux (reachability-correction auxiliary objective).

Extends joint_finetune_v2.py with:
  - --rc-aux flag enables the reachability head + per-horizon prediction
    weights from Li et al. 2026 (arXiv 2605.07278).
  - Multi-horizon prediction loss with optional non-uniform weights.
  - ReachabilityHead trained with positives (within budget along trajectory),
    temporal hard negatives (h<Δ), batch (cross-trajectory) negatives, and
    optional predicted-rollout pairs with stop-grad.

When --rc-aux is NOT set, behavior is identical to v2.
"""
from __future__ import annotations
import argparse, json, sys, time, warnings, math, random
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, "/workspace/lewm_mario_pkg")
from mario_lewm.model import LeWorldModel, LeWorldModelConfig, SIGReg
from mario_lewm.dataset import discover_episodes, MarioTraceDataset
from rc_aux import (
    ReachabilityHead, compute_reachability_loss,
    multi_horizon_pred_loss, rollout_open_loop,
)

BLOCK = 5

class RewardHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z): return self.net(z).squeeze(-1)

class WindowDS(torch.utils.data.Dataset):
    def __init__(self, base: MarioTraceDataset, raw_root: Path, history: int,
                 horizons, w_x: float, w_score: float, w_coins: float,
                 w_ptype: float, w_death: float, w_alive: float = 0.0,
                 milestone_x: int = 0, milestone_bonus: float = 0.0):
        """horizons can be a single int or a list of ints (multi-horizon mix).
        For multi-horizon, the target is the SUM of single-horizon targets across
        each listed horizon, equally weighted.

        milestone_x / milestone_bonus: if a block in the window crosses x_pre <
        milestone_x <= x_post, add `milestone_bonus` to the target. Used to
        amplify the reward signal at known choke-points (e.g. tall pipe at x=898).
        """
        self.base = base; self.history = history
        if isinstance(horizons, int):
            horizons = [horizons]
        self.horizons = horizons
        self.milestone_x = int(milestone_x)
        self.milestone_bonus = float(milestone_bonus)
        cache = {}
        for ep in self.base.episodes:
            with np.load(Path(raw_root) / f"{ep.name}.npz", allow_pickle=False) as d:
                fields = d.files
                cache[ep.name] = {
                    "x":     np.asarray(d["x_pos"]),
                    "lives": np.asarray(d["lives"]) if "lives" in fields else np.zeros_like(d["x_pos"]),
                    "score": np.asarray(d["score"]) if "score" in fields else np.zeros_like(d["x_pos"]),
                    "coins": np.asarray(d["coins"]) if "coins" in fields else np.zeros_like(d["x_pos"]),
                    "ptype": np.asarray(d["ptype"]) if "ptype" in fields else np.zeros_like(d["x_pos"]),
                }

        ms_x = self.milestone_x; ms_b = self.milestone_bonus
        def single_horizon_target(ar, first_block, H):
            total, died = 0.0, False
            blocks_alive = 0
            comps_local = {"x": 0.0, "score": 0.0, "coins": 0.0, "ptype": 0.0,
                           "death": 0.0, "alive": 0.0, "milestone": 0.0}
            for k in range(H):
                blk = first_block + k
                i_pre  = blk * BLOCK
                i_post = min((blk + 1) * BLOCK, len(ar["x"]) - 1)
                if i_pre >= len(ar["x"]): break
                x_pre  = float(ar["x"][i_pre])
                x_post = float(ar["x"][i_post])
                dx     = x_post - x_pre
                dscore = float(ar["score"][i_post]) - float(ar["score"][i_pre])
                dcoins = float(ar["coins"][i_post]) - float(ar["coins"][i_pre])
                dptype = float(ar["ptype"][i_post]) - float(ar["ptype"][i_pre])
                r = w_x*dx + w_score*dscore + w_coins*dcoins + w_ptype*dptype + w_alive
                # milestone bonus: block crosses milestone_x in this 5-frame window
                if ms_x > 0 and x_pre < ms_x <= x_post:
                    r += ms_b
                    comps_local["milestone"] += ms_b
                total += r
                comps_local["x"]     += w_x*dx
                comps_local["score"] += w_score*dscore
                comps_local["coins"] += w_coins*dcoins
                comps_local["ptype"] += w_ptype*dptype
                comps_local["alive"] += w_alive
                blocks_alive += 1
                if i_post < len(ar["lives"]) and i_pre < len(ar["lives"]) \
                   and int(ar["lives"][i_post]) < int(ar["lives"][i_pre]):
                    died = True; break
            if died:
                total += w_death
                comps_local["death"] = w_death
            return total, comps_local

        N = len(self.base.index)
        targets = np.zeros(N, dtype=np.float32)
        first_block_x_pre = np.zeros(N, dtype=np.int32)  # x at start of first predicted block
        first_block_dx    = np.zeros(N, dtype=np.float32)  # Δx of first predicted block
        comps_keys = ("x","score","coins","ptype","death","alive","milestone")
        comps = {k: np.zeros(N) for k in comps_keys}
        for w, (ep_id, start) in enumerate(self.base.index):
            ar = cache[self.base.episodes[ep_id].name]
            first_block = start + history - 1
            total = 0.0
            for H in self.horizons:
                t, c = single_horizon_target(ar, first_block, H)
                total += t / len(self.horizons)  # average across horizons
                for k, v in c.items():
                    comps[k][w] += v / len(self.horizons)
            targets[w] = total
            # cache first-block x_pre and dx for downstream oversampling logic
            i_pre  = first_block * BLOCK
            i_post = min((first_block + 1) * BLOCK, len(ar["x"]) - 1)
            if i_pre < len(ar["x"]):
                first_block_x_pre[w] = int(ar["x"][i_pre])
                first_block_dx[w]    = float(ar["x"][i_post]) - float(ar["x"][i_pre])
        self.targets = targets
        self.first_block_x_pre = first_block_x_pre
        self.first_block_dx    = first_block_dx
        self.mean = float(targets.mean()); self.std = float(targets.std() + 1e-6)
        print(f"reward target: mean={self.mean:.2f}  std={self.std:.2f}  horizons={self.horizons}")
        for k, v in comps.items():
            print(f"  comp[{k:9s}]: mean={v.mean():+.3f}  std={v.std():.3f}  "
                  f"min={v.min():+.1f}  max={v.max():+.1f}")

    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        item["target_reward"] = torch.tensor((self.targets[idx] - self.mean) / self.std, dtype=torch.float32)
        return item

class EpisodeBatchSampler(Sampler):
    def __init__(self, indices, dataset_index, batch_size, shuffle, drop_last, seed):
        self.indices = list(indices); self.idx = dataset_index
        self.batch_size = batch_size; self.shuffle = shuffle
        self.drop_last = drop_last; self.seed = seed; self.epoch = 0
        self.by_ep = defaultdict(list)
        for local, g in enumerate(self.indices):
            self.by_ep[self.idx[g][0]].append(local)
    def set_epoch(self, e): self.epoch = e
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        ep_ids = list(self.by_ep.keys())
        if self.shuffle: rng.shuffle(ep_ids)
        out = []
        for ep_id in ep_ids:
            pos = list(self.by_ep[ep_id])
            if self.shuffle: rng.shuffle(pos)
            for s in range(0, len(pos), self.batch_size):
                b = pos[s:s+self.batch_size]
                if len(b) < self.batch_size and self.drop_last: continue
                out.append(b)
        if self.shuffle: rng.shuffle(out)
        return iter(out)
    def __len__(self):
        return sum((len(p) // self.batch_size) if self.drop_last else math.ceil(len(p)/self.batch_size)
                   for p in self.by_ep.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/best.pt")
    ap.add_argument("--dataset-root", default="/workspace/data/tas_precomputed")
    ap.add_argument("--raw-root", default="/workspace/data/tas_full")
    ap.add_argument("--out-dir", default="/workspace/runs/joint_v2", type=Path)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--multi-horizons", type=str, default="",
                     help="Comma list of horizons to average target over, e.g. '4,8,12'. Overrides --horizon if set.")
    ap.add_argument("--w-x", type=float, default=1.0)
    ap.add_argument("--w-score", type=float, default=0.05)
    ap.add_argument("--w-coins", type=float, default=5.0)
    ap.add_argument("--w-ptype", type=float, default=20.0)
    ap.add_argument("--w-death", type=float, default=-10.0)
    ap.add_argument("--w-alive", type=float, default=0.0,
                     help="Bonus added per block survived in window (death-aware shaping).")
    # Variant A — oversample windows whose first block falls in a parking-prone
    # x range AND has positive Δx (rare successful-progress samples).
    ap.add_argument("--oversample-x-lo", type=int, default=0,
                     help="Lower bound of first-block x for oversampling (0 disables).")
    ap.add_argument("--oversample-x-hi", type=int, default=0,
                     help="Upper bound of first-block x for oversampling.")
    ap.add_argument("--oversample-mult", type=int, default=1,
                     help="Replication factor for qualifying training samples (1 = no-op).")
    ap.add_argument("--oversample-require-progress", type=int, default=1,
                     help="If 1, only oversample windows whose first block has Δx>0.")
    # Variant B — milestone reward bonus for blocks that cross a target x.
    ap.add_argument("--milestone-x", type=int, default=0,
                     help="If a window block crosses x_pre < milestone_x <= x_post, add bonus to reward target. 0 disables.")
    ap.add_argument("--milestone-bonus", type=float, default=0.0,
                     help="Reward (in raw units, before normalization) added when milestone is crossed.")
    ap.add_argument("--reward-weight", type=float, default=0.5)
    # RC-aux flags
    ap.add_argument("--rc-aux", action="store_true",
                     help="Enable RC-aux (multi-horizon pred + reachability head).")
    ap.add_argument("--rc-weight-mh", type=float, default=1.0,
                     help="Coefficient on the multi-horizon prediction loss (replaces the v2 single mean).")
    ap.add_argument("--rc-weight-reach", type=float, default=0.5,
                     help="Coefficient on the reachability loss (paper's β).")
    ap.add_argument("--rc-h-max", type=int, default=8,
                     help="Max reachability horizon (head's max_horizon and the rollout depth used for pairs).")
    ap.add_argument("--rc-mh-weighting", type=str, default="uniform",
                     choices=["uniform", "linear", "power"],
                     help="Per-horizon weight schedule for the prediction loss.")
    ap.add_argument("--rc-mh-power", type=float, default=1.0,
                     help="Exponent for --rc-mh-weighting=power.")
    ap.add_argument("--rc-pred-weight", type=float, default=0.5,
                     help="Weight on predicted-rollout pair loss (ρ_pred). 0 disables.")
    ap.add_argument("--rc-temporal-neg-weight", type=float, default=1.0,
                     help="Weight on temporal hard negatives. 0 falls back to batch-only negatives (and the head will collapse — see paper §3.2).")
    ap.add_argument("--rc-head-hidden", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--wandb-project", type=str, default="lewm-mario")
    ap.add_argument("--wandb-run-name", type=str, default="joint-v3-rcaux")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])
    model = LeWorldModel(cfg); model.load_state_dict(ck["model_state"])
    model.to(device)
    head = RewardHead(in_dim=cfg.action_embed_dim).to(device)
    sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj).to(device)
    reach_head = None
    if args.rc_aux:
        reach_head = ReachabilityHead(
            embed_dim=cfg.action_embed_dim,
            hidden_dim=args.rc_head_hidden,
            max_horizon=args.rc_h_max,
        ).to(device)
        print(f"RC-aux enabled: head embed_dim={cfg.action_embed_dim} hidden={args.rc_head_hidden} h_max={args.rc_h_max}")
        print(f"  weights: mh={args.rc_weight_mh} reach={args.rc_weight_reach} "
              f"pred={args.rc_pred_weight} temporal_neg={args.rc_temporal_neg_weight}")
        print(f"  pred horizon weighting: {args.rc_mh_weighting} (power={args.rc_mh_power})")

    episodes = discover_episodes(args.dataset_root)
    # For RC-aux we need windows of length history + H_max so we can do
    # autoregressive multi-step rollout against encoded targets. The model
    # architecture (cfg.num_preds) is independent — only the dataset window
    # length changes.
    dataset_num_preds = args.rc_h_max if args.rc_aux else cfg.num_preds
    base = MarioTraceDataset(episodes, cfg.history_size, dataset_num_preds, cfg.image_size,
                              stride=1, npz_load_mode="lazy", max_cached_episodes=8)
    if args.rc_aux:
        print(f"dataset window = history({cfg.history_size}) + num_preds({dataset_num_preds}) = {cfg.history_size + dataset_num_preds} blocks")
    horizons_arg = [int(s) for s in args.multi_horizons.split(",")] if args.multi_horizons.strip() else args.horizon
    ds = WindowDS(base, args.raw_root, cfg.history_size, horizons_arg,
                  args.w_x, args.w_score, args.w_coins, args.w_ptype, args.w_death, args.w_alive,
                  milestone_x=args.milestone_x, milestone_bonus=args.milestone_bonus)
    g = torch.Generator().manual_seed(3072)
    cut = int(len(ds) * 0.9)
    perm = torch.randperm(len(ds), generator=g)
    tr_idx = perm[:cut].tolist(); va_idx = perm[cut:].tolist()
    # Variant A: oversample training samples whose first block hits a parking-prone
    # x range with positive Δx (rare success windows). Validation set untouched.
    if args.oversample_x_lo > 0 and args.oversample_x_hi > 0 and args.oversample_mult > 1:
        x_pre = ds.first_block_x_pre
        dx0   = ds.first_block_dx
        in_range = (x_pre >= args.oversample_x_lo) & (x_pre <= args.oversample_x_hi)
        if args.oversample_require_progress:
            qualifies = in_range & (dx0 > 0)
        else:
            qualifies = in_range
        n_q_total = int(qualifies.sum())
        tr_set = set(tr_idx)
        extra = []
        for w in np.where(qualifies)[0]:
            if int(w) in tr_set:
                # add (mult-1) extra copies of this index
                extra.extend([int(w)] * (args.oversample_mult - 1))
        n_q_train = sum(1 for w in np.where(qualifies)[0] if int(w) in tr_set)
        tr_idx = tr_idx + extra
        print(f"oversample: x∈[{args.oversample_x_lo},{args.oversample_x_hi}] "
              f"require_progress={args.oversample_require_progress} mult={args.oversample_mult}")
        print(f"  qualifying samples: {n_q_total} total, {n_q_train} in train split")
        print(f"  added {len(extra)} extra copies; train_idx grew {len(tr_idx) - len(extra)} → {len(tr_idx)}")
    tr_sampler = EpisodeBatchSampler(tr_idx, base.index, args.batch_size, True, True, 3072)
    va_sampler = EpisodeBatchSampler(va_idx, base.index, args.batch_size, False, False, 3072)
    tr_loader = DataLoader(ds, batch_sampler=tr_sampler, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers>0)
    va_loader = DataLoader(ds, batch_sampler=va_sampler, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers>0)

    params = list(model.parameters()) + list(head.parameters())
    if reach_head is not None:
        params += list(reach_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(tr_loader))

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    history = cfg.history_size
    n_preds = cfg.num_preds  # model arch n_preds (always 1 for ckpts we'll see)
    lambd = cfg.sigreg_weight
    beta = args.reward_weight
    H_roll = args.rc_h_max if args.rc_aux else None  # rollout horizon for RC-aux

    print(f"joint v2 train: pairs={len(ds)} train={len(tr_idx)} val={len(va_idx)} "
          f"epochs={args.epochs} lr={args.lr} reward_weight={beta} "
          f"horizons={ds.horizons} w_alive={args.w_alive}")
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_sampler.set_epoch(epoch)
        model.train(); head.train()
        if reach_head is not None: reach_head.train()
        running = defaultdict(float); seen = 0; t0 = time.perf_counter()
        for batch in tr_loader:
            pix = batch["pixels"].to(device, non_blocking=True)
            act = batch["action"].to(device, non_blocking=True)
            tgt_reward_norm = batch["target_reward"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                enc = model.encode({"pixels": pix, "action": act})
                emb = enc["emb"]; act_emb = enc["act_emb"]
                if args.rc_aux:
                    # Autoregressive multi-step rollout for H_roll steps.
                    init_emb = emb[:, :history]
                    init_act = act_emb[:, :history]
                    # Need actions at positions history..history+H_roll-2 for the
                    # remaining H_roll-1 rollout steps.
                    future_acts = act_emb[:, history:history + max(0, H_roll - 1)]
                    pred_emb_multi = rollout_open_loop(model, init_emb, init_act,
                                                       future_acts, H_roll)
                    tgt_emb_multi = emb[:, history:history + H_roll]
                    pred_loss, step_losses = multi_horizon_pred_loss(
                        pred_emb_multi, tgt_emb_multi,
                        weighting=args.rc_mh_weighting,
                        weight_power=args.rc_mh_power,
                    )
                    pred_emb_for_head = pred_emb_multi  # for reward head + reach
                    tgt_emb_for_reach = tgt_emb_multi
                else:
                    ctx_emb = emb[:, :history]; ctx_act = act_emb[:, :history]
                    tgt_emb = emb[:, n_preds:]
                    pred_emb_for_head = model.predict(ctx_emb, ctx_act)
                    pred_loss = (pred_emb_for_head - tgt_emb).pow(2).mean()
                    step_losses = None
                    tgt_emb_for_reach = None
                sigreg_loss = sigreg(emb.transpose(0, 1))
                z_for_reward = pred_emb_for_head[:, -1]
                pred_reward = head(z_for_reward)
                reward_loss = ((pred_reward - tgt_reward_norm) ** 2).mean()
                loss = (args.rc_weight_mh if args.rc_aux else 1.0) * pred_loss \
                       + lambd * sigreg_loss + beta * reward_loss
                if reach_head is not None:
                    src_anchor = emb[:, history - 1]
                    reach_loss, reach_stats = compute_reachability_loss(
                        reach_head,
                        src_emb=src_anchor,
                        future_true=tgt_emb_for_reach,
                        future_pred=pred_emb_for_head,
                        pred_weight=args.rc_pred_weight,
                        temporal_neg_weight=args.rc_temporal_neg_weight,
                        stop_grad_pred=True,
                    )
                    loss = loss + args.rc_weight_reach * reach_loss
                else:
                    reach_loss = torch.tensor(0.0, device=device)
                    reach_stats = {}
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            bs = pix.size(0); seen += bs
            running["loss"]        += float(loss) * bs
            running["pred_loss"]   += float(pred_loss) * bs
            running["sigreg_loss"] += float(sigreg_loss) * bs
            running["reward_loss"] += float(reward_loss) * bs
            if reach_head is not None:
                running["reach_loss"] += float(reach_loss) * bs
                for k, v in reach_stats.items():
                    running[k] += float(v) * bs
            if step_losses is not None:
                for k, v in enumerate(step_losses):
                    running[f"pred_h{k+1}"] += float(v) * bs
        train_metrics = {f"train/{k}": v / max(1, seen) for k, v in running.items()}
        model.eval(); head.eval()
        if reach_head is not None: reach_head.eval()
        v_running = defaultdict(float); v_seen = 0
        with torch.no_grad():
            for batch in va_loader:
                pix = batch["pixels"].to(device, non_blocking=True)
                act = batch["action"].to(device, non_blocking=True)
                tgt = batch["target_reward"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                    enc = model.encode({"pixels": pix, "action": act})
                    emb = enc["emb"]; act_emb = enc["act_emb"]
                    if args.rc_aux:
                        init_emb = emb[:, :history]
                        init_act = act_emb[:, :history]
                        future_acts = act_emb[:, history:history + max(0, H_roll - 1)]
                        pred_emb_multi = rollout_open_loop(model, init_emb, init_act,
                                                           future_acts, H_roll)
                        tgt_emb_multi = emb[:, history:history + H_roll]
                        pred_loss, _ = multi_horizon_pred_loss(
                            pred_emb_multi, tgt_emb_multi,
                            weighting=args.rc_mh_weighting,
                            weight_power=args.rc_mh_power,
                        )
                        pred_emb_for_head = pred_emb_multi
                        tgt_emb_for_reach = tgt_emb_multi
                    else:
                        ctx_emb = emb[:, :history]; ctx_act = act_emb[:, :history]
                        tgt_emb = emb[:, n_preds:]
                        pred_emb_for_head = model.predict(ctx_emb, ctx_act)
                        pred_loss = (pred_emb_for_head - tgt_emb).pow(2).mean()
                        tgt_emb_for_reach = None
                    sigreg_loss = sigreg(emb.transpose(0, 1))
                    pred_reward = head(pred_emb_for_head[:, -1])
                    reward_loss = ((pred_reward - tgt) ** 2).mean()
                    loss = (args.rc_weight_mh if args.rc_aux else 1.0) * pred_loss \
                           + lambd * sigreg_loss + beta * reward_loss
                    if reach_head is not None:
                        reach_loss, reach_stats = compute_reachability_loss(
                            reach_head,
                            src_emb=emb[:, history - 1],
                            future_true=tgt_emb_for_reach,
                            future_pred=pred_emb_for_head,
                            pred_weight=args.rc_pred_weight,
                            temporal_neg_weight=args.rc_temporal_neg_weight,
                            stop_grad_pred=True,
                        )
                        loss = loss + args.rc_weight_reach * reach_loss
                    else:
                        reach_loss = torch.tensor(0.0, device=device)
                        reach_stats = {}
                bs = pix.size(0); v_seen += bs
                v_running["loss"]        += float(loss) * bs
                v_running["pred_loss"]   += float(pred_loss) * bs
                v_running["sigreg_loss"] += float(sigreg_loss) * bs
                v_running["reward_loss"] += float(reward_loss) * bs
                if reach_head is not None:
                    v_running["reach_loss"] += float(reach_loss) * bs
                    for k, v in reach_stats.items():
                        v_running[k] += float(v) * bs
        val_metrics = {f"val/{k}": v / max(1, v_seen) for k, v in v_running.items()}
        elapsed = time.perf_counter() - t0
        rec = {"epoch": epoch, **train_metrics, **val_metrics, "elapsed_s": round(elapsed, 1),
               "samples_per_sec": seen / max(1e-6, elapsed), "lr": sched.get_last_lr()[0]}
        print(json.dumps(rec), flush=True)
        if wandb_run: wandb_run.log(rec, step=epoch)
        ckpt_blob = {
            "epoch": epoch, "config": cfg.to_dict(),
            "model_state": model.state_dict(), "head_state": head.state_dict(),
            "reward_mean": ds.mean, "reward_std": ds.std,
            "horizon": args.horizon, "horizons": ds.horizons,
            "weights": {"x": args.w_x, "score": args.w_score, "coins": args.w_coins,
                        "ptype": args.w_ptype, "death": args.w_death, "alive": args.w_alive},
            "milestone": {"x": args.milestone_x, "bonus": args.milestone_bonus},
            "oversample": {"x_lo": args.oversample_x_lo, "x_hi": args.oversample_x_hi,
                            "mult": args.oversample_mult,
                            "require_progress": args.oversample_require_progress},
            "action_library": ck["action_library"],
            "val_metrics": val_metrics, "train_metrics": train_metrics,
        }
        if reach_head is not None:
            ckpt_blob["reach_head_state"] = reach_head.state_dict()
            ckpt_blob["rc_aux"] = {
                "h_max": args.rc_h_max, "head_hidden": args.rc_head_hidden,
                "weight_mh": args.rc_weight_mh, "weight_reach": args.rc_weight_reach,
                "pred_weight": args.rc_pred_weight,
                "temporal_neg_weight": args.rc_temporal_neg_weight,
                "mh_weighting": args.rc_mh_weighting, "mh_power": args.rc_mh_power,
            }
        torch.save(ckpt_blob, args.out_dir / "latest.pt")
        if val_metrics["val/loss"] < best_val:
            best_val = val_metrics["val/loss"]
            best_blob = {k: v for k, v in ckpt_blob.items() if k not in ("val_metrics","train_metrics")}
            torch.save(best_blob, args.out_dir / "best.pt")
    print(f"saved best.pt val_loss={best_val:.4f}")
    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    main()
