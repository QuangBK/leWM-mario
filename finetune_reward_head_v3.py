"""Reward head v3: composite reward = α·Δx + β·Δscore + γ·Δcoins + λ·Δptype + δ·death

Builds on v2 (multi-horizon, MLP 192→128→128→1) but uses RAM-derived score / coins
/ player-type captured by the upgraded `tas_replay.py`. The intent is to give the
planner positive credit for kills, coin pickups, and power-ups — the things that
cluster around the x=594 plateau (staircase + pipe area).

Weights are CLI flags so we can sweep:
  --w-x      (default 1.0)        — pixels per block
  --w-score  (default 0.05)       — game score Δ; kill=100/stomp, mushroom=1000
  --w-coins  (default 5.0)        — Δ coin counter
  --w-ptype  (default 20.0)       — Δ player_type (small→big = +1, big→fire = +1)
  --w-death  (default -10.0)      — terminal death penalty
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/root/lewm_mario")
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.dataset import discover_episodes, MarioTraceDataset

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

def compute_targets(mario_ds, raw_root: Path, history: int, k_horizon: int,
                    w_x: float, w_score: float, w_coins: float, w_ptype: float,
                    w_death: float):
    """Per-window composite reward target summed over next k_horizon blocks.
    Reward(t→t+1) = w_x·Δx + w_score·Δscore + w_coins·Δcoins + w_ptype·Δptype.
    If Mario dies inside the window: add w_death and stop accumulating early."""
    raw_root = Path(raw_root)
    cache = {}
    def get(name):
        if name not in cache:
            with np.load(raw_root / f"{name}.npz", allow_pickle=False) as d:
                fields = d.files
                cache[name] = {
                    "x":     np.asarray(d["x_pos"]),
                    "lives": np.asarray(d["lives"]) if "lives" in fields else np.zeros_like(d["x_pos"]),
                    "score": np.asarray(d["score"]) if "score" in fields else np.zeros_like(d["x_pos"]),
                    "coins": np.asarray(d["coins"]) if "coins" in fields else np.zeros_like(d["x_pos"]),
                    "ptype": np.asarray(d["ptype"]) if "ptype" in fields else np.zeros_like(d["x_pos"]),
                }
        return cache[name]

    N = len(mario_ds.index)
    targets = np.zeros(N, dtype=np.float32)
    components = {"x": np.zeros(N), "score": np.zeros(N), "coins": np.zeros(N), "ptype": np.zeros(N), "death": np.zeros(N)}
    for w_idx, (ep_id, start) in enumerate(mario_ds.index):
        ep = mario_ds.episodes[ep_id]
        ar = get(ep.name)
        first_block = start + history - 1
        total = 0.0
        died = False
        for k in range(k_horizon):
            blk = first_block + k
            i_pre  = blk * BLOCK
            i_post = min((blk + 1) * BLOCK, len(ar["x"]) - 1)
            if i_pre >= len(ar["x"]): break
            dx     = float(ar["x"][i_post])     - float(ar["x"][i_pre])
            dscore = float(ar["score"][i_post]) - float(ar["score"][i_pre])
            dcoins = float(ar["coins"][i_post]) - float(ar["coins"][i_pre])
            dptype = float(ar["ptype"][i_post]) - float(ar["ptype"][i_pre])
            r = w_x * dx + w_score * dscore + w_coins * dcoins + w_ptype * dptype
            total += r
            components["x"][w_idx]     += w_x * dx
            components["score"][w_idx] += w_score * dscore
            components["coins"][w_idx] += w_coins * dcoins
            components["ptype"][w_idx] += w_ptype * dptype
            if i_post < len(ar["lives"]) and i_pre < len(ar["lives"]) \
               and int(ar["lives"][i_post]) < int(ar["lives"][i_pre]):
                died = True; break
        if died:
            total += w_death
            components["death"][w_idx] = w_death
        targets[w_idx] = total

    print("=== reward decomposition (mean per window) ===")
    for k, v in components.items():
        print(f"  {k:6s}: mean={v.mean():+.3f}  std={v.std():.3f}  "
              f"min={v.min():+.1f}  max={v.max():+.1f}")
    print(f"  TOTAL : mean={targets.mean():+.3f}  std={targets.std():.3f}  "
          f"min={targets.min():+.1f}  max={targets.max():+.1f}")
    return torch.from_numpy(targets)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/ckpt/best.pt")
    ap.add_argument("--dataset-root", default="/root/data/tas_precomputed")
    ap.add_argument("--raw-root", default="/root/data/tas_full")
    ap.add_argument("--latent-cache", default="/root/runs/reward_head/lat_targets.pt", type=Path)
    ap.add_argument("--out", default="/root/runs/reward_head_v3/reward_head_v3.pt", type=Path)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--w-x", type=float, default=1.0)
    ap.add_argument("--w-score", type=float, default=0.05)
    ap.add_argument("--w-coins", type=float, default=5.0)
    ap.add_argument("--w-ptype", type=float, default=20.0)
    ap.add_argument("--w-death", type=float, default=-10.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--wandb-project", type=str, default="lewm-mario")
    ap.add_argument("--wandb-run-name", type=str, default="reward-head-v3")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])

    if not args.latent_cache.exists():
        raise FileNotFoundError(f"latent cache missing: {args.latent_cache} — run v1 first to generate it")
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    latents = cache["latents"]
    print(f"loaded latent cache: {latents.shape}")

    episodes = discover_episodes(args.dataset_root)
    ds = MarioTraceDataset(episodes, cfg.history_size, cfg.num_preds, cfg.image_size,
                           stride=1, npz_load_mode="lazy", max_cached_episodes=4)
    targets = compute_targets(ds, args.raw_root, cfg.history_size, args.horizon,
                              args.w_x, args.w_score, args.w_coins, args.w_ptype, args.w_death)

    g = torch.Generator().manual_seed(3072)
    perm = torch.randperm(len(latents), generator=g)
    cut = int(len(perm) * 0.9)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    tr_lat, tr_y = latents[tr_idx].to(device), targets[tr_idx].to(device)
    va_lat, va_y = latents[va_idx].to(device), targets[va_idx].to(device)
    mu, sd = float(tr_y.mean()), float(tr_y.std() + 1e-6)
    tr_y_norm = (tr_y - mu) / sd
    va_y_norm = (va_y - mu) / sd

    head = RewardHead(in_dim=cfg.action_embed_dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, (len(tr_lat) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                                config=vars(args))

    print(f"training reward head v3: train={len(tr_lat)} val={len(va_lat)} "
          f"hidden={args.hidden} weights=x{args.w_x},s{args.w_score},c{args.w_coins},"
          f"p{args.w_ptype},d{args.w_death}")
    best_val = float("inf"); best_state = None
    for epoch in range(1, args.epochs + 1):
        head.train()
        t0 = time.perf_counter()
        permE = torch.randperm(len(tr_lat), device=device)
        losses = []
        for i in range(0, len(tr_lat), args.batch_size):
            idx = permE[i: i + args.batch_size]
            pred = head(tr_lat[idx])
            loss = ((pred - tr_y_norm[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(float(loss))
        head.eval()
        with torch.no_grad():
            va_pred = head(va_lat)
            va_loss = ((va_pred - va_y_norm) ** 2).mean().item()
            va_mae = float(((va_pred * sd + mu) - va_y).abs().mean())
            tr_pred = head(tr_lat)
            tr_mae = float(((tr_pred * sd + mu) - tr_y).abs().mean())
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_loss": va_loss, "val_mae": va_mae, "train_mae": tr_mae,
               "elapsed_s": round(time.perf_counter() - t0, 2)}
        print(json.dumps(rec), flush=True)
        if wandb_run: wandb_run.log(rec, step=epoch)
        if va_loss < best_val:
            best_val = va_loss; best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    torch.save({
        "head_state": best_state if best_state else head.state_dict(),
        "config": {"in_dim": cfg.action_embed_dim, "hidden": args.hidden,
                   "reward_mean": mu, "reward_std": sd,
                   "w_x": args.w_x, "w_score": args.w_score, "w_coins": args.w_coins,
                   "w_ptype": args.w_ptype, "w_death": args.w_death,
                   "horizon": args.horizon},
        "base_ckpt": args.ckpt, "base_epoch": ck["epoch"],
    }, args.out)
    print(f"saved {args.out}  best_val={best_val:.4f}")
    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    main()
