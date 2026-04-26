"""Reward head v2: multi-horizon target, bigger MLP, tunable death penalty.

Improvements over v1:
- Target = sum of Δx_pos over the next K blocks (default K=4) instead of just 1.
  Lower-variance signal; planner objective Σ r̂(ẑ_t) is closer to ground truth.
- Bigger reward MLP: 192 → 128 → 128 → 1 (was 192 → 64 → 1).
- Death penalty configurable (default -10, was -50).
- Re-uses the latent cache produced by v1.
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
                    death_penalty: float):
    """Per-window target: sum of Δx_pos over next k_horizon blocks plus death_penalty
    if Mario dies in any of those blocks."""
    raw_root = Path(raw_root)
    x_cache, life_cache = {}, {}
    def get(name):
        if name not in x_cache:
            with np.load(raw_root / f"{name}.npz", allow_pickle=False) as d:
                x_cache[name] = np.asarray(d["x_pos"])
                life_cache[name] = np.asarray(d["lives"]) if "lives" in d.files else np.zeros_like(x_cache[name])
        return x_cache[name], life_cache[name]

    N = len(mario_ds.index)
    targets = np.zeros(N, dtype=np.float32)
    for w_idx, (ep_id, start) in enumerate(mario_ds.index):
        ep = mario_ds.episodes[ep_id]
        x, lives = get(ep.name)
        first_block = start + history - 1  # block we want to predict from
        total_dx = 0.0
        died = False
        for k in range(k_horizon):
            blk = first_block + k
            i_pre = blk * BLOCK
            i_post = min((blk + 1) * BLOCK, len(x) - 1)
            if i_pre >= len(x): break
            dx = float(x[i_post]) - float(x[i_pre])
            total_dx += dx
            if i_post < len(lives) and i_pre < len(lives) and int(lives[i_post]) < int(lives[i_pre]):
                died = True; break
        if died: total_dx += death_penalty
        targets[w_idx] = total_dx
    return torch.from_numpy(targets)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/ckpt/best.pt")
    ap.add_argument("--dataset-root", default="/root/data/tas_precomputed")
    ap.add_argument("--raw-root", default="/root/data/tas_full")
    ap.add_argument("--latent-cache", default="/root/runs/reward_head/lat_targets.pt", type=Path)
    ap.add_argument("--out", default="/root/runs/reward_head_v2/reward_head_v2.pt", type=Path)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--death-penalty", type=float, default=-10.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--wandb-project", type=str, default="lewm-mario")
    ap.add_argument("--wandb-run-name", type=str, default="reward-head-v2")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])

    if not args.latent_cache.exists():
        raise FileNotFoundError(f"latent cache missing: {args.latent_cache} — run v1 first")
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    latents = cache["latents"]
    print(f"loaded latent cache: {latents.shape}")

    # rebuild dataset to re-derive targets at the new horizon / penalty
    episodes = discover_episodes(args.dataset_root)
    ds = MarioTraceDataset(episodes, cfg.history_size, cfg.num_preds, cfg.image_size,
                           stride=1, npz_load_mode="lazy", max_cached_episodes=4)
    targets = compute_targets(ds, args.raw_root, cfg.history_size, args.horizon, args.death_penalty)
    print(f"targets: mean={float(targets.mean()):.2f}  std={float(targets.std()):.2f}  "
          f"min={float(targets.min()):.1f}  max={float(targets.max()):.1f}  "
          f"horizon={args.horizon}  death_penalty={args.death_penalty}")

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

    print(f"training reward head v2: train={len(tr_lat)} val={len(va_lat)} "
          f"hidden={args.hidden} batch={args.batch_size}")
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
            va_mae_px = float(((va_pred * sd + mu) - va_y).abs().mean())
            tr_pred = head(tr_lat)
            tr_mae_px = float(((tr_pred * sd + mu) - tr_y).abs().mean())
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_loss": va_loss, "val_mae_px": va_mae_px, "train_mae_px": tr_mae_px,
               "elapsed_s": round(time.perf_counter() - t0, 2)}
        print(json.dumps(rec), flush=True)
        if wandb_run: wandb_run.log(rec, step=epoch)
        if va_loss < best_val:
            best_val = va_loss; best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    torch.save({
        "head_state": best_state if best_state else head.state_dict(),
        "config": {"in_dim": cfg.action_embed_dim, "hidden": args.hidden,
                   "reward_mean": mu, "reward_std": sd,
                   "death_penalty": args.death_penalty, "horizon": args.horizon},
        "base_ckpt": args.ckpt, "base_epoch": ck["epoch"],
    }, args.out)
    print(f"saved {args.out}  best_val={best_val:.4f}")
    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    main()
