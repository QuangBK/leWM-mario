"""Train a small reward head on top of a frozen LeWM Mario encoder.

Path B step 1: r̂(z) predicts per-block Δx_pos (with a death penalty added when
Mario loses a life within the block).

Pipeline:
  1. Pre-encode every blocked frame in the dataset into a 192-d latent tensor
     (one pass through the frozen ViT). Cache to disk.
  2. Compute per-window reward target from raw tas_full x_pos arrays.
  3. Train the reward MLP on (latent, target) pairs in pure-GPU mode.

This avoids the CPU-bound dataloader bottleneck of feeding raw 224x224 frames
through the encoder every epoch.
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/root/lewm_mario")
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.dataset import discover_episodes, MarioTraceDataset

DEATH_PENALTY = -50.0
BLOCK = 5

class RewardHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z): return self.net(z).squeeze(-1)

@torch.no_grad()
def precompute_latents_and_targets(model, mario_ds, raw_root: Path, history: int,
                                    image_size: int, device, batch_size: int = 64):
    """Returns (latents (N, D), targets (N,)) where N is the number of training windows.
    Encodes the (history-1)-th frame of each window — the latent that we want
    the reward head to predict the next-block reward from."""
    raw_root = Path(raw_root)
    x_cache: dict[str, np.ndarray] = {}
    life_cache: dict[str, np.ndarray] = {}

    def get_xpos(name):
        if name not in x_cache:
            with np.load(raw_root / f"{name}.npz", allow_pickle=False) as d:
                x_cache[name] = np.asarray(d["x_pos"])
                life_cache[name] = np.asarray(d["lives"]) if "lives" in d.files else np.zeros_like(x_cache[name])
        return x_cache[name], life_cache[name]

    N = len(mario_ds.index)
    latents_out = torch.empty(N, model.config.action_embed_dim, dtype=torch.float32)
    targets_out = torch.empty(N, dtype=torch.float32)

    # Process in episode order to maximize npz cache hit rate
    by_ep: dict[int, list[int]] = {}
    for win_idx, (ep_id, start) in enumerate(mario_ds.index):
        by_ep.setdefault(ep_id, []).append(win_idx)

    print(f"precomputing latents+targets for {N} windows across {len(by_ep)} episodes...", flush=True)
    t0 = time.perf_counter()
    done = 0
    for ep_id, win_indices in by_ep.items():
        episode = mario_ds.episodes[ep_id]
        x, lives = get_xpos(episode.name)
        # load all frames for this episode once
        if episode.frames_npz is None: continue
        all_frames = mario_ds._get_npz_frames(ep_id)  # (T, H, W, 3) uint8
        # encode all the (history-1)-th frames in batches
        # for window starting at `start`, we need frame at index (start + history - 1)
        starts = np.array([mario_ds.index[w][1] for w in win_indices])
        wanted = starts + history - 1  # (W,)
        # fetch unique frames once, encode, then map back
        uniq = np.unique(wanted)
        # encode in chunks
        for chunk_start in range(0, len(uniq), batch_size):
            chunk = uniq[chunk_start: chunk_start + batch_size]
            frames_chunk = all_frames[chunk]  # (B, H, W, 3) uint8
            # mimic MarioTraceDataset normalization
            t = torch.from_numpy(frames_chunk).permute(0, 3, 1, 2).float() / 255.0
            if t.shape[-1] != image_size:
                t = torch.nn.functional.interpolate(t, size=(image_size, image_size),
                                                     mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            t = ((t - mean) / std).to(device, non_blocking=True)
            t = t.unsqueeze(1)  # (B, 1, 3, H, W)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                emb = model.encode({"pixels": t})["emb"][:, 0].float()  # (B, D)
            uniq_to_emb = {int(idx): emb[i].cpu() for i, idx in enumerate(chunk)}
            # write back to the right window slots
            for w in win_indices:
                start_w = mario_ds.index[w][1]
                want = start_w + history - 1
                if want in uniq_to_emb:
                    latents_out[w] = uniq_to_emb[want]
                    # reward target
                    pred_block = start_w + history - 1
                    i_pre = pred_block * BLOCK
                    i_post = min((pred_block + 1) * BLOCK, len(x) - 1)
                    if i_pre >= len(x):
                        targets_out[w] = 0.0
                    else:
                        dx = float(x[i_post]) - float(x[i_pre])
                        if i_post < len(lives) and i_pre < len(lives) and int(lives[i_post]) < int(lives[i_pre]):
                            dx += DEATH_PENALTY
                        targets_out[w] = dx
        done += len(win_indices)
        if done % 5000 < len(win_indices):
            print(f"  done {done}/{N} windows  elapsed_s={time.perf_counter()-t0:.1f}", flush=True)
    return latents_out, targets_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/ckpt/best.pt")
    ap.add_argument("--dataset-root", default="/root/data/tas_precomputed")
    ap.add_argument("--raw-root", default="/root/data/tas_full")
    ap.add_argument("--out", default="/root/runs/reward_head/reward_head.pt", type=Path)
    ap.add_argument("--cache", default="/root/runs/reward_head/lat_targets.pt", type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])
    base = LeWorldModel(cfg); base.load_state_dict(ck["model_state"])
    base.to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)

    if args.cache.exists():
        print(f"loading cached latents from {args.cache}")
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
        latents = cache["latents"]; targets = cache["targets"]
    else:
        episodes = discover_episodes(args.dataset_root)
        ds = MarioTraceDataset(episodes, cfg.history_size, cfg.num_preds, cfg.image_size,
                               stride=1, npz_load_mode="lazy", max_cached_episodes=4)
        latents, targets = precompute_latents_and_targets(
            base, ds, args.raw_root, cfg.history_size, cfg.image_size, device)
        torch.save({"latents": latents, "targets": targets}, args.cache)
        print(f"cached to {args.cache}  shape={latents.shape}")

    mu, sd = float(targets.mean()), float(targets.std() + 1e-6)
    print(f"reward target stats: mean={mu:.3f} std={sd:.3f} min={targets.min():.1f} max={targets.max():.1f}")

    # 90/10 split
    g = torch.Generator().manual_seed(3072)
    perm = torch.randperm(len(latents), generator=g)
    cut = int(len(perm) * 0.9)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    tr_lat, tr_y = latents[tr_idx].to(device), targets[tr_idx].to(device)
    va_lat, va_y = latents[va_idx].to(device), targets[va_idx].to(device)
    tr_y_norm = (tr_y - mu) / sd
    va_y_norm = (va_y - mu) / sd

    head = RewardHead(in_dim=cfg.action_embed_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, (len(tr_lat) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                                config=vars(args))

    print(f"training reward head: train={len(tr_lat)} val={len(va_lat)} batch={args.batch_size}")
    for epoch in range(1, args.epochs + 1):
        head.train()
        t0 = time.perf_counter()
        perm_e = torch.randperm(len(tr_lat), device=device)
        losses = []
        for i in range(0, len(tr_lat), args.batch_size):
            idx = perm_e[i: i + args.batch_size]
            x = tr_lat[idx]; y = tr_y_norm[idx]
            pred = head(x)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step()
            losses.append(float(loss))
        head.eval()
        with torch.no_grad():
            va_pred = head(va_lat)
            va_loss = ((va_pred - va_y_norm) ** 2).mean().item()
            va_mae_px = float(((va_pred * sd + mu) - va_y).abs().mean())
            tr_pred = head(tr_lat)
            tr_mae_px = float(((tr_pred * sd + mu) - tr_y).abs().mean())
        elapsed = time.perf_counter() - t0
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_loss": va_loss, "val_mae_px": va_mae_px, "train_mae_px": tr_mae_px,
               "elapsed_s": round(elapsed, 2)}
        print(json.dumps(rec), flush=True)
        if wandb_run is not None:
            wandb_run.log(rec, step=epoch)

    torch.save({
        "head_state": head.state_dict(),
        "config": {"in_dim": cfg.action_embed_dim, "hidden": 64,
                   "reward_mean": mu, "reward_std": sd, "death_penalty": DEATH_PENALTY},
        "base_ckpt": args.ckpt, "base_epoch": ck["epoch"],
    }, args.out)
    print(f"saved {args.out}")
    if wandb_run is not None: wandb_run.finish()

if __name__ == "__main__":
    main()
