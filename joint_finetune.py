"""Joint encoder+predictor+reward head finetune.

Resumes from best.pt and trains end-to-end with three loss terms:
  L = L_pred + λ * L_sigreg + β * L_reward

The reward head reads from `predicted next-state` latents (model.predict output)
so reward learning shapes the predictor too. No stop-grad — the encoder learns
to encode reward-relevant features.

Same dataset as Phase 1 (precomputed 224x224 frames + raw x_pos for reward
targets). Single-horizon reward target with low death penalty so we don't
under-prefer risk.
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

sys.path.insert(0, "/root/lewm_mario")
from mario_lewm.model import LeWorldModel, LeWorldModelConfig, SIGReg
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

class WindowDS(torch.utils.data.Dataset):
    """Wraps MarioTraceDataset to also yield the per-window reward target."""
    def __init__(self, base: MarioTraceDataset, raw_root: Path, history: int,
                 horizon: int, death_penalty: float):
        self.base = base; self.history = history; self.horizon = horizon
        x_cache, life_cache = {}, {}
        for ep in self.base.episodes:
            with np.load(Path(raw_root) / f"{ep.name}.npz", allow_pickle=False) as d:
                x_cache[ep.name] = np.asarray(d["x_pos"])
                life_cache[ep.name] = np.asarray(d["lives"]) if "lives" in d.files else np.zeros_like(x_cache[ep.name])
        targets = np.zeros(len(self.base.index), dtype=np.float32)
        for w, (ep_id, start) in enumerate(self.base.index):
            name = self.base.episodes[ep_id].name
            x, lives = x_cache[name], life_cache[name]
            first_block = start + history - 1
            total_dx, died = 0.0, False
            for k in range(horizon):
                blk = first_block + k
                i_pre = blk * BLOCK; i_post = min((blk + 1) * BLOCK, len(x) - 1)
                if i_pre >= len(x): break
                total_dx += float(x[i_post]) - float(x[i_pre])
                if i_post < len(lives) and i_pre < len(lives) and int(lives[i_post]) < int(lives[i_pre]):
                    died = True; break
            if died: total_dx += death_penalty
            targets[w] = total_dx
        self.targets = targets
        self.mean = float(targets.mean()); self.std = float(targets.std() + 1e-6)
        print(f"reward target: mean={self.mean:.2f}  std={self.std:.2f}  "
              f"horizon={horizon}  death_penalty={death_penalty}")

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
    ap.add_argument("--ckpt", default="/root/ckpt/best.pt")
    ap.add_argument("--dataset-root", default="/root/data/tas_precomputed")
    ap.add_argument("--raw-root", default="/root/data/tas_full")
    ap.add_argument("--out-dir", default="/root/runs/joint_v1", type=Path)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--death-penalty", type=float, default=-10.0)
    ap.add_argument("--reward-weight", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--wandb-project", type=str, default="lewm-mario")
    ap.add_argument("--wandb-run-name", type=str, default="joint-v1")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])
    model = LeWorldModel(cfg); model.load_state_dict(ck["model_state"])
    model.to(device)
    head = RewardHead(in_dim=cfg.action_embed_dim).to(device)
    sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj).to(device)

    episodes = discover_episodes(args.dataset_root)
    base = MarioTraceDataset(episodes, cfg.history_size, cfg.num_preds, cfg.image_size,
                              stride=1, npz_load_mode="lazy", max_cached_episodes=8)
    ds = WindowDS(base, args.raw_root, cfg.history_size, args.horizon, args.death_penalty)
    g = torch.Generator().manual_seed(3072)
    cut = int(len(ds) * 0.9)
    perm = torch.randperm(len(ds), generator=g)
    tr_idx = perm[:cut].tolist(); va_idx = perm[cut:].tolist()
    tr_sampler = EpisodeBatchSampler(tr_idx, base.index, args.batch_size, True, True, 3072)
    va_sampler = EpisodeBatchSampler(va_idx, base.index, args.batch_size, False, False, 3072)
    tr_loader = DataLoader(ds, batch_sampler=tr_sampler, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers>0)
    va_loader = DataLoader(ds, batch_sampler=va_sampler, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers>0)

    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(tr_loader))

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    history = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.sigreg_weight
    beta = args.reward_weight

    print(f"joint train: pairs={len(ds)} train={len(tr_idx)} val={len(va_idx)} "
          f"epochs={args.epochs} lr={args.lr} reward_weight={beta}")
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_sampler.set_epoch(epoch)
        model.train(); head.train()
        running = defaultdict(float); seen = 0; t0 = time.perf_counter()
        for batch in tr_loader:
            pix = batch["pixels"].to(device, non_blocking=True)
            act = batch["action"].to(device, non_blocking=True)
            tgt_reward_norm = batch["target_reward"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                enc = model.encode({"pixels": pix, "action": act})
                emb = enc["emb"]; act_emb = enc["act_emb"]
                ctx_emb = emb[:, :history]; ctx_act = act_emb[:, :history]
                tgt_emb = emb[:, n_preds:]
                pred_emb = model.predict(ctx_emb, ctx_act)
                pred_loss = (pred_emb - tgt_emb).pow(2).mean()
                sigreg_loss = sigreg(emb.transpose(0, 1))
                # reward head reads from the predicted-next latent for the LAST history block
                z_for_reward = pred_emb[:, -1]
                pred_reward = head(z_for_reward)
                reward_loss = ((pred_reward - tgt_reward_norm) ** 2).mean()
                loss = pred_loss + lambd * sigreg_loss + beta * reward_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            bs = pix.size(0); seen += bs
            running["loss"] += float(loss) * bs
            running["pred_loss"] += float(pred_loss) * bs
            running["sigreg_loss"] += float(sigreg_loss) * bs
            running["reward_loss"] += float(reward_loss) * bs
        train_metrics = {f"train/{k}": v / max(1, seen) for k, v in running.items()}
        # val
        model.eval(); head.eval()
        v_running = defaultdict(float); v_seen = 0
        with torch.no_grad():
            for batch in va_loader:
                pix = batch["pixels"].to(device, non_blocking=True)
                act = batch["action"].to(device, non_blocking=True)
                tgt = batch["target_reward"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                    enc = model.encode({"pixels": pix, "action": act})
                    emb = enc["emb"]; act_emb = enc["act_emb"]
                    ctx_emb = emb[:, :history]; ctx_act = act_emb[:, :history]
                    tgt_emb = emb[:, n_preds:]
                    pred_emb = model.predict(ctx_emb, ctx_act)
                    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
                    sigreg_loss = sigreg(emb.transpose(0, 1))
                    pred_reward = head(pred_emb[:, -1])
                    reward_loss = ((pred_reward - tgt) ** 2).mean()
                    loss = pred_loss + lambd * sigreg_loss + beta * reward_loss
                bs = pix.size(0); v_seen += bs
                v_running["loss"] += float(loss) * bs
                v_running["pred_loss"] += float(pred_loss) * bs
                v_running["sigreg_loss"] += float(sigreg_loss) * bs
                v_running["reward_loss"] += float(reward_loss) * bs
        val_metrics = {f"val/{k}": v / max(1, v_seen) for k, v in v_running.items()}
        elapsed = time.perf_counter() - t0
        rec = {"epoch": epoch, **train_metrics, **val_metrics, "elapsed_s": round(elapsed, 1),
               "samples_per_sec": seen / max(1e-6, elapsed), "lr": sched.get_last_lr()[0]}
        print(json.dumps(rec), flush=True)
        if wandb_run: wandb_run.log(rec, step=epoch)
        torch.save({
            "epoch": epoch, "config": cfg.to_dict(),
            "model_state": model.state_dict(), "head_state": head.state_dict(),
            "reward_mean": ds.mean, "reward_std": ds.std,
            "horizon": args.horizon, "death_penalty": args.death_penalty,
            "action_library": ck["action_library"],
            "val_metrics": val_metrics, "train_metrics": train_metrics,
        }, args.out_dir / "latest.pt")
        if val_metrics["val/loss"] < best_val:
            best_val = val_metrics["val/loss"]
            torch.save({
                "epoch": epoch, "config": cfg.to_dict(),
                "model_state": model.state_dict(), "head_state": head.state_dict(),
                "reward_mean": ds.mean, "reward_std": ds.std,
                "horizon": args.horizon, "death_penalty": args.death_penalty,
                "action_library": ck["action_library"],
            }, args.out_dir / "best.pt")
    print(f"saved best.pt val_loss={best_val:.4f}")
    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    main()
