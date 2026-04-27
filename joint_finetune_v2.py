"""Joint encoder+predictor+reward head finetune with composite (score+coin+ptype+x) reward.

Same shape as joint_finetune.py but the reward target is a weighted sum over
multiple RAM-derived signals, captured by the upgraded tas_replay.py:
  reward(t→t+H) = w_x·Δx + w_score·Δscore + w_coins·Δcoins + w_ptype·Δptype
                 + w_death·died

Use --w-* flags to sweep. Defaults match the reward_head_v3 defaults so a v2
joint train should be a drop-in replacement for joint_v1 with richer signal.
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
                 horizon: int, w_x: float, w_score: float, w_coins: float,
                 w_ptype: float, w_death: float):
        self.base = base; self.history = history; self.horizon = horizon
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
        targets = np.zeros(len(self.base.index), dtype=np.float32)
        comps = {k: np.zeros(len(self.base.index)) for k in ("x","score","coins","ptype","death")}
        for w, (ep_id, start) in enumerate(self.base.index):
            ar = cache[self.base.episodes[ep_id].name]
            first_block = start + history - 1
            total, died = 0.0, False
            for k in range(horizon):
                blk = first_block + k
                i_pre  = blk * BLOCK
                i_post = min((blk + 1) * BLOCK, len(ar["x"]) - 1)
                if i_pre >= len(ar["x"]): break
                dx     = float(ar["x"][i_post])     - float(ar["x"][i_pre])
                dscore = float(ar["score"][i_post]) - float(ar["score"][i_pre])
                dcoins = float(ar["coins"][i_post]) - float(ar["coins"][i_pre])
                dptype = float(ar["ptype"][i_post]) - float(ar["ptype"][i_pre])
                r = w_x*dx + w_score*dscore + w_coins*dcoins + w_ptype*dptype
                total += r
                comps["x"][w]     += w_x*dx
                comps["score"][w] += w_score*dscore
                comps["coins"][w] += w_coins*dcoins
                comps["ptype"][w] += w_ptype*dptype
                if i_post < len(ar["lives"]) and i_pre < len(ar["lives"]) \
                   and int(ar["lives"][i_post]) < int(ar["lives"][i_pre]):
                    died = True; break
            if died:
                total += w_death; comps["death"][w] = w_death
            targets[w] = total
        self.targets = targets
        self.mean = float(targets.mean()); self.std = float(targets.std() + 1e-6)
        print(f"reward target: mean={self.mean:.2f}  std={self.std:.2f}  horizon={horizon}")
        for k, v in comps.items():
            print(f"  comp[{k:6s}]: mean={v.mean():+.2f}  std={v.std():.2f}  "
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
    ap.add_argument("--w-x", type=float, default=1.0)
    ap.add_argument("--w-score", type=float, default=0.05)
    ap.add_argument("--w-coins", type=float, default=5.0)
    ap.add_argument("--w-ptype", type=float, default=20.0)
    ap.add_argument("--w-death", type=float, default=-10.0)
    ap.add_argument("--reward-weight", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--wandb-project", type=str, default="lewm-mario")
    ap.add_argument("--wandb-run-name", type=str, default="joint-v2-score")
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
    ds = WindowDS(base, args.raw_root, cfg.history_size, args.horizon,
                  args.w_x, args.w_score, args.w_coins, args.w_ptype, args.w_death)
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

    print(f"joint v2 train: pairs={len(ds)} train={len(tr_idx)} val={len(va_idx)} "
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
                z_for_reward = pred_emb[:, -1]
                pred_reward = head(z_for_reward)
                reward_loss = ((pred_reward - tgt_reward_norm) ** 2).mean()
                loss = pred_loss + lambd * sigreg_loss + beta * reward_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            bs = pix.size(0); seen += bs
            running["loss"]        += float(loss) * bs
            running["pred_loss"]   += float(pred_loss) * bs
            running["sigreg_loss"] += float(sigreg_loss) * bs
            running["reward_loss"] += float(reward_loss) * bs
        train_metrics = {f"train/{k}": v / max(1, seen) for k, v in running.items()}
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
                v_running["loss"]        += float(loss) * bs
                v_running["pred_loss"]   += float(pred_loss) * bs
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
            "horizon": args.horizon,
            "weights": {"x": args.w_x, "score": args.w_score, "coins": args.w_coins,
                        "ptype": args.w_ptype, "death": args.w_death},
            "action_library": ck["action_library"],
            "val_metrics": val_metrics, "train_metrics": train_metrics,
        }, args.out_dir / "latest.pt")
        if val_metrics["val/loss"] < best_val:
            best_val = val_metrics["val/loss"]
            torch.save({
                "epoch": epoch, "config": cfg.to_dict(),
                "model_state": model.state_dict(), "head_state": head.state_dict(),
                "reward_mean": ds.mean, "reward_std": ds.std,
                "horizon": args.horizon,
                "weights": {"x": args.w_x, "score": args.w_score, "coins": args.w_coins,
                            "ptype": args.w_ptype, "death": args.w_death},
                "action_library": ck["action_library"],
            }, args.out_dir / "best.pt")
    print(f"saved best.pt val_loss={best_val:.4f}")
    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    main()
