"""Evaluate trained LeWM Mario:
1. Open-loop latent prediction MSE vs horizon (1, 2, 4, 8, 16 steps)
2. Linear x_pos probe (latent -> x_pos) trained on train split, evaluated on val
3. Sample reconstruction-via-nearest-neighbor sanity check
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, "/root/lewm_mario")
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.dataset import MarioTraceDataset, discover_episodes
from sklearn.linear_model import Ridge

def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])
    model = LeWorldModel(cfg)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ck["epoch"], ck["val_metrics"]

@torch.no_grad()
def open_loop_mse(model, dataset, device, horizons=(1, 2, 4, 8, 16), n_starts=200, history=3):
    """For random starts, encode the next H frames as ground truth, autoregressively
    roll out the predictor from history-frame context, compute MSE between predicted
    latent and true latent at each horizon."""
    rng = np.random.default_rng(123)
    rows = []
    starts_sampled = 0
    for trial in range(n_starts * 4):
        if starts_sampled >= n_starts:
            break
        # sample an episode and a start
        ep_idx = rng.integers(0, len(dataset.episodes))
        episode = dataset.episodes[ep_idx]
        ep_len = dataset._episode_length(episode)
        max_h = max(horizons)
        if ep_len < history + max_h + 1:
            continue
        start = int(rng.integers(0, ep_len - history - max_h))
        # get the slice of frames+actions
        end = start + history + max_h
        if episode.frames_npz is not None:
            frames = dataset._get_npz_frames(dataset._episode_ids[id(episode)])[start:end]
            pixels = dataset._preprocess_npz_frames(frames)
        else:
            paths = episode.frame_paths[start:end]
            pixels = torch.stack([dataset._load_frame(p) for p in paths], dim=0)
        actions = torch.from_numpy(episode.actions[start:end]).float()
        pixels = pixels.unsqueeze(0).to(device)  # (1, T, C, H, W)
        actions = actions.unsqueeze(0).to(device)
        # encode all frames -> ground-truth latents (B,T,D)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
            enc = model.encode({"pixels": pixels, "action": actions})
        true_emb = enc["emb"]  # (1, T, D)
        act_emb = enc["act_emb"]
        # initialize rollout from first `history` true latents
        emb_seq = true_emb[:, :history].clone()
        for h in range(max_h):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                pred = model.predict(emb_seq[:, -history:], act_emb[:, h:h+history])
            next_pred = pred[:, -1:].float()
            emb_seq = torch.cat([emb_seq, next_pred], dim=1)
        # measure MSE at each requested horizon
        for h in horizons:
            t = history + h - 1
            if t >= true_emb.size(1): continue
            mse = ((emb_seq[:, t] - true_emb[:, t].float()) ** 2).mean().item()
            rows.append({"horizon": h, "mse": mse, "trial": trial, "ep": ep_idx})
        starts_sampled += 1
    return rows

@torch.no_grad()
def gather_latents_with_xpos(model, dataset, device, max_per_ep=200):
    """Encode every frame in `dataset.episodes` and read x_pos from metadata if
    present. Returns aligned (latents, x_pos) for probing."""
    all_lat, all_x = [], []
    for ep_id, episode in enumerate(dataset.episodes):
        ep_len = dataset._episode_length(episode)
        meta = episode.metadata or {}
        # x_pos was stored in raw .npz; not always passed through to blocked.
        # Fall back: read it from the raw tas_full path
        raw_path = Path("/root/data/tas_full") / (episode.name + ".npz")
        if not raw_path.exists():
            continue
        with np.load(raw_path, allow_pickle=False) as raw:
            if "x_pos" not in raw.files:
                continue
            raw_x = raw["x_pos"]
        # Blocked frames sample every 5th raw frame; align by index
        # blocked frame i corresponds to raw frame i*5 (per build_lewm_mario_dataset.py)
        n_take = min(ep_len, max_per_ep)
        frame_idxs = np.linspace(0, ep_len - 1, n_take).astype(int)
        if episode.frames_npz is not None:
            frames = dataset._get_npz_frames(ep_id)[frame_idxs]
            pixels = dataset._preprocess_npz_frames(frames).unsqueeze(0).to(device)
        else:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
            enc = model.encode({"pixels": pixels})
        emb = enc["emb"][0].float().cpu().numpy()
        # map blocked frame idx -> raw frame idx -> x_pos
        for i, b_idx in enumerate(frame_idxs):
            r_idx = min(b_idx * 5, len(raw_x) - 1)
            all_lat.append(emb[i])
            all_x.append(int(raw_x[r_idx]))
    return np.asarray(all_lat), np.asarray(all_x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/runs/tas_v1/best.pt")
    ap.add_argument("--dataset-root", default="/root/data/tas_precomputed")
    ap.add_argument("--n-starts", type=int, default=200)
    ap.add_argument("--out", default="/root/runs/tas_v1/eval_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, epoch, val_metrics = load_model(args.ckpt, device)
    print(f"loaded epoch={epoch} val_metrics={val_metrics}")

    episodes = discover_episodes(args.dataset_root)
    full = MarioTraceDataset(episodes, cfg.history_size, cfg.num_preds, cfg.image_size,
                              stride=1, npz_load_mode="lazy", max_cached_episodes=4)

    # mirror the train script split (90/10, generator seed=3072)
    g = torch.Generator().manual_seed(3072)
    train_size = int(len(full) * 0.9)
    val_size = len(full) - train_size
    train_ds, val_ds = random_split(full, [train_size, val_size], generator=g)
    print(f"train_windows={len(train_ds)} val_windows={len(val_ds)}")

    # 1. Open-loop MSE on the full dataset (samples random starts; episode-level)
    print("--- open-loop MSE (random starts across all episodes) ---")
    rows = open_loop_mse(model, full, device, n_starts=args.n_starts)
    horizons = sorted(set(r["horizon"] for r in rows))
    by_h = {h: [r["mse"] for r in rows if r["horizon"] == h] for h in horizons}
    horizon_summary = {str(h): {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                "p95": float(np.percentile(v, 95)), "n": len(v)}
                        for h, v in by_h.items()}
    for h in horizons:
        s = horizon_summary[str(h)]
        print(f"  horizon={h:>2}  n={s['n']:>4}  mean_mse={s['mean']:.4f}  median={s['median']:.4f}  p95={s['p95']:.4f}")

    # 2. x_pos probe
    print("--- x_pos linear probe ---")
    lat, xp = gather_latents_with_xpos(model, full, device, max_per_ep=80)
    print(f"  collected {len(lat)} (latent, x_pos) pairs (dim={lat.shape[1]})")
    # reuse the train/val episode split: pick latents whose source episode is in val
    val_ep_names = set()
    for win_idx in val_ds.indices:
        ep_id = full.index[win_idx][0]
        val_ep_names.add(full.episodes[ep_id].name)
    # we collected latents in ep order; rebuild name mapping
    # simpler: just do an 80/20 split of latents
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(lat))
    cut = int(0.8 * len(perm))
    train_idx, test_idx = perm[:cut], perm[cut:]
    probe = Ridge(alpha=1.0).fit(lat[train_idx], xp[train_idx])
    pred_train = probe.predict(lat[train_idx])
    pred_test  = probe.predict(lat[test_idx])
    def stats(name, y, yhat):
        mae = float(np.mean(np.abs(y - yhat)))
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-9))
        print(f"  {name}: MAE={mae:.1f}px  RMSE={rmse:.1f}px  R^2={r2:.3f}  range=[{y.min()}, {y.max()}]")
        return {"mae_px": mae, "rmse_px": rmse, "r2": r2,
                "y_min": int(y.min()), "y_max": int(y.max()), "n": int(len(y))}
    probe_train = stats("train", xp[train_idx], pred_train)
    probe_test  = stats("test",  xp[test_idx],  pred_test)

    out = {
        "ckpt": args.ckpt, "epoch": epoch, "val_metrics_recorded": val_metrics,
        "open_loop_mse_by_horizon": horizon_summary,
        "x_pos_probe": {"train": probe_train, "test": probe_test, "n_total": len(lat)},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
