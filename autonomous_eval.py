"""Autonomous Mario play with LeWM + reward head.

CEM in latent space with cost = -Σ r̂(ẑ_t) over the planning horizon.
No goal frame — Mario plays until death or step budget.
Captures every emulator frame to mp4 for the first --max-videos episodes.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import cv2
import gym_super_mario_bros

sys.path.insert(0, "/root/lewm_mario")
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.fm2 import fm2_row_to_nes_action

RAM_LIVES, RAM_X_PAGE, RAM_X_PIXEL = 0x075A, 0x6D, 0x86
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def make_env(): return gym_super_mario_bros.make("SuperMarioBros-1-1-v0").unwrapped
def read_xpos(env): return int(env.ram[RAM_X_PAGE]) * 256 + int(env.ram[RAM_X_PIXEL])
def read_lives(env): return int(np.int8(env.ram[RAM_LIVES]))

class RewardHead(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, z): return self.net(z).squeeze(-1)

def preprocess(frames_uint8, image_size):
    t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-1] != image_size:
        t = torch.nn.functional.interpolate(t, size=(image_size, image_size),
                                             mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std

@torch.no_grad()
def encode(model, frames_uint8, image_size, device):
    pix = preprocess(frames_uint8, image_size).to(device).unsqueeze(0)
    return model.encode({"pixels": pix})["emb"][0]

@torch.no_grad()
def cem_plan_reward(model, head, history_emb, action_library, horizon,
                    n_samples, n_iters, elite_frac, device, reward_mean, reward_std):
    """CEM that maximizes Σ r̂(ẑ_t) along the predicted rollout."""
    history = history_emb.size(0)
    N = action_library.size(0)
    elite_n = max(1, int(round(n_samples * elite_frac)))
    logits = torch.zeros(horizon, N, device=device)
    best_score = torch.tensor(float("-inf"), device=device)
    best_plan = None
    history_emb_b = history_emb.unsqueeze(0)
    for _ in range(n_iters):
        probs = logits.softmax(dim=-1)
        idxs = torch.multinomial(probs, num_samples=n_samples, replacement=True).T
        candidate_actions = action_library[idxs]
        emb = history_emb_b.expand(n_samples, -1, -1).clone()
        rewards_per_step = []
        for h in range(horizon):
            ctx_act = candidate_actions[:, max(0, h - history + 1): h + 1]
            if ctx_act.size(1) < history:
                pad = torch.zeros(n_samples, history - ctx_act.size(1),
                                   action_library.size(1), device=device)
                ctx_act = torch.cat([pad, ctx_act], dim=1)
            ctx_act_emb = model.action_encoder(ctx_act)
            pred = model.predict(emb[:, -history:], ctx_act_emb)
            emb = torch.cat([emb, pred[:, -1:]], dim=1)
            r_norm = head(pred[:, -1])
            rewards_per_step.append(r_norm)
        score = torch.stack(rewards_per_step, dim=1).sum(dim=1)  # (S,)
        elite_ids = score.topk(elite_n, largest=True).indices
        elite_idxs = idxs[elite_ids]
        new_logits = torch.zeros_like(logits)
        for step in range(horizon):
            counts = torch.bincount(elite_idxs[:, step], minlength=N).float()
            new_logits[step] = (counts + 1e-3).log()
        logits = new_logits
        round_best = score[elite_ids[0]]
        if round_best > best_score:
            best_score = round_best
            best_plan = candidate_actions[elite_ids[0]]
    return best_plan, best_score.item()

def annotate(canvas, lines):
    y = 16
    for line in lines:
        cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        y += 14

def write_video(frames, path, fps=30):
    if not frames: return
    H, W, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    for f in frames: vw.write(f)
    vw.release()

def run_one(model, head, cfg, total_blocks, horizon, n_samples, n_iters,
            action_library, device, video_path, reward_mean, reward_std):
    history = cfg.history_size; image_size = cfg.image_size; block = 5
    env = make_env(); env.reset()
    out = env.step(0); cur_frame = out[0].astype(np.uint8)
    init = np.stack([cur_frame] * history, axis=0)
    history_emb = encode(model, init, image_size, device)
    recent = [cur_frame] * history
    video_frames = [] if video_path else None
    x0 = read_xpos(env)
    cur_block = 0
    while cur_block < total_blocks:
        plan, score = cem_plan_reward(model, head, history_emb, action_library,
                                       horizon, n_samples, n_iters, 0.1, device,
                                       reward_mean, reward_std)
        macro = plan[0].view(block, 8).cpu().numpy()
        block_died = False
        for sub in range(block):
            out = env.step(fm2_row_to_nes_action(macro[sub]))
            obs = out[0]; done = out[2] if len(out) >= 3 else False
            cx, cl = read_xpos(env), read_lives(env)
            if video_path:
                bgr = cv2.cvtColor(obs.astype(np.uint8), cv2.COLOR_RGB2BGR)
                annotate(bgr, [
                    f"step {cur_block*block + sub}  x={cx}  lives={cl}",
                    f"plan_reward_score={score:.2f}",
                ])
                video_frames.append(bgr)
            if done or cl < 0:
                block_died = True; break
        if block_died:
            break
        recent.append(obs.astype(np.uint8)); recent = recent[-history:]
        history_emb = encode(model, np.stack(recent), image_size, device)
        cur_block += 1
    final_x = read_xpos(env); final_l = read_lives(env)
    env.close()
    if video_path:
        write_video(video_frames, video_path, fps=30)
    return {"x_start": x0, "x_final": final_x, "x_progress": final_x - x0,
            "blocks_executed": cur_block, "final_lives": final_l,
            "video": str(video_path) if video_path else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/ckpt/best.pt")
    ap.add_argument("--reward-head", default="/root/runs/reward_head/reward_head.pt")
    ap.add_argument("--out-dir", default="/root/auto_videos", type=Path)
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=4)
    ap.add_argument("--total-blocks", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--n-iters", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])
    model = LeWorldModel(cfg); model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    action_library = ck["action_library"].to(device)

    rh_ck = torch.load(args.reward_head, map_location="cpu", weights_only=False)
    head = RewardHead(in_dim=cfg.action_embed_dim).to(device)
    head.load_state_dict(rh_ck["head_state"]); head.eval()
    for p in head.parameters(): p.requires_grad_(False)
    rm = rh_ck["config"]["reward_mean"]; rs = rh_ck["config"]["reward_std"]
    print(f"loaded model epoch={ck['epoch']}  reward_head mean={rm:.2f} std={rs:.2f}")

    all_results = []
    for i in range(args.n_episodes):
        save_video = i < args.max_videos
        vp = args.out_dir / f"auto_ep_{i:02d}.mp4" if save_video else None
        res = run_one(model, head, cfg, args.total_blocks, args.horizon,
                       args.n_samples, args.n_iters, action_library, device, vp, rm, rs)
        print(json.dumps({"ep": i, **res}), flush=True)
        all_results.append(res)
    (args.out_dir / "auto_eval_summary.json").write_text(json.dumps(all_results, indent=2))
    if all_results:
        x = [r["x_progress"] for r in all_results]
        print(f"\n=== summary over {len(all_results)} episodes ===")
        print(f"x_progress: mean={np.mean(x):.1f}  median={np.median(x):.1f}  "
              f"max={max(x)}  min={min(x)}")
        deaths = sum(1 for r in all_results if r["final_lives"] < r.get("start_lives", 2))
        print(f"final_x: max={max(r['x_final'] for r in all_results)}")

if __name__ == "__main__":
    main()
