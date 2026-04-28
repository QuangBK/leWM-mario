"""Same as autonomous_eval_v2 but with a "stuck detector" that randomizes
the next K blocks of actions when Mario hasn't moved in W blocks. Helps the
planner break out of attractors like x=898 (the tall pipe in 1-1).

New flags:
  --stuck-window 30       blocks of x_pos history to inspect
  --stuck-threshold 5     px — if max(x[-W:]) - min(x[-W:]) < threshold, stuck
  --stuck-recover-blocks 4  number of blocks to randomize when stuck
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
    def __init__(self, in_dim, hidden=64, n_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z).squeeze(-1)

def load_artifacts(args, device):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])
    model = LeWorldModel(cfg); model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    action_library = ck["action_library"].to(device)
    if "head_state" in ck:
        rm, rs = float(ck["reward_mean"]), float(ck["reward_std"])
        head = RewardHead(in_dim=cfg.action_embed_dim, hidden=128, n_layers=2).to(device)
        head.load_state_dict(ck["head_state"])
        head.eval()
        for p in head.parameters(): p.requires_grad_(False)
        return model, head, action_library, rm, rs, cfg
    rh_ck = torch.load(args.reward_head, map_location="cpu", weights_only=False)
    rh_cfg = rh_ck["config"]
    hidden = int(rh_cfg.get("hidden", 64))
    n_lin = sum(1 for k in rh_ck["head_state"] if k.endswith(".weight") and "net" in k)
    n_layers = max(1, n_lin - 1)
    head = RewardHead(in_dim=cfg.action_embed_dim, hidden=hidden, n_layers=n_layers).to(device)
    head.load_state_dict(rh_ck["head_state"]); head.eval()
    for p in head.parameters(): p.requires_grad_(False)
    return model, head, action_library, float(rh_cfg["reward_mean"]), float(rh_cfg["reward_std"]), cfg

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
def cem_plan(model, head, history_emb, action_library, horizon, n_samples,
             n_iters, elite_frac, device):
    history = history_emb.size(0)
    N = action_library.size(0)
    elite_n = max(1, int(round(n_samples * elite_frac)))
    logits = torch.zeros(horizon, N, device=device)
    best_score = torch.tensor(float("-inf"), device=device); best_plan = None
    history_emb_b = history_emb.unsqueeze(0)
    for _ in range(n_iters):
        probs = logits.softmax(dim=-1)
        idxs = torch.multinomial(probs, num_samples=n_samples, replacement=True).T
        candidate_actions = action_library[idxs]
        emb = history_emb_b.expand(n_samples, -1, -1).clone()
        rewards = []
        for h in range(horizon):
            ctx_act = candidate_actions[:, max(0, h - history + 1): h + 1]
            if ctx_act.size(1) < history:
                pad = torch.zeros(n_samples, history - ctx_act.size(1),
                                   action_library.size(1), device=device)
                ctx_act = torch.cat([pad, ctx_act], dim=1)
            ctx_act_emb = model.action_encoder(ctx_act)
            pred = model.predict(emb[:, -history:], ctx_act_emb)
            emb = torch.cat([emb, pred[:, -1:]], dim=1)
            rewards.append(head(pred[:, -1]))
        score = torch.stack(rewards, dim=1).sum(dim=1)
        elite_ids = score.topk(elite_n, largest=True).indices
        elite_idxs = idxs[elite_ids]
        new_logits = torch.zeros_like(logits)
        for s in range(horizon):
            counts = torch.bincount(elite_idxs[:, s], minlength=N).float()
            new_logits[s] = (counts + 1e-3).log()
        logits = new_logits
        rb = score[elite_ids[0]]
        if rb > best_score:
            best_score = rb; best_plan = candidate_actions[elite_ids[0]]
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
            action_library, device, video_path, label,
            stuck_window, stuck_threshold, stuck_recover_blocks):
    history = cfg.history_size; image_size = cfg.image_size; block = 5
    env = make_env(); env.reset()
    out = env.step(0); cur_frame = out[0].astype(np.uint8)
    init = np.stack([cur_frame] * history, axis=0)
    history_emb = encode(model, init, image_size, device)
    recent = [cur_frame] * history
    video_frames = [] if video_path else None
    x0 = read_xpos(env); cur_block = 0
    x_history = []
    rng = np.random.default_rng()
    recover_left = 0
    n_stuck_recoveries = 0
    while cur_block < total_blocks:
        if recover_left > 0:
            # Sample random action from library to force exploration
            idx = int(rng.integers(0, action_library.size(0)))
            plan_block = action_library[idx].view(block, 8)
            score_val = float("nan")
            recover_left -= 1
            mode = "RND"
        else:
            plan, score = cem_plan(model, head, history_emb, action_library, horizon,
                                    n_samples, n_iters, 0.1, device)
            plan_block = plan[0].view(block, 8)
            score_val = score
            mode = "CEM"
        macro = plan_block.cpu().numpy()
        died = False
        for sub in range(block):
            o = env.step(fm2_row_to_nes_action(macro[sub]))
            obs = o[0]; done = o[2] if len(o) >= 3 else False
            cx, cl = read_xpos(env), read_lives(env)
            if video_path:
                bgr = cv2.cvtColor(obs.astype(np.uint8), cv2.COLOR_RGB2BGR)
                annotate(bgr, [
                    f"[{label}] step {cur_block*block + sub}  x={cx}  lives={cl}",
                    f"{mode} score={score_val:.2f}  recoveries={n_stuck_recoveries}",
                ])
                video_frames.append(bgr)
            if done or cl < 0: died = True; break
        if died: break
        recent.append(obs.astype(np.uint8)); recent = recent[-history:]
        history_emb = encode(model, np.stack(recent), image_size, device)
        x_history.append(read_xpos(env))
        # Stuck detector: if last `stuck_window` x positions are all within
        # `stuck_threshold` of each other, fire `stuck_recover_blocks` of random play.
        if recover_left == 0 and len(x_history) >= stuck_window:
            window = x_history[-stuck_window:]
            if max(window) - min(window) < stuck_threshold:
                recover_left = stuck_recover_blocks
                n_stuck_recoveries += 1
                # clear the window so we don't re-fire immediately after recovery
                x_history = x_history[-1:]
        cur_block += 1
    final_x = read_xpos(env); final_l = read_lives(env)
    env.close()
    if video_path: write_video(video_frames, video_path, fps=30)
    return {"x_start": x0, "x_final": final_x, "x_progress": final_x - x0,
            "blocks_executed": cur_block, "final_lives": final_l,
            "n_stuck_recoveries": n_stuck_recoveries,
            "video": str(video_path) if video_path else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/ckpt/best.pt")
    ap.add_argument("--reward-head", default=None)
    ap.add_argument("--out-dir", default="/root/auto_videos_v3", type=Path)
    ap.add_argument("--label", default="v3")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--max-videos", type=int, default=4)
    ap.add_argument("--total-blocks", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--n-iters", type=int, default=8)
    ap.add_argument("--stuck-window", type=int, default=30)
    ap.add_argument("--stuck-threshold", type=int, default=5)
    ap.add_argument("--stuck-recover-blocks", type=int, default=4)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head, action_library, rm, rs, cfg = load_artifacts(args, device)
    print(f"loaded label={args.label}  reward_mean={rm:.2f}  reward_std={rs:.2f}")
    print(f"stuck-detector: window={args.stuck_window} threshold={args.stuck_threshold} "
          f"recover_blocks={args.stuck_recover_blocks}")

    results = []
    for i in range(args.n_episodes):
        save_video = i < args.max_videos
        vp = args.out_dir / f"{args.label}_ep_{i:02d}.mp4" if save_video else None
        res = run_one(model, head, cfg, args.total_blocks, args.horizon, args.n_samples,
                       args.n_iters, action_library, device, vp, args.label,
                       args.stuck_window, args.stuck_threshold, args.stuck_recover_blocks)
        print(json.dumps({"ep": i, **res}), flush=True)
        results.append(res)
    summary_path = args.out_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    if results:
        x = [r["x_progress"] for r in results]
        print(f"\n=== summary [{args.label}] over {len(results)} episodes ===")
        print(f"x_progress: mean={np.mean(x):.1f}  median={np.median(x):.1f}  "
              f"max={max(x)}  min={min(x)}")
        deaths = sum(1 for r in results if r["final_lives"] < 2)
        recoveries = sum(r.get("n_stuck_recoveries", 0) for r in results)
        print(f"deaths: {deaths}/{len(results)}  final_x_max={max(r['x_final'] for r in results)}  "
              f"total_recoveries={recoveries}")

if __name__ == "__main__":
    main()
