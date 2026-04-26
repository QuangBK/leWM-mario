"""Online goal-conditioned eval of LeWM Mario.

Per episode:
  1. Reset gym-super-mario-bros, replay a 'warmup' from a chosen TAS to seed Mario in-game.
  2. Take frame at warmup_end as the start; encode it -> z_start.
  3. Take frame at warmup_end + goal_offset as the goal; encode -> z_goal.
  4. CEM in latent space: sample action sequences from the trained action library,
     roll out the predictor for `horizon` blocks, score by latent-MSE to z_goal,
     update via elite stats. Repeat n_iters.
  5. Execute the first block of the best plan in the env. Roll the goal forward.

Captures every emulator frame to /root/eval_videos/<episode>.mp4 with overlays
(real | goal panels), only for the first --max-videos episodes.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
import cv2
import gym_super_mario_bros
from nes_py import NESEnv

sys.path.insert(0, "/root/lewm_mario")
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.fm2 import parse_fm2, fm2_row_to_nes_action

RAM_LIVES, RAM_X_PAGE, RAM_X_PIXEL = 0x075A, 0x6D, 0x86

def make_smb_env():
    """Returns the underlying SuperMarioBrosEnv (NESEnv subclass) so we can
    step with raw 8-bit FM2 bytes and read RAM directly. .unwrapped peels off
    all the gym wrappers (TimeLimit, etc.) that constrain actions."""
    return gym_super_mario_bros.make("SuperMarioBros-1-1-v0").unwrapped
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def read_xpos(env): return int(env.ram[RAM_X_PAGE]) * 256 + int(env.ram[RAM_X_PIXEL])
def read_lives(env): return int(np.int8(env.ram[RAM_LIVES]))

def preprocess(frames_uint8: np.ndarray, image_size: int) -> torch.Tensor:
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
def cem_plan(model, history_emb, action_library, goal_emb, horizon,
             n_samples, n_iters, elite_frac, device):
    history = history_emb.size(0)
    N = action_library.size(0)
    elite_n = max(1, int(round(n_samples * elite_frac)))
    logits = torch.zeros(horizon, N, device=device)
    best_cost = torch.tensor(float("inf"), device=device)
    best_plan = None
    history_emb_b = history_emb.unsqueeze(0)
    for _ in range(n_iters):
        probs = logits.softmax(dim=-1)
        idxs = torch.multinomial(probs, num_samples=n_samples, replacement=True).T
        candidate_actions = action_library[idxs]  # (S, horizon, A_raw)
        emb = history_emb_b.expand(n_samples, -1, -1).clone()
        for h in range(horizon):
            ctx_act = candidate_actions[:, max(0, h - history + 1): h + 1]
            if ctx_act.size(1) < history:
                pad = torch.zeros(n_samples, history - ctx_act.size(1),
                                   action_library.size(1), device=device)
                ctx_act = torch.cat([pad, ctx_act], dim=1)
            ctx_act_emb = model.action_encoder(ctx_act)
            pred = model.predict(emb[:, -history:], ctx_act_emb)
            emb = torch.cat([emb, pred[:, -1:]], dim=1)
        final_pred = emb[:, -1]
        cost = ((final_pred - goal_emb.unsqueeze(0)) ** 2).mean(dim=-1)
        elite_ids = cost.topk(elite_n, largest=False).indices
        elite_idxs = idxs[elite_ids]
        new_logits = torch.zeros_like(logits)
        for step in range(horizon):
            counts = torch.bincount(elite_idxs[:, step], minlength=N).float()
            new_logits[step] = (counts + 1e-3).log()
        logits = new_logits
        round_best_cost = cost[elite_ids[0]]
        if round_best_cost < best_cost:
            best_cost = round_best_cost
            best_plan = candidate_actions[elite_ids[0]]
    return best_plan, best_cost.item()

def replay(env, fm2_actions, n_steps):
    for t in range(n_steps):
        env.step(fm2_row_to_nes_action(fm2_actions[t]))

def annotate(canvas, lines):
    y = 16
    for line in lines:
        cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        y += 14

def stitch(real_bgr, goal_bgr, lines):
    H, W, _ = real_bgr.shape
    canvas = np.zeros((H, W * 2 + 8, 3), dtype=np.uint8)
    canvas[:, :W] = real_bgr
    canvas[:, W + 8:] = goal_bgr
    annotate(canvas, lines)
    return canvas

def write_video(frames, path, fps=30):
    if not frames: return
    H, W, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    for f in frames: vw.write(f)
    vw.release()

def run_one(model, cfg, fm2_path, menu_skip, total_blocks, goal_offset_blocks,
            horizon, n_samples, n_iters, action_library, device, video_path):
    """
    `menu_skip` = how many leading FM2 frames to drop (the title/menu portion).
    SuperMarioBrosEnv.reset() puts Mario in-game already, so menu inputs would pause/desync.
    """
    history = cfg.history_size
    image_size = cfg.image_size
    block = 5
    fm2 = parse_fm2(fm2_path)
    in_game = fm2[menu_skip:]
    if len(in_game) < (goal_offset_blocks + total_blocks + history) * block:
        return None
    env = make_smb_env(); env.reset()
    oracle = make_smb_env(); oracle.reset()
    future_frames = []
    oracle_done = False
    for t in range((goal_offset_blocks + total_blocks + history) * block):
        if t >= len(in_game) or oracle_done: break
        out = oracle.step(fm2_row_to_nes_action(in_game[t]))
        obs = out[0]; oracle_done = out[2] if len(out) >= 3 else False
        if t % block == 0:
            future_frames.append(obs.astype(np.uint8))
    oracle.close()
    if len(future_frames) < goal_offset_blocks + 1:
        env.close(); return None
    future_frames = np.stack(future_frames)
    future_emb = encode(model, future_frames, image_size, device)

    # initial history: step env one no-op to get the current screen
    out = env.step(0)
    cur_frame = out[0].astype(np.uint8)
    init_hist = np.stack([cur_frame] * history, axis=0)
    history_emb = encode(model, init_hist, image_size, device)

    video_frames = []
    x0 = read_xpos(env)
    cur_block = 0
    recent = [cur_frame] * history
    save_video = video_path is not None
    while cur_block < total_blocks:
        goal_idx = min(goal_offset_blocks + cur_block, future_emb.size(0) - 1)
        z_goal = future_emb[goal_idx]
        best_plan, best_cost = cem_plan(model, history_emb, action_library, z_goal,
                                         horizon, n_samples, n_iters, 0.1, device)
        macro = best_plan[0].view(block, 8).cpu().numpy()
        for sub in range(block):
            out = env.step(fm2_row_to_nes_action(macro[sub]))
            obs = out[0]; done = out[2] if len(out) >= 3 else False
            cx, cl = read_xpos(env), read_lives(env)
            if save_video:
                real_bgr = cv2.cvtColor(obs.astype(np.uint8), cv2.COLOR_RGB2BGR)
                gi = min(goal_offset_blocks + cur_block, len(future_frames) - 1)
                goal_bgr = cv2.cvtColor(future_frames[gi], cv2.COLOR_RGB2BGR)
                video_frames.append(stitch(real_bgr, goal_bgr, [
                    f"step {cur_block*block + sub}  x={cx}  lives={cl}",
                    f"goal_block={gi}  cem_cost={best_cost:.3f}",
                    "real | goal",
                ]))
            if done or cl < 0: break
        if done or read_lives(env) < 0: break
        recent.append(obs.astype(np.uint8))
        recent = recent[-history:]
        history_emb = encode(model, np.stack(recent), image_size, device)
        cur_block += 1
    final_x = read_xpos(env)
    env.close()
    if save_video:
        write_video(video_frames, video_path, fps=30)
    return {
        "fm2": Path(fm2_path).name,
        "x_start": x0, "x_final": final_x, "x_progress": final_x - x0,
        "blocks_executed": cur_block,
        "video": str(video_path) if save_video else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/ckpt/best.pt")
    ap.add_argument("--traces-dir", default="/root/lewm_mario/traces")
    ap.add_argument("--out-dir", default="/root/eval_videos", type=Path)
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=3)
    ap.add_argument("--menu-skip", type=int, default=200,
                    help="FM2 frames to drop (typical title/menu length); env starts already in 1-1")
    ap.add_argument("--total-blocks", type=int, default=40)
    ap.add_argument("--goal-offset-blocks", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--n-iters", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])
    model = LeWorldModel(cfg)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    action_library = ck["action_library"].to(device)
    print(f"loaded ckpt epoch={ck['epoch']} action_lib={action_library.size(0)} "
          f"image={cfg.image_size} history={cfg.history_size}")

    fm2_paths = sorted(Path(args.traces_dir).glob("*.fm2"))
    rng = np.random.default_rng(42)
    pool = list(rng.permutation(len(fm2_paths)))

    all_results = []; ok = 0; vids = 0
    for idx in pool:
        if ok >= args.n_episodes: break
        fm2 = fm2_paths[int(idx)]
        save_video = vids < args.max_videos
        vp = (args.out_dir / f"ep_{ok:02d}_{fm2.stem.replace(' ', '_').replace(',','')[:50]}.mp4"
              if save_video else None)
        print(f"running on {fm2.name}{' [video]' if save_video else ''}", flush=True)
        try:
            res = run_one(model, cfg, fm2, args.menu_skip, args.total_blocks,
                           args.goal_offset_blocks, args.horizon, args.n_samples,
                           args.n_iters, action_library, device, vp)
        except Exception as e:
            print(f"  failed: {e!r}"); continue
        if res is None:
            print(f"  skipped (warmup desync or short TAS)"); continue
        print(f"  result: {json.dumps(res)}")
        all_results.append(res); ok += 1
        if save_video: vids += 1

    (args.out_dir / "online_eval_summary.json").write_text(json.dumps(all_results, indent=2))
    if all_results:
        x = [r["x_progress"] for r in all_results]
        print(f"\n=== summary over {len(all_results)} episodes ===")
        print(f"x_progress: mean={np.mean(x):.1f}  median={np.median(x):.1f}  "
              f"max={max(x)}  min={min(x)}")
        print(f"videos: {vids} files in {args.out_dir}/")

if __name__ == "__main__":
    main()
