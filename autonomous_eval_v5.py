"""autonomous_eval_v5 = v4 (past-actions + stuck detector) + RC-aux gate.

Reward-conditioned discriminator variant of the RC-aux gate (paper is
goal-conditioned only). For each CEM candidate, score:

    R = R_phi(z_t, ẑ_{t+H}, H)

where ẑ_{t+H} is the planner's own predicted endpoint. Then gate the CEM
score via cost-space multiplication (eq 25 inverted for maximization):

    cost = score.max() - score                # 0 for best, >0 for worse
    cost_rc = cost * max(floor, 1 − λ_plan·R)
    score_rc = score.max() - cost_rc

Reachable plans (high R) get their cost gap to the top scaled down, pulling
them up. Unreachable plans (low R) keep their full cost gap. λ=0 recovers
the base v4 planner.

Attacks the x=898 OOD-overconfidence found in diagnose_x898.py: if at
x=895 the predictor hallucinates ẑ_{t+H} corresponding to "x≈1300", R_phi
will score low (no such transitions in training), and CEM will no longer
favor that plan over a plan whose predicted endpoint stays in-distribution.
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
from rc_aux import ReachabilityHead, endpoint_reachability_score

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
def cem_plan_with_past(model, head, history_emb, action_library, horizon, n_samples,
                        n_iters, elite_frac, device, past_actions,
                        reach_head=None, lambda_plan=0.0, gate_floor=0.1):
    """past_actions: tensor of shape [N_past, 40] of recent EXECUTED block actions
    (most recent last). Will be used as the prefix for the predictor's ctx_act
    instead of zero-padding.

    reach_head/lambda_plan/gate_floor: optional RC-aux endpoint-reachability gate.
    """
    history = history_emb.size(0)
    N = action_library.size(0)
    action_dim = action_library.size(1)
    elite_n = max(1, int(round(n_samples * elite_frac)))
    logits = torch.zeros(horizon, N, device=device)
    best_score = torch.tensor(float("-inf"), device=device); best_plan = None
    history_emb_b = history_emb.unsqueeze(0)

    # Build the past-action prefix that will be prepended at h=0.
    # past_actions has shape [N_past, action_dim]. Take the last `history-1`
    # (since the current candidate action contributes the history-th slot).
    if past_actions.numel() > 0:
        past_use = past_actions[-(history - 1):]  # up to history-1 most recent
    else:
        past_use = torch.zeros(0, action_dim, device=device)
    n_past_have = past_use.size(0)
    n_pad_needed = max(0, (history - 1) - n_past_have)
    if n_pad_needed > 0:
        pad = torch.zeros(n_pad_needed, action_dim, device=device)
        past_use = torch.cat([pad, past_use], dim=0) if past_use.numel() > 0 else pad
    # past_use now has shape [history-1, action_dim]

    for _ in range(n_iters):
        probs = logits.softmax(dim=-1)
        idxs = torch.multinomial(probs, num_samples=n_samples, replacement=True).T
        candidate_actions = action_library[idxs]  # [n_samples, horizon, action_dim]
        emb = history_emb_b.expand(n_samples, -1, -1).clone()
        rewards = []
        # Build full action sequence: [past_use (history-1)] + candidates (horizon)
        past_b = past_use.unsqueeze(0).expand(n_samples, -1, -1)  # [n_samples, history-1, action_dim]
        full_actions = torch.cat([past_b, candidate_actions], dim=1)  # [n_samples, history-1+horizon, action_dim]
        for h in range(horizon):
            # The (history-1) slots before h are past+earlier-candidates,
            # then the current candidate slot at index history-1+h
            start = h  # offset: window of `history` actions ending at history-1+h
            end = h + history
            ctx_act = full_actions[:, start:end]
            ctx_act_emb = model.action_encoder(ctx_act)
            pred = model.predict(emb[:, -history:], ctx_act_emb)
            emb = torch.cat([emb, pred[:, -1:]], dim=1)
            rewards.append(head(pred[:, -1]))
        score = torch.stack(rewards, dim=1).sum(dim=1)
        # RC-aux endpoint-reachability gate (discriminator on predicted endpoint).
        if reach_head is not None and lambda_plan > 0.0:
            src_anchor = history_emb[-1]  # z_t = last context latent
            endpoint = emb[:, -1]          # ẑ_{t+H}
            R = endpoint_reachability_score(reach_head, src_anchor, endpoint, horizon)
            # cost-space gate, then re-flip to score
            best = score.max()
            cost = best - score
            mult = torch.clamp(1.0 - lambda_plan * R, min=gate_floor)
            cost_rc = cost * mult
            score = best - cost_rc
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
            stuck_window, stuck_threshold, stuck_recover_blocks,
            reach_head=None, lambda_plan=0.0, gate_floor=0.1):
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
    # Track past executed action blocks (40-d each)
    past_actions = torch.zeros(0, action_library.size(1), device=device)
    while cur_block < total_blocks:
        if recover_left > 0:
            idx = int(rng.integers(0, action_library.size(0)))
            plan_block = action_library[idx].view(block, 8)
            score_val = float("nan")
            recover_left -= 1
            mode = "RND"
            executed_action = action_library[idx].clone()  # 40-d
        else:
            plan, score = cem_plan_with_past(model, head, history_emb, action_library, horizon,
                                              n_samples, n_iters, 0.1, device, past_actions,
                                              reach_head=reach_head,
                                              lambda_plan=lambda_plan,
                                              gate_floor=gate_floor)
            plan_block = plan[0].view(block, 8)
            score_val = score
            mode = "CEM"
            executed_action = plan[0].clone().view(-1)  # flatten to 40-d
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
        # Append executed action to past_actions, keep last `history` blocks
        past_actions = torch.cat([past_actions, executed_action.unsqueeze(0)], dim=0)
        if past_actions.size(0) > history:
            past_actions = past_actions[-history:]
        x_history.append(read_xpos(env))
        if recover_left == 0 and len(x_history) >= stuck_window:
            window = x_history[-stuck_window:]
            if max(window) - min(window) < stuck_threshold:
                recover_left = stuck_recover_blocks
                n_stuck_recoveries += 1
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
    ap.add_argument("--out-dir", default="/root/auto_videos_v4", type=Path)
    ap.add_argument("--label", default="v4")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--max-videos", type=int, default=4)
    ap.add_argument("--total-blocks", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--n-iters", type=int, default=8)
    ap.add_argument("--stuck-window", type=int, default=30)
    ap.add_argument("--stuck-threshold", type=int, default=5)
    ap.add_argument("--stuck-recover-blocks", type=int, default=4)
    ap.add_argument("--lambda-plan", type=float, default=0.0,
                    help="RC-aux endpoint-reachability gate strength. 0 = base planner.")
    ap.add_argument("--gate-floor", type=float, default=0.1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head, action_library, rm, rs, cfg = load_artifacts(args, device)
    print(f"v5 (past-actions CEM + RC-aux gate)  label={args.label}  "
          f"reward_mean={rm:.2f}  reward_std={rs:.2f}")

    # Optionally load the reachability head from the ckpt (if --lambda-plan > 0).
    reach_head = None
    if args.lambda_plan > 0.0:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if "reach_head_state" in ck:
            rc_cfg = ck.get("rc_aux", {})
            reach_head = ReachabilityHead(
                embed_dim=cfg.action_embed_dim,
                hidden_dim=int(rc_cfg.get("head_hidden", 512)),
                max_horizon=int(rc_cfg.get("h_max", 8)),
            ).to(device).eval()
            reach_head.load_state_dict(ck["reach_head_state"])
            for p in reach_head.parameters(): p.requires_grad_(False)
            print(f"loaded reach_head (h_max={reach_head.max_horizon}) "
                  f"λ_plan={args.lambda_plan}")
        else:
            print(f"WARNING: --lambda-plan={args.lambda_plan} > 0 but ckpt has no reach_head_state. "
                  "Running ungated.")

    results = []
    for i in range(args.n_episodes):
        save_video = i < args.max_videos
        vp = args.out_dir / f"{args.label}_ep_{i:02d}.mp4" if save_video else None
        res = run_one(model, head, cfg, args.total_blocks, args.horizon, args.n_samples,
                       args.n_iters, action_library, device, vp, args.label,
                       args.stuck_window, args.stuck_threshold, args.stuck_recover_blocks,
                       reach_head=reach_head,
                       lambda_plan=args.lambda_plan,
                       gate_floor=args.gate_floor)
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
