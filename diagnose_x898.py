"""Diagnostic: probe what the joint_h8 model predicts at x≈898 (the parking
point). Tests three hypotheses for why Mario parks there:

  H1 — predictor sees death past the pipe → predicts low reward for run-right
  H2 — action library doesn't contain a successful pipe-jump sequence
  H3 — CEM gets stuck in a local optimum even though good plans exist

Method: drive Mario to x≈898 with a known-working prefix (CEM with the model,
use first ~120 blocks), then at that state evaluate every action in the
library across many candidate 8-block plans:
  - distribution of predicted total reward (h=8 sum)
  - top-K plans and their first-block button signatures
  - mean action library entry "lean direction" (does any entry encode 'jump
    over tall pipe'?)
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import gym_super_mario_bros

sys.path.insert(0, "/root/lewm_mario")
from mario_lewm.model import LeWorldModel, LeWorldModelConfig
from mario_lewm.fm2 import fm2_row_to_nes_action

RAM_LIVES, RAM_X_PAGE, RAM_X_PIXEL = 0x075A, 0x6D, 0x86
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FM2_NAMES = ["right", "left", "down", "up", "start", "select", "B", "A"]

def make_env(): return gym_super_mario_bros.make("SuperMarioBros-1-1-v0").unwrapped
def read_xpos(env): return int(env.ram[RAM_X_PAGE]) * 256 + int(env.ram[RAM_X_PIXEL])
def read_lives(env): return int(np.int8(env.ram[RAM_LIVES]))

class RewardHead(nn.Module):
    def __init__(self, in_dim, hidden=128, n_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
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

def fm2_signature(action_block_40d):
    """Take a 40-d block (5 frames × 8 buttons), return the dominant
    button presses across the 5 frames."""
    a = action_block_40d.reshape(5, 8)  # 5 frames × 8 buttons
    sig = []
    for f in range(5):
        on = [FM2_NAMES[i] for i in range(8) if a[f, i] > 0.5]
        sig.append("+".join(on) if on else "·")
    return " ".join(sig)

@torch.no_grad()
def cem_plan_simple(model, head, history_emb, action_library, horizon, n_samples,
                     n_iters, elite_frac, device):
    history = history_emb.size(0)
    N = action_library.size(0)
    elite_n = max(1, int(round(n_samples * elite_frac)))
    logits = torch.zeros(horizon, N, device=device)
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
        new_logits = torch.zeros_like(logits)
        for s in range(horizon):
            counts = torch.bincount(idxs[elite_ids][:, s], minlength=N).float()
            new_logits[s] = (counts + 1e-3).log()
        logits = new_logits
    # Return best plan
    best_id = score.argmax().item()
    return candidate_actions[best_id], score[best_id].item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/phase3/joint_h8_best.pt")
    ap.add_argument("--out", default="/workspace/runs/diagnose_x898.json", type=Path)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n-samples-cem", type=int, default=256)
    ap.add_argument("--target-x", type=int, default=898)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = LeWorldModelConfig(**ck["config"])
    model = LeWorldModel(cfg); model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    head = RewardHead(in_dim=cfg.action_embed_dim, hidden=128, n_layers=2).to(device)
    head.load_state_dict(ck["head_state"])
    head.eval()
    action_library = ck["action_library"].to(device)
    rm, rs = float(ck["reward_mean"]), float(ck["reward_std"])
    history = cfg.history_size; image_size = cfg.image_size; block = 5
    print(f"loaded ckpt; reward_mean={rm:.2f} reward_std={rs:.2f}")
    print(f"action_library: shape={tuple(action_library.shape)}")

    # Stage 1: drive to x≈target_x using CEM with this model
    env = make_env(); env.reset()
    out = env.step(0); cur_frame = out[0].astype(np.uint8)
    init = np.stack([cur_frame] * history, axis=0)
    history_emb = encode(model, init, image_size, device)
    recent = [cur_frame] * history
    cur_block = 0
    while cur_block < 200:  # cap drive at 200 blocks
        plan, _ = cem_plan_simple(model, head, history_emb, action_library,
                                   args.horizon, args.n_samples_cem, 8, 0.1, device)
        macro = plan[0].view(block, 8).cpu().numpy()
        died = False
        for sub in range(block):
            o = env.step(fm2_row_to_nes_action(macro[sub]))
            obs = o[0]; done = o[2] if len(o) >= 3 else False
            if done or read_lives(env) < 0: died = True; break
        if died:
            print(f"died at block {cur_block} x={read_xpos(env)}")
            return
        recent.append(obs.astype(np.uint8)); recent = recent[-history:]
        history_emb = encode(model, np.stack(recent), image_size, device)
        cur_block += 1
        cx = read_xpos(env)
        if cx >= args.target_x - 5:
            print(f"reached x={cx} after {cur_block} blocks")
            break
    if read_xpos(env) < args.target_x - 30:
        print(f"only got to x={read_xpos(env)}, aborting diagnostic")
        return

    # Stage 2: at this state, score every single action library entry as a
    # 1-block plan (i.e. constant action repeated for h=8 blocks).
    print("\n=== stage 2: score every library entry as a constant 8-block plan ===")
    N = action_library.size(0)
    # build constant-plan tensor: [N, h, action_dim]
    const_plans = action_library.unsqueeze(1).expand(N, args.horizon, -1).contiguous()
    history_emb_b = history_emb.unsqueeze(0)
    with torch.no_grad():
        emb = history_emb_b.expand(N, -1, -1).clone()
        rewards_per_block = []
        for h in range(args.horizon):
            ctx_act = const_plans[:, max(0, h - history + 1): h + 1]
            if ctx_act.size(1) < history:
                pad = torch.zeros(N, history - ctx_act.size(1),
                                   action_library.size(1), device=device)
                ctx_act = torch.cat([pad, ctx_act], dim=1)
            ctx_act_emb = model.action_encoder(ctx_act)
            pred = model.predict(emb[:, -history:], ctx_act_emb)
            emb = torch.cat([emb, pred[:, -1:]], dim=1)
            rewards_per_block.append(head(pred[:, -1]).cpu().numpy())
        rewards_per_block = np.stack(rewards_per_block, axis=0)  # [h, N]
        total_rewards = rewards_per_block.sum(axis=0)  # [N]
    real_rewards = total_rewards * rs + rm * args.horizon  # un-normalize

    sorted_idx = np.argsort(-real_rewards)
    print(f"\nTop 10 single-action 8-block plans:")
    for rank, i in enumerate(sorted_idx[:10]):
        sig = fm2_signature(action_library[i].cpu().numpy())
        print(f"  #{rank+1}  predicted_reward={real_rewards[i]:+7.2f}  buttons={sig}")
    print(f"\nBottom 10:")
    for rank, i in enumerate(sorted_idx[-10:]):
        sig = fm2_signature(action_library[i].cpu().numpy())
        print(f"  #{N-9+rank}  predicted_reward={real_rewards[i]:+7.2f}  buttons={sig}")
    print(f"\nDistribution: min={real_rewards.min():+.2f}  median={np.median(real_rewards):+.2f}  max={real_rewards.max():+.2f}  std={real_rewards.std():.2f}")

    # Stage 3: full CEM at this state — what does it pick?
    print("\n=== stage 3: full CEM (n=256, iters=8) — what plan wins? ===")
    plan, score = cem_plan_simple(model, head, history_emb, action_library,
                                   args.horizon, args.n_samples_cem, 8, 0.1, device)
    real_score = score * rs + rm * args.horizon
    print(f"CEM picked plan with predicted total reward = {real_score:+.2f}")
    print(f"CEM plan first block buttons: {fm2_signature(plan[0].cpu().numpy())}")
    print(f"CEM plan all blocks:")
    for h in range(args.horizon):
        print(f"  h={h}  buttons={fm2_signature(plan[h].cpu().numpy())}")

    # save summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "ckpt": str(args.ckpt),
        "x_at_diagnostic": int(read_xpos(env)),
        "n_actions": int(N),
        "predicted_reward_dist": {
            "min": float(real_rewards.min()),
            "max": float(real_rewards.max()),
            "median": float(np.median(real_rewards)),
            "std": float(real_rewards.std()),
        },
        "top10_actions": [
            {"rank": i+1, "predicted_reward": float(real_rewards[sorted_idx[i]]),
             "buttons": fm2_signature(action_library[sorted_idx[i]].cpu().numpy())}
            for i in range(min(10, N))
        ],
        "cem_plan_predicted_reward": float(real_score),
        "cem_plan": [fm2_signature(plan[h].cpu().numpy()) for h in range(args.horizon)],
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {args.out}")

if __name__ == "__main__":
    main()
