"""Roll out MadMario's pretrained DDQN on SuperMarioBros-1-1-v0 and save
.npz episodes in the schema lewm_mario expects (frames=actions+1, FM2 8-button
action vectors, capture_initial_frame=True).
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from skimage import transform as sk_transform

sys.path.insert(0, "/root/MadMario")
from neural import MarioNet  # noqa: E402

# FM2 button order: R, L, D, U, T(=Start), S(=Select), B, A
ACTION_RIGHT      = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
ACTION_RIGHT_A    = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
MADMARIO_TO_FM2   = np.stack([ACTION_RIGHT, ACTION_RIGHT_A])

def preprocess(rgb: np.ndarray) -> np.ndarray:
    # MadMario wrappers: RGB -> grayscale (0.2125R + 0.7154G + 0.0721B per skimage),
    # resize to 84x84, scale to [0, 1].
    gray = np.dot(rgb[..., :3].astype(np.float32), [0.2125, 0.7154, 0.0721]) / 255.0
    return sk_transform.resize(gray, (84, 84), anti_aliasing=False).astype(np.float32)

def build_net(ckpt_path: str, device: torch.device) -> MarioNet:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = MarioNet((4, 84, 84), 2).float()
    net.load_state_dict(ckpt["model"])
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net

def rollout_episode(env, net, device, epsilon, max_steps, decision_skip=4, rng=None):
    rng = rng or np.random.default_rng()
    obs = env.reset()
    frame_buf = deque([preprocess(obs)] * 4, maxlen=4)
    frames = [obs.copy().astype(np.uint8)]  # initial frame at t=0
    actions = []
    rewards = []
    x_pos_series = []
    current_action_idx = 0
    info_last = {}
    for t in range(max_steps):
        if t % decision_skip == 0:
            if rng.random() < epsilon:
                current_action_idx = int(rng.integers(0, 2))
            else:
                state = np.stack(list(frame_buf), axis=0)[None]  # (1,4,84,84)
                with torch.no_grad():
                    q = net(torch.from_numpy(state).to(device), model="online")
                current_action_idx = int(q.argmax(dim=1).item())
        out = env.step(current_action_idx)
        obs, reward, done = out[0], float(out[1]), bool(out[2])
        info_last = out[-1]
        frames.append(obs.astype(np.uint8))
        actions.append(MADMARIO_TO_FM2[current_action_idx])
        rewards.append(reward)
        x_pos_series.append(int(info_last.get("x_pos", 0)))
        frame_buf.append(preprocess(obs))
        if done or info_last.get("flag_get", False):
            break
    return {
        "frames": np.stack(frames, axis=0).astype(np.uint8),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "x_pos": np.asarray(x_pos_series, dtype=np.int32),
        "info_final": info_last,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/root/MadMario/trained_mario.chkpt")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--num-episodes", type=int, default=10)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--decision-skip", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-index", type=int, default=0,
                    help="filename prefix offset for resuming/parallel runs")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_net(args.checkpoint, device)

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    rng = np.random.default_rng(args.seed)
    summary = []
    t0 = time.perf_counter()
    for i in range(args.num_episodes):
        ep_out = rollout_episode(env, net, device, args.epsilon, args.max_steps,
                                 args.decision_skip, rng)
        T = len(ep_out["actions"])
        meta = {
            "captured_frames": int(ep_out["frames"].shape[0]),
            "capture_initial_frame": True,
            "episode_idx": args.start_index + i,
            "epsilon": args.epsilon,
            "decision_skip": args.decision_skip,
            "max_steps": args.max_steps,
            "max_x_pos": int(ep_out["x_pos"].max() if T else 0),
            "final_x_pos": int(ep_out["x_pos"][-1] if T else 0),
            "final_score": int(ep_out["info_final"].get("score", 0)),
            "died": bool(ep_out["info_final"].get("life", 2) < 2),
            "got_flag": bool(ep_out["info_final"].get("flag_get", False)),
            "total_reward": float(ep_out["rewards"].sum()),
            "world": int(ep_out["info_final"].get("world", 1)),
            "stage": int(ep_out["info_final"].get("stage", 1)),
        }
        out_path = args.output_dir / f"madmario_ep_{args.start_index + i:05d}.npz"
        np.savez_compressed(
            out_path,
            frames=ep_out["frames"],
            actions=ep_out["actions"],
            rewards=ep_out["rewards"],
            x_pos=ep_out["x_pos"],
            metadata_json=json.dumps(meta),
        )
        elapsed = time.perf_counter() - t0
        print(json.dumps({"i": i, "T": T, **{k: v for k, v in meta.items()
              if k in ("max_x_pos", "final_x_pos", "got_flag", "died")},
              "elapsed_s": round(elapsed, 1)}), flush=True)
        summary.append({"path": str(out_path), **meta})

    (args.output_dir / "rollout_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done episodes={args.num_episodes} elapsed_s={time.perf_counter()-t0:.1f}")

if __name__ == "__main__":
    main()
