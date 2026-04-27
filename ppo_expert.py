"""Train a PPO expert on SuperMarioBros-1-1-v0 and capture rollouts in the same
npz schema as tas_replay so the existing precompute / training pipeline can
ingest them unchanged.

Two phases:
  1. `train` — train PPO with stable-baselines3 (CnnPolicy on stacked frames).
     Saves `ppo_mario.zip` to `--ckpt-out`.
  2. `rollout` — load PPO ckpt, run N episodes with deterministic + epsilon-greedy
     action sampling, capture frames + actions (FM2 8-d) + RAM (x_pos, lives,
     score, coins, ptype, timer) into one npz per episode under `--out-dir`.

Action mapping: PPO uses gym-super-mario-bros's `RIGHT_ONLY` / `SIMPLE_MOVEMENT`
button list, but tas_replay's npz schema stores raw 8-d FM2 rows
(R, L, D, U, T, S, B, A). We map each `nes_py.actions.Action` index back to
its raw button bitmap and pad to 8 dims.

Usage:
  ppo_expert.py train --steps 4000000 --ckpt-out /workspace/runs/ppo/ppo.zip
  ppo_expert.py rollout --ckpt /workspace/runs/ppo/ppo.zip \\
      --n-episodes 60 --out-dir /workspace/data/ppo_full
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np

# Map nes_py button names → FM2 row indices (R, L, D, U, T, S, B, A).
FM2_ORDER = ["right", "left", "down", "up", "start", "select", "B", "A"]
def buttons_to_fm2(button_list):
    """button_list is a list of strings like ['right', 'A']."""
    row = np.zeros(8, dtype=np.float32)
    for b in button_list:
        if b in FM2_ORDER:
            row[FM2_ORDER.index(b)] = 1.0
    return row

# SMB1 RAM addresses (mirror tas_replay.py exactly so the schema matches).
RAM_LIVES, RAM_X_PAGE, RAM_X_PIXEL = 0x075A, 0x6D, 0x86
RAM_PLAYER_STATE, RAM_WORLD, RAM_LEVEL, RAM_FLAG_GET = 0x000E, 0x075F, 0x075C, 0x001D
RAM_SCORE_LO, RAM_SCORE_HI = 0x07DD, 0x07E2
RAM_COINS, RAM_PLAYER_TYPE = 0x075E, 0x0756
RAM_TIMER_HUNDR, RAM_TIMER_TENS, RAM_TIMER_ONES = 0x07F8, 0x07F9, 0x07FA

def _read_score(ram):
    digits = [int(ram[a]) & 0x0F for a in range(RAM_SCORE_LO, RAM_SCORE_HI + 1)]
    val = 0
    for d in digits: val = val * 10 + d
    return val * 10
def _read_timer(ram):
    return ((int(ram[RAM_TIMER_HUNDR]) & 0x0F) * 100
          + (int(ram[RAM_TIMER_TENS])  & 0x0F) * 10
          + (int(ram[RAM_TIMER_ONES])  & 0x0F))
def read_state(env_unwrapped):
    ram = env_unwrapped.ram
    return {
        "x_pos":  int(ram[RAM_X_PAGE]) * 256 + int(ram[RAM_X_PIXEL]),
        "lives":  int(np.int8(ram[RAM_LIVES])),
        "score":  _read_score(ram),
        "coins":  ((int(ram[RAM_COINS]) >> 4) & 0x0F) * 10 + (int(ram[RAM_COINS]) & 0x0F),
        "ptype":  int(ram[RAM_PLAYER_TYPE]),
        "timer":  _read_timer(ram),
    }

def make_train_env(n_envs: int, seed: int = 0):
    """Vectorized training env: SuperMarioBros-1-1-v0 + SIMPLE_MOVEMENT + framestack."""
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
    from stable_baselines3.common.atari_wrappers import (
        WarpFrame, MaxAndSkipEnv, ClipRewardEnv, NoopResetEnv,
    )

    def _mk(rank):
        def _f():
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = MaxAndSkipEnv(env, skip=4)
            env = WarpFrame(env, width=84, height=84)
            env = ClipRewardEnv(env)
            env.seed(seed + rank)
            return env
        return _f

    venv = SubprocVecEnv([_mk(i) for i in range(n_envs)])
    venv = VecFrameStack(venv, n_stack=4)
    return venv

def cmd_train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    venv = make_train_env(args.n_envs, seed=42)
    model = PPO(
        "CnnPolicy", venv,
        learning_rate=2.5e-4, n_steps=512, batch_size=64,
        n_epochs=4, gamma=0.99, gae_lambda=0.95, clip_range=0.1,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        tensorboard_log=str(Path(args.ckpt_out).parent / "tb"),
        verbose=1, device="cuda",
    )
    cb = CheckpointCallback(save_freq=max(1, args.steps // 20),
                             save_path=str(Path(args.ckpt_out).parent / "checkpoints"),
                             name_prefix="ppo_mario")
    print(f"PPO train: steps={args.steps} n_envs={args.n_envs} → {args.ckpt_out}")
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    model.save(args.ckpt_out)
    venv.close()
    print(f"saved {args.ckpt_out}")

def cmd_rollout(args):
    """Run trained PPO and capture rollouts in the same npz schema as tas_replay."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
    from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(args.ckpt, device="cuda")

    summary = []
    for ep in range(args.n_episodes):
        # raw env so we can capture the FULL 240×256×3 RGB frame for the dataset,
        # plus a parallel preprocessed env (84×84 stack) for the policy input.
        raw_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        raw_env_js = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
        # build the policy-input env separately by applying same wrappers
        pol_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        pol_env_js = JoypadSpace(pol_env, SIMPLE_MOVEMENT)
        pol_env_js = MaxAndSkipEnv(pol_env_js, skip=4)
        pol_env_js = WarpFrame(pol_env_js, width=84, height=84)
        venv = VecFrameStack(DummyVecEnv([lambda: pol_env_js]), n_stack=4)

        raw_obs = raw_env_js.reset()
        pol_obs = venv.reset()
        # nes_py.unwrap is the raw NESEnv exposing .ram
        nesenv = raw_env_js.env  # JoypadSpace → SuperMarioBrosEnv

        frames = [raw_obs.copy().astype(np.uint8)]
        actions, x_series, life_series, score_series = [], [], [], []
        coin_series, ptype_series, timer_series = [], [], []
        epsilon = float(args.epsilon)
        rng = np.random.default_rng(1000 + ep)
        died, max_x = False, 0
        for t in range(args.max_frames):
            if rng.random() < epsilon:
                act_idx = rng.integers(0, len(SIMPLE_MOVEMENT))
            else:
                act_idx, _ = model.predict(pol_obs, deterministic=True)
                act_idx = int(act_idx[0]) if hasattr(act_idx, "__len__") else int(act_idx)
            buttons = SIMPLE_MOVEMENT[int(act_idx)]
            row = buttons_to_fm2(buttons)
            # MaxAndSkipEnv inside venv applies frame_skip=4, but raw_env is unskipped.
            # To keep alignment with tas_replay's frame_skip=5 dataset block, we step
            # the raw env 4 times per policy decision (matches the policy's view).
            for _sub in range(4):
                raw_obs, _r, raw_done, raw_info = raw_env_js.step(int(act_idx))
                if raw_done: break
            pol_obs, _pr, pol_done, _pinfo = venv.step([int(act_idx)])
            st = read_state(nesenv)
            frames.append(raw_obs.copy().astype(np.uint8))
            actions.append(row)
            x_series.append(st["x_pos"]); life_series.append(st["lives"])
            score_series.append(st["score"]); coin_series.append(st["coins"])
            ptype_series.append(st["ptype"]); timer_series.append(st["timer"])
            max_x = max(max_x, st["x_pos"])
            if raw_done or st["lives"] < 0:
                died = True; break
        venv.close(); raw_env_js.close()

        if len(actions) < args.min_frames:
            print(json.dumps({"ep": ep, "ok": False, "frames": len(actions), "max_x": max_x}), flush=True)
            continue

        meta = {
            "captured_frames": len(frames), "capture_initial_frame": True,
            "source": "ppo", "max_x_pos": int(max_x), "final_x_pos": int(x_series[-1]),
            "final_lives": int(life_series[-1]), "final_score": int(score_series[-1]),
            "final_coins": int(coin_series[-1]), "captured_action_frames": len(actions),
            "epsilon": epsilon, "ckpt": args.ckpt,
        }
        out_name = f"ppo_{ep:03d}.npz"
        np.savez_compressed(
            out_dir / out_name,
            frames=np.stack(frames, 0).astype(np.uint8),
            actions=np.stack(actions, 0).astype(np.float32),
            x_pos=np.asarray(x_series, dtype=np.int32),
            lives=np.asarray(life_series, dtype=np.int8),
            score=np.asarray(score_series, dtype=np.int32),
            coins=np.asarray(coin_series, dtype=np.int16),
            ptype=np.asarray(ptype_series, dtype=np.int8),
            timer=np.asarray(timer_series, dtype=np.int16),
            metadata_json=json.dumps(meta),
        )
        rec = {"ep": ep, "ok": True, "frames": len(frames), "max_x": int(max_x), **meta}
        print(json.dumps(rec), flush=True)
        summary.append(rec)
    (out_dir / "ppo_rollout_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE rollouts={len(summary)} → {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=4_000_000)
    tr.add_argument("--n-envs", type=int, default=8)
    tr.add_argument("--ckpt-out", default="/workspace/runs/ppo/ppo_mario.zip")
    ro = sub.add_parser("rollout")
    ro.add_argument("--ckpt", required=True)
    ro.add_argument("--n-episodes", type=int, default=60)
    ro.add_argument("--max-frames", type=int, default=2400)
    ro.add_argument("--min-frames", type=int, default=200)
    ro.add_argument("--epsilon", type=float, default=0.05)
    ro.add_argument("--out-dir", default="/workspace/data/ppo_full")
    args = ap.parse_args()
    {"train": cmd_train, "rollout": cmd_rollout}[args.cmd](args)

if __name__ == "__main__":
    main()
