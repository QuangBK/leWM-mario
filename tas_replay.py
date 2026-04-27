"""Replay TAS .fm2 files through raw nes_py against the gym_super_mario_bros
ROM, capturing (frame, action) pairs. Uses NESEnv directly so we can replay
from cold boot (title screen → menu → gameplay) with the full 8-bit controller
byte from each FM2 row. Reads x_pos / life directly from SMB1 RAM so we can
detect desync (Mario dies before TAS expects, or stops progressing).

Saves .npz episodes in the schema lewm_mario expects:
  frames: (T+1, 240, 256, 3) uint8
  actions: (T, 8) float32 (FM2 button order R, L, D, U, T, S, B, A)
  metadata_json with capture_initial_frame=True, captured_frames=T+1.
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from nes_py import NESEnv

sys.path.insert(0, "/root/lewm_mario")
from mario_lewm.fm2 import parse_fm2, fm2_row_to_nes_action  # noqa: E402

GSMB_ROM = "/opt/conda/lib/python3.11/site-packages/gym_super_mario_bros/_roms/super-mario-bros.nes"

# SMB1 RAM addresses (well-documented, see datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map)
RAM_LIVES        = 0x075A   # lives left (signed; 0xFF = -1 = game over)
RAM_X_PAGE       = 0x6D     # current screen page
RAM_X_PIXEL      = 0x86     # x within page  (true x = page*256 + pixel)
RAM_PLAYER_STATE = 0x000E   # 0x06 = dying, 0x0B = dying-fall, 0x08 = climbing
RAM_WORLD        = 0x075F   # world index (0-based)
RAM_LEVEL        = 0x075C   # level index (0-based)
RAM_FLAG_GET     = 0x001D   # nonzero when sliding flag
# Score is 6 BCD digits at 0x07DD-0x07E2 (each byte 0-9). Multiply each by power
# of 10 then by 10 (in-game score is the BCD value × 10). Top digits at 0x07DD.
RAM_SCORE_LO     = 0x07DD
RAM_SCORE_HI     = 0x07E2  # inclusive
RAM_COINS        = 0x075E   # BCD, 0-99
RAM_PLAYER_TYPE  = 0x0756   # 0=small, 1=big, 2=fire
RAM_TIMER_HUNDR  = 0x07F8   # 100s digit BCD
RAM_TIMER_TENS   = 0x07F9
RAM_TIMER_ONES   = 0x07FA

def _read_score(ram):
    # 6 BCD digits, 0x07DD = highest, 0x07E2 = lowest. Game score = BCD × 10.
    digits = [int(ram[a]) & 0x0F for a in range(RAM_SCORE_LO, RAM_SCORE_HI + 1)]
    val = 0
    for d in digits:
        val = val * 10 + d
    return val * 10

def _read_timer(ram):
    return (int(ram[RAM_TIMER_HUNDR]) & 0x0F) * 100 \
         + (int(ram[RAM_TIMER_TENS])  & 0x0F) * 10  \
         + (int(ram[RAM_TIMER_ONES])  & 0x0F)

def read_state(env):
    ram = env.ram
    return {
        "x_pos":   int(ram[RAM_X_PAGE]) * 256 + int(ram[RAM_X_PIXEL]),
        "lives":   int(np.int8(ram[RAM_LIVES])),
        "pstate":  int(ram[RAM_PLAYER_STATE]),
        "world":   int(ram[RAM_WORLD]) + 1,
        "level":   int(ram[RAM_LEVEL]) + 1,
        "flag":    bool(ram[RAM_FLAG_GET]),
        "score":   _read_score(ram),
        "coins":   ((int(ram[RAM_COINS]) >> 4) & 0x0F) * 10 + (int(ram[RAM_COINS]) & 0x0F),
        "ptype":   int(ram[RAM_PLAYER_TYPE]),
        "timer":   _read_timer(ram),
    }

def replay_one(fm2_path: Path, max_frames: int | None = None,
               stuck_window: int = 240, min_capture_x: int = 50):
    actions_2d = parse_fm2(fm2_path).astype(np.float32)
    if max_frames:
        actions_2d = actions_2d[:max_frames]

    env = NESEnv(GSMB_ROM)
    obs = env.reset()

    # capture initial frame
    frames = [obs.copy().astype(np.uint8)]
    actions_used = []
    x_pos_series = []
    lives_series = []

    last_seen_x = 0
    stuck = 0
    desync = None
    start_lives = None
    captured_started = False
    score_series = []
    coins_series = []
    ptype_series = []
    timer_series = []

    for t, row in enumerate(actions_2d):
        byte_action = fm2_row_to_nes_action(row)
        out = env.step(byte_action)
        # NESEnv.step returns 4-tuple in old gym API
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            done = term or trunc
        else:
            obs, reward, done, info = out

        st = read_state(env)
        if start_lives is None:
            start_lives = st["lives"]

        # Trim leading menu frames: only start saving once Mario has entered
        # the level (x_pos crosses min_capture_x for the first time).
        if not captured_started:
            if st["x_pos"] >= min_capture_x and st["lives"] >= 0:
                captured_started = True
                # rewrite the initial-frame slot to be the current frame so
                # frames[0] corresponds to the first captured action's *prior*
                # observation (paper convention frames=actions+1).
                frames = [obs.copy().astype(np.uint8)]
            else:
                # still in menu / pre-game; keep ticking but don't save
                continue

        frames.append(obs.copy().astype(np.uint8))
        actions_used.append(row)
        x_pos_series.append(st["x_pos"])
        lives_series.append(st["lives"])
        score_series.append(st["score"])
        coins_series.append(st["coins"])
        ptype_series.append(st["ptype"])
        timer_series.append(st["timer"])

        # Desync detection
        if st["lives"] < (start_lives or 0):
            desync = f"died_at_t={t}_x={st['x_pos']}"
            break
        if done:
            desync = f"env_done_at_t={t}_x={st['x_pos']}"
            break
        if st["x_pos"] == last_seen_x:
            stuck += 1
            if stuck > stuck_window:
                desync = f"stuck_at_x={st['x_pos']}_t={t}_for_{stuck_window}_frames"
                break
        else:
            stuck = 0
            last_seen_x = st["x_pos"]

    env.close()

    # If we never entered the level, return None (TAS could not be aligned)
    if not captured_started or len(actions_used) == 0:
        return None

    return {
        "frames":  np.stack(frames, axis=0).astype(np.uint8),
        "actions": np.stack(actions_used, axis=0).astype(np.float32),
        "x_pos":   np.asarray(x_pos_series, dtype=np.int32),
        "lives":   np.asarray(lives_series, dtype=np.int8),
        "score":   np.asarray(score_series, dtype=np.int32),
        "coins":   np.asarray(coins_series, dtype=np.int16),
        "ptype":   np.asarray(ptype_series, dtype=np.int8),
        "timer":   np.asarray(timer_series, dtype=np.int16),
        "desync_reason": desync,
        "fm2_total": int(len(actions_2d)),
        "start_lives": int(start_lives),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default="/root/lewm_mario/traces")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=20000)
    ap.add_argument("--stuck-window", type=int, default=240)
    ap.add_argument("--include", nargs="+", default=None,
                    help="Only replay FM2 filenames containing any of these substrings")
    ap.add_argument("--min-frames", type=int, default=200,
                    help="Reject TASes producing fewer than this many in-level frames")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fm2_paths = sorted(Path(args.traces_dir).glob("*.fm2"))
    if args.include:
        fm2_paths = [p for p in fm2_paths if any(s.lower() in p.name.lower() for s in args.include)]

    print(f"replaying {len(fm2_paths)} TAS files", flush=True)
    summary = []
    t_total = time.perf_counter()
    for i, fm2 in enumerate(fm2_paths):
        t0 = time.perf_counter()
        try:
            out = replay_one(fm2, max_frames=args.max_frames, stuck_window=args.stuck_window)
        except Exception as e:
            rec = {"name": fm2.stem, "ok": False, "error": repr(e)}
            print(json.dumps(rec), flush=True)
            summary.append(rec)
            continue
        if out is None:
            rec = {"name": fm2.stem, "ok": False, "error": "never_entered_level"}
            print(json.dumps(rec), flush=True)
            summary.append(rec)
            continue
        n_frames = int(out["frames"].shape[0])
        n_actions = int(out["actions"].shape[0])
        if n_actions < args.min_frames:
            rec = {"name": fm2.stem, "ok": False, "error": f"too_short_{n_actions}",
                   "max_x": int(out["x_pos"].max()) if n_actions else 0,
                   "desync_reason": out["desync_reason"]}
            print(json.dumps(rec), flush=True)
            summary.append(rec)
            continue

        meta = {
            "captured_frames": n_frames,
            "capture_initial_frame": True,
            "source_fm2": fm2.name,
            "fm2_total_frames": out["fm2_total"],
            "captured_action_frames": n_actions,
            "max_x_pos": int(out["x_pos"].max()),
            "final_x_pos": int(out["x_pos"][-1]),
            "min_lives": int(out["lives"].min()),
            "start_lives": out["start_lives"],
            "desync_reason": out["desync_reason"],
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        out_name = f"tas_{i:03d}_{fm2.stem.replace(' ', '_').replace(',', '')[:60]}.npz"
        np.savez_compressed(
            args.output_dir / out_name,
            frames=out["frames"],
            actions=out["actions"],
            x_pos=out["x_pos"],
            lives=out["lives"],
            score=out["score"],
            coins=out["coins"],
            ptype=out["ptype"],
            timer=out["timer"],
            metadata_json=json.dumps(meta),
        )
        rec = {"name": fm2.stem, "ok": True, "out": out_name, **meta}
        print(json.dumps(rec), flush=True)
        summary.append(rec)

    (args.output_dir / "tas_replay_summary.json").write_text(json.dumps(summary, indent=2))
    ok = sum(1 for r in summary if r.get("ok"))
    total = len(summary)
    total_frames = sum(r.get("captured_action_frames", 0) for r in summary if r.get("ok"))
    print(f"DONE ok={ok}/{total} total_action_frames={total_frames} elapsed_s={time.perf_counter()-t_total:.1f}", flush=True)

if __name__ == "__main__":
    main()
