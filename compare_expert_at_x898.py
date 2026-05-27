"""Compare what the TAS expert does at x∈[850,920] (the tall pipe) vs the
structure of our action library + the model's CEM-picked plan at that state.

We can run this CPU-only:
  - Replay one fast warpless TAS through nes_py to get (action[T,8], x_pos[T])
  - Slice the [850,920] window, print expert button trace per frame and per
    5-frame block (the unit our model plans in)
  - Load joint_best.pt's action_library [N,40] and answer:
      * for each expert block, which library entry is closest? how far?
      * what fraction of the library holds A across all 5 frames (long-jump
        primitive)? B+A across 5 frames (dash-jump primitive)?
      * does the library even *contain* a block close to the expert's
        pipe-jump signature?
  - If a model ckpt with CEM heads is available, also drive Mario to x≈895
    with that model and report what plan CEM picks at that state.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
from nes_py import NESEnv

sys.path.insert(0, "/root/leWM-project/lewm_mario")
from mario_lewm.fm2 import parse_fm2, fm2_row_to_nes_action, FM2_BUTTONS

GSMB_ROM = "/root/mario_rl_env/lib/python3.12/site-packages/gym_super_mario_bros/_roms/super-mario-bros.nes"

RAM_LIVES, RAM_X_PAGE, RAM_X_PIXEL = 0x075A, 0x6D, 0x86

def read_xpos(env): return int(env.ram[RAM_X_PAGE]) * 256 + int(env.ram[RAM_X_PIXEL])
def read_lives(env): return int(np.int8(env.ram[RAM_LIVES]))

def fm2_row_sig(row):
    """Return a compact button signature for a single 8-d controller frame."""
    on = [FM2_BUTTONS[i] for i in range(8) if row[i] > 0.5]
    return "+".join(on) if on else "·"

def fm2_block_sig(block_5x8):
    return " ".join(fm2_row_sig(block_5x8[f]) for f in range(5))

def replay_tas_record_actions(fm2_path: Path, max_frames: int, target_x: int):
    """Step through TAS until x crosses target_x + buffer; return aligned arrays."""
    rows = parse_fm2(fm2_path).astype(np.float32)[:max_frames]
    env = NESEnv(GSMB_ROM)
    env.reset()
    actions, xs, lives = [], [], []
    captured = False
    last_x = 0
    for t, row in enumerate(rows):
        out = env.step(fm2_row_to_nes_action(row))
        done = out[2] if len(out) == 4 else (out[2] or out[3])
        x = read_xpos(env); lv = read_lives(env)
        # only start recording once Mario has entered the level proper
        if not captured:
            if x >= 50 and lv >= 0:
                captured = True
            else:
                continue
        actions.append(row); xs.append(x); lives.append(lv)
        if lv < 0 or done:
            break
        if x > target_x + 200:
            break
        last_x = x
    env.close()
    return np.stack(actions), np.array(xs), np.array(lives)

def find_window(xs, lo, hi):
    mask = (xs >= lo) & (xs <= hi)
    if not mask.any(): return None
    return int(np.argmax(mask)), int(len(xs) - 1 - np.argmax(mask[::-1]))

def block_distance(expert_block_40d, library_40d):
    """Library is binary-ish (0/1). Expert is binary. Hamming distance."""
    e = (expert_block_40d > 0.5).astype(np.int32)
    L = (library_40d > 0.5).astype(np.int32)
    return np.abs(L - e[None, :]).sum(axis=1)  # [N]

def library_stats(lib_40d):
    """Counts of structural primitives in the library."""
    a = (lib_40d > 0.5).reshape(-1, 5, 8)  # [N, 5 frames, 8 buttons]
    R, L, D, U, T, S, B, A = 0, 1, 2, 3, 4, 5, 6, 7
    n = a.shape[0]
    hold_A_5    = (a[:, :, A].sum(axis=1) == 5).sum()
    hold_A_3p   = (a[:, :, A].sum(axis=1) >= 3).sum()
    hold_A_any  = (a[:, :, A].sum(axis=1) >= 1).sum()
    hold_BA_5   = ((a[:, :, A] & a[:, :, B]).sum(axis=1) == 5).sum()
    hold_BA_3p  = ((a[:, :, A] & a[:, :, B]).sum(axis=1) >= 3).sum()
    hold_RA_5   = ((a[:, :, A] & a[:, :, R]).sum(axis=1) == 5).sum()
    hold_RA_3p  = ((a[:, :, A] & a[:, :, R]).sum(axis=1) >= 3).sum()
    cruise      = (a[:, :, R].sum(axis=1) == 5).sum() - hold_RA_5
    only_right  = ((a[:, :, R].sum(axis=1) == 5) & (a.sum(axis=(1,2)) == 5)).sum()
    return {
        "n_total": int(n),
        "hold_A_full_5":      int(hold_A_5),
        "hold_A_3plus":       int(hold_A_3p),
        "hold_A_any":         int(hold_A_any),
        "hold_B+A_full_5":    int(hold_BA_5),
        "hold_B+A_3plus":     int(hold_BA_3p),
        "hold_R+A_full_5":    int(hold_RA_5),
        "hold_R+A_3plus":     int(hold_RA_3p),
        "right_held_5_no_A":  int(cruise),
        "right_only_pure":    int(only_right),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm2", default="/root/leWM-project/lewm_mario/traces/SMB_AISSON_warpless.fm2",
                    help="FM2 file to replay (warpless preferred for clean x progression)")
    ap.add_argument("--ckpt", default="/root/leWM-project/lewm_mario_artifacts/v2_joint/joint_best.pt",
                    help="Joint ckpt with .action_library (Phase 2 h=4 by default — Phase 3 h=8 ckpt is on the deleted pod)")
    ap.add_argument("--max-frames", type=int, default=8000)
    ap.add_argument("--x-lo", type=int, default=850)
    ap.add_argument("--x-hi", type=int, default=950)
    ap.add_argument("--out", default="/tmp/expert_vs_lib_x898.json", type=Path)
    args = ap.parse_args()

    print(f"=== expert TAS replay ===  fm2={Path(args.fm2).name}")
    actions, xs, lives = replay_tas_record_actions(Path(args.fm2), args.max_frames, args.x_hi)
    print(f"  recorded {len(actions)} frames; x reached {xs.max() if len(xs) else 0}, lives_min={int(lives.min()) if len(lives) else 'n/a'}")

    win = find_window(xs, args.x_lo, args.x_hi)
    if win is None:
        print(f"  TAS did not enter window x∈[{args.x_lo},{args.x_hi}] — try a different fm2")
        return
    i0, i1 = win
    print(f"  window x∈[{args.x_lo},{args.x_hi}] is frames [{i0}, {i1}] (n={i1-i0+1})")

    # Per-frame trace
    print("\n--- expert per-frame trace through pipe ---")
    print(f"  {'frame':>5}  {'x':>4}  buttons")
    for k in range(i0, min(i0 + 80, i1 + 1)):
        print(f"  {k:>5}  {xs[k]:>4}  {fm2_row_sig(actions[k])}")

    # Per-block (5-frame) signatures — this is the unit our model plans in
    print("\n--- expert per-block (5-frame) trace through pipe ---")
    block_lo = (i0 // 5) * 5
    block_hi = ((i1 // 5) + 1) * 5
    expert_blocks_in_window = []
    expert_block_xs = []
    for b0 in range(block_lo, block_hi, 5):
        if b0 + 5 > len(actions): break
        block = actions[b0:b0 + 5]  # [5, 8]
        block_x_start = int(xs[b0])
        block_x_end   = int(xs[b0 + 4])
        block_flat = block.reshape(40)
        sig = fm2_block_sig(block)
        print(f"  block@frame={b0:>5}  x={block_x_start:>4}->{block_x_end:>4}  [{sig}]")
        expert_blocks_in_window.append(block_flat.copy())
        expert_block_xs.append((block_x_start, block_x_end))

    # Action library structural analysis
    print(f"\n=== action library (from {Path(args.ckpt).name}) ===")
    import torch
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    lib = ck["action_library"].numpy() if hasattr(ck["action_library"], "numpy") else np.asarray(ck["action_library"])
    print(f"  shape={lib.shape}")
    stats = library_stats(lib)
    for k, v in stats.items():
        pct = 100 * v / stats["n_total"]
        print(f"  {k:<24} = {v:>5}  ({pct:5.1f}%)")

    # For each expert block in window, find nearest library entry
    closest_records = []
    n_exact = 0
    for (eblock, (x0, x1)) in zip(expert_blocks_in_window, expert_block_xs):
        d = block_distance(eblock, lib)
        j = int(np.argmin(d))
        sig_e = fm2_block_sig(eblock.reshape(5, 8))
        sig_l = fm2_block_sig(lib[j].reshape(5, 8))
        if int(d[j]) == 0: n_exact += 1
        closest_records.append({
            "x_start": x0, "x_end": x1, "expert": sig_e,
            "lib_nearest": sig_l, "hamming": int(d[j]),
        })
    print(f"\n--- expert→library mapping summary ---")
    print(f"  expert blocks in window: {len(closest_records)}")
    print(f"  exactly present in library:    {n_exact} / {len(closest_records)}  ({100*n_exact/max(1,len(closest_records)):.1f}%)")
    max_h = max(r["hamming"] for r in closest_records) if closest_records else 0
    print(f"  worst hamming distance:        {max_h} bits (out of 40)")

    # How many blocks does the expert spend AT x=898 (bouncing off pipe)?
    n_stalled = sum(1 for r in closest_records if r["x_start"] == r["x_end"] == 898)
    n_moving  = sum(1 for r in closest_records if r["x_end"] > r["x_start"])
    print(f"\n--- expert progress dissection in the window ---")
    print(f"  blocks stalled at x=898 (no progress): {n_stalled}")
    print(f"  blocks where x increased (motion):     {n_moving}")
    # Count categories of expert behavior in the window
    from collections import Counter
    cat = Counter()
    for r in closest_records:
        sig = r["expert"]
        if "+A" in sig and "R" in sig and r["x_start"] == 898 and r["x_end"] == 898:
            cat["run+jump @ x=898 (failed crossing)"] += 1
        elif "+A" in sig and "R" not in sig.split(" ")[0]:
            cat["jump-in-place @ x=898 (gain altitude)"] += 1
        elif "R" in sig and "+A" not in sig and r["x_start"] == r["x_end"] == 898:
            cat["run-into-wall @ x=898 (no jump)"] += 1
        elif r["x_end"] > r["x_start"]:
            cat["motion blocks (after clearing pipe)"] += 1
        else:
            cat["other"] += 1
    print(f"  category breakdown:")
    for k, v in cat.most_common():
        print(f"    {v:>3}  {k}")

    # Check whether the library has any "tall-jump" primitive: A held 4-5 frames in a row
    print("\n--- library entries that LOOK like long-jump primitives ---")
    a_arr = (lib > 0.5).reshape(-1, 5, 8)
    A_idx = 7  # FM2 order: R,L,D,U,T,S,B,A
    R_idx = 0
    long_jump_mask = (a_arr[:, :, A_idx].sum(axis=1) >= 4) & (a_arr[:, :, R_idx].sum(axis=1) >= 3)
    print(f"  candidates (A≥4 frames AND R≥3 frames): {long_jump_mask.sum()} / {len(lib)}")
    for j in np.where(long_jump_mask)[0][:10]:
        print(f"   #{j:>4} : {fm2_block_sig(lib[j].reshape(5, 8))}")

    summary = {
        "fm2": str(args.fm2),
        "ckpt": str(args.ckpt),
        "expert_window_x": [args.x_lo, args.x_hi],
        "expert_n_blocks_in_window": len(expert_blocks_in_window),
        "library_stats": stats,
        "expert_to_library_mapping": closest_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {args.out}")

if __name__ == "__main__":
    main()
