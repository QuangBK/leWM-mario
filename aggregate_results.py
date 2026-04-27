"""Aggregate Phase 3 eval JSONs into a Markdown comparison table.

Walks /workspace/runs/eval/<label>/<label>_summary.json (per autonomous_eval_v2.py
output convention) and prints a single comparison table sorted by mean x_progress.

Usage:
  python3 aggregate_results.py /workspace/runs/eval > /workspace/runs/phase3_results.md
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import statistics as stats

def load_summary(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root", type=Path)
    args = ap.parse_args()

    rows = []
    for d in sorted(args.eval_root.iterdir()):
        if not d.is_dir(): continue
        cands = list(d.glob("*_summary.json"))
        if not cands: continue
        data = load_summary(cands[0])
        if not data: continue
        if not isinstance(data, list) or len(data) == 0: continue
        x_prog = [r["x_progress"] for r in data if "x_progress" in r]
        x_final = [r["x_final"] for r in data if "x_final" in r]
        deaths = sum(1 for r in data if r.get("final_lives", 2) < 2)
        if not x_prog: continue
        rows.append({
            "label": d.name,
            "n": len(x_prog),
            "mean_x": stats.mean(x_prog),
            "median_x": stats.median(x_prog),
            "max_x": max(x_prog),
            "min_x": min(x_prog),
            "max_final_x": max(x_final) if x_final else 0,
            "deaths": deaths,
        })

    rows.sort(key=lambda r: -r["mean_x"])

    print("# Phase 3 results comparison\n")
    print("| Variant | n | mean x_progress | median | max | max final_x | deaths |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['n']} | {r['mean_x']:.0f} | "
              f"{r['median_x']:.0f} | {r['max_x']} | {r['max_final_x']} | "
              f"{r['deaths']}/{r['n']} |")

    print()
    print("(Sorted by mean x_progress, descending. Deaths = episodes ending with `final_lives < 2`.)")

if __name__ == "__main__":
    main()
