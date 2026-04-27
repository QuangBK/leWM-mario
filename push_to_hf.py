"""Push intermediate Phase 3 ckpts and result JSONs to obamaTeo/lewm-mario.

Usage:
  python3 push_to_hf.py file:/workspace/runs/joint_v2_score/best.pt:joint_v2_score.pt
  python3 push_to_hf.py dir:/workspace/runs/eval/joint_v2_score:eval/joint_v2_score
  python3 push_to_hf.py file:/workspace/runs/vit_small/best.pt:vit_small_best.pt

Each spec is "file:<local>:<remote>" or "dir:<local>:<remote>". Multiple
specs allowed.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO = "obamaTeo/lewm-mario"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+",
                    help="file:<local>:<remote> or dir:<local>:<remote>")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var missing", file=sys.stderr); sys.exit(1)
    create_repo(repo_id=args.repo, token=token, private=True, exist_ok=True)
    api = HfApi(token=token)

    for spec in args.specs:
        kind, local, remote = spec.split(":", 2)
        local = Path(local)
        if not local.exists():
            print(f"SKIP missing: {local}")
            continue
        if kind == "file":
            print(f"upload file {local} → {args.repo}/{remote}")
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=args.repo, repo_type="model")
        elif kind == "dir":
            print(f"upload dir {local} → {args.repo}/{remote}")
            api.upload_folder(folder_path=str(local), path_in_repo=remote,
                               repo_id=args.repo, repo_type="model")
        else:
            print(f"unknown spec kind: {kind}", file=sys.stderr); continue
    print("done")

if __name__ == "__main__":
    main()
