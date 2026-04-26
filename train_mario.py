from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, random_split
from torch.utils.tensorboard import SummaryWriter
import wandb

from mario_lewm.dataset import MarioTraceDataset, discover_episodes
from mario_lewm.fm2 import build_action_library
from mario_lewm.model import LeWorldModel, LeWorldModelConfig


class EpisodeBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        subset_indices: list[int],
        dataset_index: list[tuple[int, int]],
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        self.subset_indices = list(subset_indices)
        self.dataset_index = dataset_index
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._positions_by_episode: dict[int, list[int]] = defaultdict(list)
        for local_pos, global_idx in enumerate(self.subset_indices):
            episode_id, _ = self.dataset_index[global_idx]
            self._positions_by_episode[episode_id].append(local_pos)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        episode_ids = list(self._positions_by_episode.keys())
        if self.shuffle:
            rng.shuffle(episode_ids)
        batches: list[list[int]] = []
        for episode_id in episode_ids:
            positions = list(self._positions_by_episode[episode_id])
            if self.shuffle:
                rng.shuffle(positions)
            for start in range(0, len(positions), self.batch_size):
                batch = positions[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for positions in self._positions_by_episode.values():
            if self.drop_last:
                total += len(positions) // self.batch_size
            else:
                total += math.ceil(len(positions) / self.batch_size)
        return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LeWM-style world model on Super Mario FM2 traces.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--precomputed-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--encoder-hidden-dim", type=int, default=192)
    parser.add_argument("--encoder-depth", type=int, default=12)
    parser.add_argument("--encoder-heads", type=int, default=3)
    parser.add_argument("--encoder-mlp-dim", type=int, default=768)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--predictor-heads", type=int, default=16)
    parser.add_argument("--predictor-hidden-dim", type=int, default=192)
    parser.add_argument("--predictor-output-dim", type=int, default=192)
    parser.add_argument("--predictor-mlp-dim", type=int, default=2048)
    parser.add_argument("--action-embed-dim", type=int, default=192)
    parser.add_argument("--action-smoothed-dim", type=int, default=32)
    parser.add_argument("--projector-hidden-dim", type=int, default=2048)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--npz-load-mode", choices=["lazy", "preload"], default="lazy")
    parser.add_argument("--max-cached-episodes", type=int, default=4)
    parser.add_argument("--batching", choices=["episode", "random"], default="episode")
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=20, help="Save numbered epoch checkpoints every N epochs. 0 keeps only best and latest.")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(args: argparse.Namespace, action_dim: int) -> LeWorldModelConfig:
    return LeWorldModelConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        encoder_mlp_dim=args.encoder_mlp_dim,
        action_dim=action_dim,
        action_embed_dim=args.action_embed_dim,
        action_smoothed_dim=args.action_smoothed_dim,
        history_size=args.history_size,
        num_preds=args.num_preds,
        predictor_hidden_dim=args.predictor_hidden_dim,
        predictor_output_dim=args.predictor_output_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        predictor_mlp_dim=args.predictor_mlp_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
    )


def get_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def action_sequences_for_library(episodes) -> list[np.ndarray]:
    sequences = []
    for episode in episodes:
        metadata = episode.metadata or {}
        if metadata.get("lewm_terminal_pad_action"):
            num_blocks = int(metadata.get("lewm_num_action_blocks", max(0, len(episode.actions) - 1)))
            sequences.append(np.asarray(episode.actions[:num_blocks], dtype=np.float32))
        else:
            sequences.append(np.asarray(episode.actions, dtype=np.float32))
    return sequences


def evaluate(model: LeWorldModel, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype) -> dict[str, float]:
    model.eval()
    total = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
    count = 0
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixels"].to(device, non_blocking=True)
            actions = batch["action"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                output = model.compute_losses({"pixels": pixels, "action": actions})
            batch_size = pixels.size(0)
            count += batch_size
            for key in total:
                total[key] += float(output[key].detach().cpu()) * batch_size
    return {key: value / max(1, count) for key, value in total.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    effective_dataset_root = args.precomputed_root if args.precomputed_root is not None else args.dataset_root
    episodes = discover_episodes(effective_dataset_root)
    if any((episode.metadata or {}).get("capture_initial_frame") and not (episode.metadata or {}).get("lewm_action_block_size") for episode in episodes):
        raise ValueError(
            "This dataset contains initial-frame raw exports (frames = actions + 1). "
            "Build a blocked LeWM dataset first with build_lewm_mario_dataset.py, then train on that dataset."
        )
    action_dim = int(episodes[0].actions.shape[1])
    npz_episodes = sum(1 for episode in episodes if episode.frames_npz is not None)
    image_episodes = len(episodes) - npz_episodes
    total_frames = sum(min(len(episode.actions), int(episode.num_frames or len(episode.actions))) for episode in episodes)
    print(
        f"episodes_total={len(episodes)} npz_episodes={npz_episodes} "
        f"image_episodes={image_episodes} total_usable_frames={total_frames}"
    )
    print(
        f"dataset_root={effective_dataset_root} raw_dataset_root={args.dataset_root} "
        f"batch_size={args.batch_size} num_workers={args.num_workers} "
        f"npz_load_mode={args.npz_load_mode} max_cached_episodes={args.max_cached_episodes} action_dim={action_dim}"
    )
    if args.num_workers > 0 and args.npz_load_mode == "lazy":
        print(
            "warning=on Windows each DataLoader worker keeps its own episode cache; "
            "if RAM spikes, try --num-workers 0 or --max-cached-episodes 1"
        )
    full_dataset = MarioTraceDataset(
        episodes,
        args.history_size,
        args.num_preds,
        args.image_size,
        args.stride,
        npz_load_mode=args.npz_load_mode,
        max_cached_episodes=args.max_cached_episodes,
    )
    val_size = max(1, int(round(len(full_dataset) * args.val_fraction)))
    val_size = min(val_size, len(full_dataset) - 1) if len(full_dataset) > 1 else 1
    train_size = max(1, len(full_dataset) - val_size)
    if train_size + val_size > len(full_dataset):
        val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(
        f"dataset_windows={len(full_dataset)} train_windows={len(train_dataset)} "
        f"val_windows={len(val_dataset)} npz_load_mode={args.npz_load_mode} "
        f"max_cached_episodes={args.max_cached_episodes} batching={args.batching}"
    )
    common_loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader_kwargs = dict(common_loader_kwargs)
    val_loader_kwargs = dict(common_loader_kwargs)
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
    if args.batching == "episode":
        train_batch_sampler = EpisodeBatchSampler(
            subset_indices=list(train_dataset.indices),
            dataset_index=full_dataset.index,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        val_batch_sampler = EpisodeBatchSampler(
            subset_indices=list(val_dataset.indices),
            dataset_index=full_dataset.index,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            seed=args.seed,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **train_loader_kwargs)
        val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **val_loader_kwargs)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **train_loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **val_loader_kwargs,
        )
    print("dataloaders_ready=true")

    resume_path = None
    if args.resume is not None:
        resume_path = args.resume
    elif args.resume_latest:
        candidate = args.output_dir / "latest.pt"
        if candidate.exists():
            resume_path = candidate
        else:
            raise FileNotFoundError(f"--resume-latest was set, but no checkpoint exists at {candidate}")

    resume_checkpoint = None
    if resume_path is not None:
        print(f"resume_checkpoint={resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        config = LeWorldModelConfig(**resume_checkpoint["config"])
    else:
        config = build_config(args, action_dim=action_dim)

    base_model = LeWorldModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    if resume_checkpoint is not None:
        base_model.load_state_dict(resume_checkpoint["model_state"])
    model = base_model
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    param_count = sum(p.numel() for p in base_model.parameters())
    print(f"device={device} precision={args.precision} compile={args.compile} params={param_count:,}")
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * math.ceil(len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.precision == "fp16")
    amp_dtype = get_dtype(args.precision)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    start_epoch = 1
    best_val = float("inf")
    global_step = 0
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        if "scaler_state" in resume_checkpoint and scaler.is_enabled():
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val = float(
            resume_checkpoint.get("best_val", resume_checkpoint.get("val_metrics", {}).get("loss", float("inf")))
        )
        global_step = int(resume_checkpoint.get("global_step", 0))
        print(f"resumed_state epoch={start_epoch - 1} global_step={global_step} best_val={best_val}")

    action_library = build_action_library(action_sequences_for_library(episodes))
    metadata = {
        "config": config.to_dict(),
        "episodes": [episode.name for episode in episodes],
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "action_dim": action_dim,
        "action_library": action_library.tolist(),
        "buttons": ["R", "L", "D", "U", "T", "S", "B", "A"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"))
    wandb_run = None
    if args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            dir=str(args.output_dir),
        )
    def _log(metrics, step=None):
        for k, v in metrics.items():
            writer.add_scalar(k, v, step if step is not None else 0)
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            if args.batching == "episode":
                train_loader.batch_sampler.set_epoch(epoch)
            running = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
            seen = 0
            epoch_start = time.perf_counter()
            for step_in_epoch, batch in enumerate(train_loader, start=1):
                if global_step == 0 and step_in_epoch == 1:
                    print(
                        f"first_batch_ready=true epoch={epoch} step_in_epoch={step_in_epoch} "
                        f"batch_pixels_shape={tuple(batch['pixels'].shape)}"
                    )
                pixels = batch["pixels"].to(device, non_blocking=True)
                actions = batch["action"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    output = model.compute_losses({"pixels": pixels, "action": actions})
                if scaler.is_enabled():
                    scaler.scale(output["loss"]).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output["loss"].backward()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
                    optimizer.step()
                scheduler.step()

                batch_size = pixels.size(0)
                seen += batch_size
                for key in running:
                    running[key] += float(output[key].detach().cpu()) * batch_size

                writer.add_scalar("train_step/loss", float(output["loss"].detach().cpu()), global_step)
                writer.add_scalar("train_step/pred_loss", float(output["pred_loss"].detach().cpu()), global_step)
                writer.add_scalar("train_step/sigreg_loss", float(output["sigreg_loss"].detach().cpu()), global_step)
                writer.add_scalar("train_step/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train_step/grad_norm", grad_norm, global_step)
                if wandb_run is not None:
                    wandb_run.log({
                        "train_step/loss": float(output["loss"].detach().cpu()),
                        "train_step/pred_loss": float(output["pred_loss"].detach().cpu()),
                        "train_step/sigreg_loss": float(output["sigreg_loss"].detach().cpu()),
                        "train_step/lr": scheduler.get_last_lr()[0],
                        "train_step/grad_norm": grad_norm,
                    }, step=global_step)
                if global_step == 0 and step_in_epoch == 1:
                    print(
                        f"first_optimizer_step=true epoch={epoch} step_in_epoch={step_in_epoch} "
                        f"loss={float(output['loss'].detach().cpu()):.6f}"
                    )
                if args.log_every_steps > 0 and step_in_epoch % args.log_every_steps == 0:
                    elapsed = max(1e-6, time.perf_counter() - epoch_start)
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step_in_epoch": step_in_epoch,
                                "global_step": global_step,
                                "loss": float(output["loss"].detach().cpu()),
                                "pred_loss": float(output["pred_loss"].detach().cpu()),
                                "sigreg_loss": float(output["sigreg_loss"].detach().cpu()),
                                "lr": scheduler.get_last_lr()[0],
                                "seen_samples": seen,
                                "samples_per_sec": seen / elapsed,
                            }
                        )
                    )
                global_step += 1

            train_metrics = {key: value / max(1, seen) for key, value in running.items()}
            val_metrics = evaluate(model, val_loader, device, amp_dtype)
            epoch_elapsed = max(1e-6, time.perf_counter() - epoch_start)
            for key, value in train_metrics.items():
                writer.add_scalar(f"train_epoch/{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val_epoch/{key}", value, epoch)
            writer.add_scalar("train_epoch/lr", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("train_epoch/samples_per_sec", seen / epoch_elapsed, epoch)
            if wandb_run is not None:
                payload = {f"train_epoch/{k}": v for k, v in train_metrics.items()}
                payload.update({f"val_epoch/{k}": v for k, v in val_metrics.items()})
                payload["train_epoch/lr"] = scheduler.get_last_lr()[0]
                payload["train_epoch/samples_per_sec"] = seen / epoch_elapsed
                payload["epoch"] = epoch
                wandb_run.log(payload, step=global_step)
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "model_state": base_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "config": config.to_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "action_library": torch.as_tensor(action_library, dtype=torch.float32),
            }
            torch.save(checkpoint, args.output_dir / "latest.pt")
            if args.save_every > 0 and epoch % args.save_every == 0:
                torch.save(checkpoint, args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt")
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(checkpoint, args.output_dir / "best.pt")
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": val_metrics,
                        "lr": scheduler.get_last_lr()[0],
                        "samples_per_sec": seen / epoch_elapsed,
                    }
                )
            )
    finally:
        writer.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
