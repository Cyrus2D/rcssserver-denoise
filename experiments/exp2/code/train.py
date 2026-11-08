#!/usr/bin/env python3
"""Train exp2 and evaluate it against fixed fusion and logged librcsc."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import smooth_l1_loss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from data import Store, Windows, evaluation_batches
from model import Tracker, wrap
from player_physics import PlayerPhysics
from schema import FEATURE_INDEX as F

DEFAULT_DATASET = Path("/home/nader/workspace/denoise/logs/exp2")
POS_SCALE = 0.05
FACE_SCALE_DEG = 0.50
AGE_LABELS = ("fresh", "age_1", "age_2", "age_3_plus")
LARGE_TURN_GROUP = "age_1_or_2_large_turn"


def errors(state, truth):
    position = torch.linalg.vector_norm(state[..., :2] - truth[..., :2], dim=-1)
    face = wrap(state[..., 4:].sum(dim=-1) - truth[..., 4:].sum(dim=-1)).abs().rad2deg()
    velocity = torch.linalg.vector_norm(state[..., 2:4] - truth[..., 2:4], dim=-1)
    return position, face, velocity


def objective(state, truth):
    valid_pos = torch.isfinite(truth[..., :2]).all(dim=-1)
    valid_face = torch.isfinite(truth[..., 4:]).all(dim=-1)
    if not (valid_pos.any() or valid_face.any()):
        return None
    clean = torch.nan_to_num(truth)
    pos, face, _ = errors(state, clean)
    pos = pos[valid_pos] / POS_SCALE
    face = face[valid_face] / FACE_SCALE_DEG
    return (
        smooth_l1_loss(pos, torch.zeros_like(pos), reduction="sum") / max(len(pos), 1)
        + smooth_l1_loss(face, torch.zeros_like(face), reduction="sum") / max(len(face), 1)
    )


def collect(batch, estimates, truth, features):
    ages = (features[..., F["pos_age"]], features[..., F["face_age"]], None)
    all_errors = {name: errors(state, truth) for name, state in estimates.items()}
    for i, (metric, age) in enumerate(zip(("position_m", "face_deg", "velocity_m_per_cycle"), ages)):
        eligible = torch.isfinite(all_errors["librcsc"][i])
        values = torch.stack([error[i][eligible] for error in all_errors.values()], dim=1)
        if not torch.isfinite(values).all():
            raise ValueError(f"Nonfinite rollout {metric} on scored samples")
        batch[metric].append(values.cpu().numpy())
        if age is not None:
            for label, group in zip(AGE_LABELS, (age == 0, age == 1, age == 2, (age >= 3) & (age < 1000))):
                batch[f"{metric}/{label}"].append(values[group[eligible]].cpu().numpy())
            large_turn_key = f"{metric}/{LARGE_TURN_GROUP}"
            if large_turn_key in batch:
                group = ((age == 1) | (age == 2)) & (features[..., F["turn"]].abs() >= math.radians(30))
                batch[large_turn_key].append(values[group[eligible]].cpu().numpy())


@torch.inference_mode()
def evaluate(store, model, fixed, split, device, batch_size, burn_in, include_hybrid=False, player_physics=None):
    model.eval()
    include_hybrid = include_hybrid or player_physics is not None
    estimators = ("gru", "fixed", "librcsc", "hybrid") if include_hybrid else ("gru", "fixed", "librcsc")
    if player_physics is not None:
        estimators += ("gru_inertia", "gru_typed", "hybrid_typed")
    stats = {key: [] for key in ("position_m", "face_deg", "velocity_m_per_cycle")}
    for key in ("position_m", "face_deg"):
        stats.update({f"{key}/{label}": [] for label in AGE_LABELS})
        if player_physics is not None:
            stats[f"{key}/{LARGE_TURN_GROUP}"] = []
    sampled_gains = {"position": [], "face": []}
    if include_hybrid:
        sampled_gains["hybrid_position"] = []
    for batch in evaluation_batches(store, split, batch_size, burn_in, player_physics):
        inputs, seed, truth, baseline, *motion_data = (value.to(device) for value in batch)
        learned, gains = model(inputs, seed)
        physics, _ = fixed(inputs, seed)
        estimates = {"gru": learned, "fixed": physics, "librcsc": baseline}
        if include_hybrid:
            estimates["hybrid"], hybrid_gains = model(inputs, seed, reference=baseline)
            observed = (inputs[:, burn_in:, F["fix_pos"]] > 0.5) & torch.isfinite(truth[:, burn_in:, 0])
            values = hybrid_gains[:, burn_in:, 0][observed].cpu().numpy()
            sampled_gains["hybrid_position"].append(values[::max(1, len(values) // 1000)])
        if player_physics is not None:
            (motion,) = motion_data
            inertia_only = torch.as_tensor(player_physics.base, device=device).expand_as(motion).clone()
            inertia_only[..., 0] = motion[..., 0]
            estimates["gru_inertia"], _ = model(inputs, seed, motion=inertia_only)
            estimates["gru_typed"], _ = model(inputs, seed, motion=motion)
            estimates["hybrid_typed"], _ = model(inputs, seed, reference=baseline, motion=motion)
        collect(stats, {name: estimates[name][:, burn_in:] for name in estimators},
                truth[:, burn_in:], inputs[:, burn_in:])
        for i, (name, mask, field) in enumerate((("position", "fix_pos", 0), ("face", "fix_face", 4))):
            observed = (inputs[:, burn_in:, F[mask]] > 0.5) & torch.isfinite(truth[:, burn_in:, field])
            values = gains[:, burn_in:, i][observed].cpu().numpy()
            sampled_gains[name].append(values[::max(1, len(values) // 1000)])
    metrics = {}
    for key, parts in stats.items():
        if not parts:
            continue
        values = np.concatenate(parts)
        if not len(values):
            continue
        means, p90 = values.mean(axis=0, dtype=np.float64), np.quantile(values, 0.90, axis=0)
        metrics[key] = {"n": len(values), **{
            name: {"mean": float(means[i]), "p90": float(p90[i])}
            for i, name in enumerate(estimators)
        }, "gru_worse_than_librcsc_pct": float(100 * np.mean(values[:, 0] > values[:, 2]))}
        for i, name in enumerate(estimators[3:], start=3):
            metrics[key][f"{name}_worse_than_librcsc_pct"] = float(100 * np.mean(values[:, i] > values[:, 2]))
    if "position_m" not in metrics or "face_deg" not in metrics:
        raise ValueError(f"No scorable pose rollouts in {split}; check segments and burn-in")
    gains = {name: np.concatenate(parts) if parts else np.empty(0) for name, parts in sampled_gains.items()}
    return metrics, gains


def log_metrics(writer, split, metrics, gains, step):
    for metric, record in metrics.items():
        tag = f"{split}/{metric}"
        writer.add_scalar(f"{tag}/count", record["n"], step)
        writer.add_scalar(f"{tag}/worse_pct", record["gru_worse_than_librcsc_pct"], step)
        for estimator in (name for name, values in record.items() if isinstance(values, dict)):
            if estimator not in ("gru", "fixed", "librcsc"):
                writer.add_scalar(f"{tag}/{estimator}_worse_pct", record[f"{estimator}_worse_than_librcsc_pct"], step)
            for stat in ("mean", "p90"):
                writer.add_scalar(f"{tag}/{stat}/{estimator}", record[estimator][stat], step)
    for key, values in gains.items():
        if values.size:
            writer.add_histogram(f"{split}/gates/{key}", values, step)
            writer.add_scalar(f"{split}/gate_mean/{key}", values.mean(), step)


def show_metrics(split, metrics):
    pos, face = metrics["position_m"], metrics["face_deg"]
    print(f"{split}: mean position / looking-direction error on paired rollout cycles", flush=True)
    order = ("librcsc", "fixed", "gru", "hybrid", "gru_inertia", "gru_typed", "hybrid_typed")
    for name in (name for name in order if name in pos):
        print(f"  {name:8s} {100 * pos[name]['mean']:.3f} cm / {face[name]['mean']:.3f} deg", flush=True)


def score(metrics):
    return (metrics["position_m"]["gru"]["mean"] / POS_SCALE
            + metrics["face_deg"]["gru"]["mean"] / FACE_SCALE_DEG)


def train(args, store, device):
    windows = Windows(store, args.window, args.stride)
    count = min(len(windows), args.windows_per_epoch)
    sampler = RandomSampler(windows, replacement=False, num_samples=count,
                            generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(windows, batch_size=args.batch_size, sampler=sampler, pin_memory=device.type == "cuda")
    model = Tracker(store.manifest["physics"], args.hidden_size).to(device)
    fixed = Tracker(store.manifest["physics"], hidden_size=0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best_score, last_improvement, step = math.inf, 0, 0
    args.run_dir.mkdir(parents=True, exist_ok=False)
    config = {"hidden_size": args.hidden_size, "burn_in": args.burn_in, "window": args.window,
              "stride": args.stride, "dataset_fingerprint": store.fingerprint, "seed": args.seed,
              "lr": args.lr, "batch_size": args.batch_size, "windows_per_epoch": count,
              "parameters": sum(parameter.numel() for parameter in model.parameters())}
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{config['parameters']:,} parameters, {count:,} windows per epoch, device {device}", flush=True)
    with SummaryWriter(str(args.run_dir)) as writer:
        writer.add_text("training/config", json.dumps(config, indent=2))
        for epoch in range(1, args.epochs + 1):
            model.train()
            batches = 0
            for inputs, initial, truth in loader:
                inputs, initial, truth = (value.to(device, non_blocking=True) for value in (inputs, initial, truth))
                optimizer.zero_grad(set_to_none=True)
                state, _ = model(inputs, initial)
                loss = objective(state[:, args.burn_in:], truth[:, args.burn_in:])
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite training loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                batches += 1
                step += 1
                if step == 1 or step % 20 == 0:
                    writer.add_scalar("training/loss", loss.item(), step)
                    writer.add_scalar("training/grad_norm_before_clip", norm.item(), step)
                    writer.add_scalar("training/lr", optimizer.param_groups[0]["lr"], step)
            if not batches:
                raise ValueError("No training windows with usable truth")
            metrics, gains = evaluate(store, model, fixed, "validation", device, args.eval_batch_size, args.burn_in)
            log_metrics(writer, "validation", metrics, gains, epoch)
            show_metrics(f"Epoch {epoch} validation", metrics)
            value = score(metrics)
            writer.add_scalar("validation/selection_score", value, epoch)
            scheduler.step(value)
            if value < best_score:
                best_score, last_improvement = value, epoch
                temporary = args.run_dir / "best.tmp"
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, temporary)
                temporary.replace(args.run_dir / "best.pt")
                (args.run_dir / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            if epoch - last_improvement >= args.patience:
                print(f"Stopped after {args.patience} epochs without validation improvement", flush=True)
                break
    print(f"Best checkpoint: {args.run_dir / 'best.pt'}")


def evaluate_checkpoint(args, store, device):
    checkpoint = torch.load(args.evaluate, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if config["dataset_fingerprint"] != store.fingerprint:
        raise ValueError("Checkpoint was trained against a different dataset manifest")
    model = Tracker(store.manifest["physics"], config["hidden_size"]).to(device)
    fixed = Tracker(store.manifest["physics"], hidden_size=0).to(device)
    model.load_state_dict(checkpoint["model"])
    physics = PlayerPhysics(args.player_physics, store, args.split) if args.player_physics else None
    suffix = "-player-physics" if physics is not None else "-hybrid" if args.hybrid else ""
    output = args.run_dir or args.evaluate.parent / f"evaluation-{args.split}{suffix}"
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --run-dir")
    metrics, gains = evaluate(store, model, fixed, args.split, device, args.eval_batch_size,
                              config["burn_in"], include_hybrid=args.hybrid, player_physics=physics)
    show_metrics(args.split, metrics)
    output.mkdir(parents=True, exist_ok=False)
    with SummaryWriter(str(output)) as writer:
        log_metrics(writer, args.split, metrics, gains, checkpoint["epoch"])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Metrics: {output / 'metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, help="New directory for checkpoints and TensorBoard events")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"),
                        help="Compute device (default: cpu); auto selects CUDA when available")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--window", type=int, default=11)
    parser.add_argument("--burn-in", type=int, default=5)
    parser.add_argument("--stride", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--windows-per-epoch", type=int, default=60_000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--evaluate", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--hybrid", action="store_true",
                        help="In checkpoint evaluation, add GRU position with librcsc velocity and orientation")
    parser.add_argument("--player-physics", type=Path, metavar="INDEX",
                        help="In checkpoint evaluation, compare inertia-only and full player physics")
    parser.add_argument("--split", choices=("validation", "test"), default="test", help="Checkpoint evaluation split")
    args = parser.parse_args()
    if (args.hybrid or args.player_physics) and not args.evaluate:
        parser.error("--hybrid and --player-physics require --evaluate")
    positive = (args.threads, args.hidden_size, args.window, args.stride, args.epochs, args.patience,
                args.batch_size, args.eval_batch_size, args.windows_per_epoch)
    if min(positive) < 1 or not 0 <= args.burn_in < args.window or args.lr <= 0:
        parser.error("positive sizes and learning rate required, with 0 <= burn-in < window")
    if not args.evaluate and not args.run_dir:
        args.run_dir = args.dataset / "runs" / "gru48"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    try:
        store = Store(args.dataset)
        if args.evaluate:
            evaluate_checkpoint(args, store, device)
        else:
            train(args, store, device)
    except (OSError, ValueError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
