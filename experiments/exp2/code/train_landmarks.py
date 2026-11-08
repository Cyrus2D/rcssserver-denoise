#!/usr/bin/env python3
"""Compare summary-only and fixed-landmark inputs on the position-only hybrid."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import smooth_l1_loss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from data import Store, Windows, evaluation_batches
from landmark_slots import LANDMARK_COUNT, LandmarkSlots
from model import Tracker
from player_physics import PlayerPhysics
from schema import FEATURE_INDEX as F
from train import AGE_LABELS, DEFAULT_DATASET, POS_SCALE


class HybridWindows(Windows):
    def __init__(self, store, physics, landmarks, steps, stride):
        super().__init__(store, steps, stride)
        self.physics = physics
        self.landmarks = landmarks

    def __getitem__(self, index):
        file, start, rows = self.window(index)
        first, end = start + 1, start + len(rows)
        arrays = (rows["features"][1:], rows["librcsc"][0], rows["truth"][1:],
                  rows["librcsc"][1:], self.physics.rows(file, first, end))
        if self.landmarks is not None:
            arrays += (self.landmarks.rows(file, first, end),)
        return tuple(torch.from_numpy(np.array(array, copy=True)) for array in arrays)


def rollout(model, batch, device):
    inputs, seed, truth, librcsc, motion, *slots = (array.to(device) for array in batch)
    estimate, gates = model(inputs, seed, reference=librcsc, motion=motion,
                            landmarks=slots[0] if slots else None)
    return inputs, estimate, truth, librcsc, gates


def position_loss(estimate, truth):
    valid = torch.isfinite(truth[..., :2]).all(dim=-1)
    if not valid.any():
        return None
    error = torch.linalg.vector_norm((estimate[..., :2][valid] - truth[..., :2][valid]) / POS_SCALE, dim=-1)
    return smooth_l1_loss(error, torch.zeros_like(error))


@torch.inference_mode()
def evaluate(store, model, physics, landmarks, split, device, batch_size, burn_in):
    model.eval()
    errors = {name: [] for name in ("overall", *AGE_LABELS)}
    gate_samples = []
    for batch in evaluation_batches(store, split, batch_size, burn_in, physics, landmarks):
        inputs, estimate, truth, librcsc, gates = rollout(model, batch, device)
        estimate, truth, librcsc = (array[:, burn_in:] for array in (estimate, truth, librcsc))
        scored = torch.isfinite(truth[..., :2]).all(dim=-1) & torch.isfinite(librcsc[..., :2]).all(dim=-1)
        error = torch.linalg.vector_norm(estimate[..., :2][scored] - truth[..., :2][scored], dim=-1)
        baseline = torch.linalg.vector_norm(librcsc[..., :2][scored] - truth[..., :2][scored], dim=-1)
        if not torch.isfinite(error).all():
            raise ValueError("Nonfinite position in a scored rollout")
        paired = torch.stack((baseline, error), dim=-1).cpu().numpy()
        errors["overall"].append(paired)
        age = inputs[:, burn_in:, F["pos_age"]][scored].cpu().numpy()
        for label, mask in zip(AGE_LABELS, (age == 0, age == 1, age == 2, (age >= 3) & (age < 1000))):
            errors[label].append(paired[mask])
        available = (inputs[:, burn_in:, F["fix_pos"]] > 0.5) & scored
        sample = gates[:, burn_in:, 0][available].cpu().numpy()
        gate_samples.append(sample[::max(1, len(sample) // 1000)])
    metrics = {}
    for group, chunks in errors.items():
        if not chunks:
            continue
        values = np.concatenate(chunks)
        if not len(values):
            continue
        mean = values.mean(axis=0, dtype=np.float64)
        p90 = np.quantile(values, .90, axis=0)
        metrics[group] = {
            "n": len(values),
            "librcsc": {"mean": float(mean[0]), "p90": float(p90[0])},
            "hybrid": {"mean": float(mean[1]), "p90": float(p90[1])},
            "hybrid_worse_pct": float(100 * np.mean(values[:, 1] > values[:, 0])),
        }
    if "overall" not in metrics:
        raise ValueError(f"No scorable positions in {split}")
    return metrics, np.concatenate(gate_samples)


def log_metrics(writer, split, metrics, gates, epoch):
    for group, record in metrics.items():
        tag = f"{split}/position_m/{group}"
        writer.add_scalar(f"{tag}/count", record["n"], epoch)
        writer.add_scalar(f"{tag}/worse_pct", record["hybrid_worse_pct"], epoch)
        for estimator in ("hybrid", "librcsc"):
            for metric in ("mean", "p90"):
                writer.add_scalar(f"{tag}/{metric}/{estimator}", record[estimator][metric], epoch)
    if len(gates):
        writer.add_histogram(f"{split}/gates/position", gates, epoch)
        writer.add_scalar(f"{split}/gate_mean/position", float(gates.mean()), epoch)


def show(split, variant, metrics, epoch):
    values = metrics["overall"]
    print(f"{variant} epoch {epoch} {split}: position mean/p90 "
          f"{100 * values['hybrid']['mean']:.3f}/{100 * values['hybrid']['p90']:.3f} cm, "
          f"librcsc {100 * values['librcsc']['mean']:.3f}/{100 * values['librcsc']['p90']:.3f} cm "
          f"({values['n']:,} paired cycles)", flush=True)


def make_model(store, device, config=None):
    config = config or {}
    return Tracker(
        store.manifest["physics"], hidden_size=config.get("hidden_size", 48), landmark_count=LANDMARK_COUNT,
        circular_angles=config.get("circular_angles", False),
        position_residual=config.get("position_residual", False),
        landmark_summary=config.get("landmark_summary", True),
    ).to(device)


def train(args, store, slots, device, output):
    training_physics = PlayerPhysics(args.dataset / "player-physics-train", store, "train")
    validation_physics = PlayerPhysics(args.dataset / "player-physics-validation", store, "validation")
    windows = HybridWindows(store, training_physics, slots, steps=11, stride=11)
    count = min(len(windows), args.windows_per_epoch)
    sampler = RandomSampler(windows, replacement=False, num_samples=count,
                            generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(windows, batch_size=args.batch_size, sampler=sampler)
    architecture = {"hidden_size": args.hidden_size, "circular_angles": args.circular_angles,
                    "position_residual": args.position_residual, "landmark_summary": args.landmark_summary}
    model = make_model(store, device, architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    best = math.inf
    step = 0
    config = {"variant": args.variant, "dataset_fingerprint": store.fingerprint, "seed": args.seed,
              "epochs": args.epochs, "batch_size": args.batch_size, "windows_per_epoch": count,
              "parameters": sum(p.numel() for p in model.parameters()), "burn_in": 5, **architecture}
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{args.variant}: {config['parameters']:,} parameters, {count:,} windows per epoch", flush=True)
    with SummaryWriter(str(output)) as writer:
        writer.add_text("training/config", json.dumps(config, indent=2))
        for epoch in range(1, args.epochs + 1):
            model.train()
            trained = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                _, state, truth, _, _ = rollout(model, batch, device)
                loss = position_loss(state[:, 5:], truth[:, 5:])
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss at step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                trained += 1
                step += 1
                if step == 1 or step % 20 == 0:
                    writer.add_scalar("training/position_loss", loss.item(), step)
                    writer.add_scalar("training/grad_norm_before_clip", norm.item(), step)
            if not trained:
                raise ValueError("No scorable training windows")
            metrics, gates = evaluate(store, model, validation_physics, slots,
                                      "validation", device, args.eval_batch_size, 5)
            log_metrics(writer, "validation", metrics, gates, epoch)
            show("validation", args.variant, metrics, epoch)
            value = metrics["overall"]["hybrid"]["mean"]
            if value < best:
                best = value
                temporary = output / "best.tmp"
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, temporary)
                temporary.replace(output / "best.pt")
                (output / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")


def evaluate_checkpoint(args, store, device):
    checkpoint = torch.load(args.evaluate, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if config["dataset_fingerprint"] != store.fingerprint:
        raise ValueError("Checkpoint was trained against a different dataset")
    slots = open_slots(args, store, config["variant"])
    physics = PlayerPhysics(args.dataset / f"player-physics-{args.split}", store, args.split)
    model = make_model(store, device, config)
    model.load_state_dict(checkpoint["model"])
    output = args.run_dir or args.evaluate.parent / f"evaluation-{args.split}"
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --run-dir")
    metrics, gates = evaluate(store, model, physics, slots, args.split, device, args.eval_batch_size, config["burn_in"])
    show(args.split, config["variant"], metrics, checkpoint["epoch"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with SummaryWriter(str(output)) as writer:
        log_metrics(writer, args.split, metrics, gates, checkpoint["epoch"])


def open_slots(args, store, variant):
    return LandmarkSlots(args.landmarks or args.dataset / "landmark-slots", store) if variant == "flags" else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("summary", "flags"), default="summary")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landmarks", type=Path, help="Landmark index directory")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--circular-angles", action="store_true", help="Encode circular sensor directions as sine and cosine")
    parser.add_argument("--position-residual", action="store_true", help="Learn a bounded two-dimensional correction on fresh fixes")
    parser.add_argument("--no-landmark-summary", dest="landmark_summary", action="store_false",
                        help="Omit the landmark geometry summaries from the network inputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--windows-per-epoch", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--evaluate", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    if min(args.threads, args.epochs, args.windows_per_epoch, args.batch_size, args.eval_batch_size, args.hidden_size) < 1:
        parser.error("Training sizes and thread count must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    try:
        store = Store(args.dataset)
        device = torch.device(args.device)
        if args.evaluate:
            evaluate_checkpoint(args, store, device)
        else:
            slots = open_slots(args, store, args.variant)
            output = args.run_dir or args.dataset / "runs" / f"step04-{args.variant}"
            train(args, store, slots, device, output)
    except (OSError, KeyError, ValueError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
