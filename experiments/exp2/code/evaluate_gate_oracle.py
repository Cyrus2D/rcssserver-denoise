#!/usr/bin/env python3
"""Measure the per-cycle headroom of a scalar position gate on validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import Store, evaluation_batches
from player_physics import PlayerPhysics
from schema import FEATURE_INDEX as F
from train import DEFAULT_DATASET
from train_landmarks import make_model


@torch.inference_mode()
def evaluate(store, checkpoint, batch_size):
    config = checkpoint["config"]
    if (config["dataset_fingerprint"] != store.fingerprint or config["variant"] != "summary"
            or config.get("position_residual", False)):
        raise ValueError("Use a matching summary checkpoint with a scalar position gate")
    physics = PlayerPhysics(store.root / "player-physics-validation", store, "validation")
    model = make_model(store, torch.device("cpu"), config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    burn_in = config["burn_in"]
    records = []
    for inputs, seed, truth, librcsc, motion in evaluation_batches(store, "validation", batch_size, burn_in, physics):
        states, gains, predictions = model(inputs, seed, reference=librcsc, motion=motion, return_prior=True)
        x = inputs[:, burn_in:]
        actual = truth[:, burn_in:, :2]
        fresh = (x[..., F["fix_pos"]] > 0.5) & torch.isfinite(actual).all(dim=-1)
        actual = actual[fresh]
        prior = predictions[:, burn_in:][fresh]
        measured = x[..., [F["fix_x"], F["fix_y"]]][fresh]
        learned = states[:, burn_in:, :2][fresh]
        baseline = librcsc[:, burn_in:, :2][fresh]

        direction = measured - prior
        length_squared = direction.square().sum(dim=-1)
        projection = ((actual - prior) * direction).sum(dim=-1) / length_squared.clamp_min(1e-12)
        best = prior + direction * projection.clamp(0, 1)[:, None]
        distances = [torch.linalg.vector_norm(position - actual, dim=-1)
                     for position in (baseline, prior, measured, learned, best)]
        if not all(torch.isfinite(values).all() for values in distances):
            raise ValueError("Nonfinite position in oracle cohort")
        if (distances[-1] > distances[-2] + 1e-5).any():
            raise ValueError("Projection is worse than the learned scalar gate")
        records.append(torch.stack(distances, dim=-1).cpu().numpy())
    errors = np.concatenate(records)
    if not len(errors):
        raise ValueError("No usable fresh position fixes")
    mean, p90 = errors.mean(axis=0, dtype=np.float64), np.quantile(errors, .90, axis=0)
    return {"n": len(errors), "epoch": checkpoint["epoch"], "split": "validation",
            "scope": "fresh fixes; same prior on each cycle; oracle does not feed back",
            "errors_m": {name: {"mean": float(mean[i]), "p90": float(p90[i])}
                         for i, name in enumerate(("librcsc", "motion_prior", "visual_fix", "learned", "oracle"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.dataset / "runs" / "step04-summary" / "best.pt"
    output = args.output or args.dataset / "runs" / "step05-gate-oracle.json"
    if output.exists():
        parser.error(f"{output} already exists; choose a new --output")
    torch.set_num_threads(args.threads)
    data = evaluate(Store(args.dataset), torch.load(checkpoint, map_location="cpu", weights_only=True), 128)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    print(f'{data["n"]:,} fresh validation fixes')
    for name, record in data["errors_m"].items():
        print(f'{name:12s} mean/p90 {100 * record["mean"]:.3f}/{100 * record["p90"]:.3f} cm')
    print(f"Oracle metrics: {output}")


if __name__ == "__main__":
    main()
