#!/usr/bin/env python3
"""Learn a bounded radial correction and compare with causal position filters."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from build_sequences import AGE, BOUNDS_START, FEATURE_START, FEATURES, POSE_START, PRED_START, PREDICTIONS

WINDOW = 8
DISTANCE = ("0–10", "10–20", "20–30", "30–40", "40–60", "60+")


class RangeGRU(nn.Module):
    def __init__(self, hidden, anchor):
        super().__init__()
        self.anchor = anchor
        self.gru = nn.GRU(FEATURES, hidden, batch_first=True)
        self.head = nn.Linear(hidden + 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, bounds):
        state, _ = self.gru(features)
        low, high, baseline = bounds.unbind(dim=-1)
        width = high - low
        anchor = (baseline - (low + high) / 2) / width.clamp(min=.01)
        context = torch.cat((state[:, -1], anchor[:, None]), dim=-1)
        offset = self.head(context).squeeze(-1).tanh() * width
        return torch.clamp(baseline + offset, low, high)


class Games:
    def __init__(self, root, split, stride=1):
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["version"] != 1:
            raise ValueError("unexpected sequence dataset version")
        self.names = []
        self.data = []
        self.points = []
        for game in manifest["games"]:
            if game["split"] != split:
                continue
            name = game["name"]
            array = np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            ages = array[:, ::stride, AGE]
            truth = array[:, ::stride, :4]
            points = np.argwhere((ages >= WINDOW) & np.isfinite(truth).all(axis=-1)).astype(np.int32)
            points[:, 1] *= stride
            if len(points):
                self.names.append(name)
                self.data.append(array)
                self.points.append(points)
        if not self.data:
            raise ValueError(f"no usable windows for {split}")

    def batch(self, game, points):
        tracks, end = points.T
        offsets = np.arange(1 - WINDOW, 1)
        data = self.data[game]
        features = data[tracks[:, None], end[:, None] + offsets, FEATURE_START:].copy()
        return features, data[tracks, end].copy()


def measurement_geometry(rows, last_features):
    origin = rows[:, POSE_START:POSE_START + 2]
    face = rows[:, POSE_START + 2]
    angle = face + np.arctan2(last_features[:, 1], last_features[:, 2])
    bounds = rows[:, BOUNDS_START:BOUNDS_START + 3]
    blend_pos = PRED_START + 2 * PREDICTIONS.index("blend")
    blend_vector = rows[:, blend_pos:blend_pos + 2] - origin
    relative_angle = (np.arctan2(blend_vector[:, 1], blend_vector[:, 0]) - angle + np.pi) % (2 * np.pi) - np.pi
    bounded_angle = angle + np.clip(relative_angle, -np.deg2rad(.5), np.deg2rad(.5))
    bounded_range = np.clip(np.linalg.norm(blend_vector, axis=-1), bounds[:, 0], bounds[:, 1])
    blend_bounds = np.column_stack((bounds[:, :2], bounded_range))
    return origin, face, angle, bounds, bounded_angle, blend_bounds


def polar_to_global(origin, angle, radius):
    return origin + radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))


def score(model, games, batch_size, detailed):
    names = PREDICTIONS + ("bounded_blend",) + (("gru",) if model is not None else ())
    distances, motions, relative, world = [], [], [], []
    game_counts, game_gains = [], []
    with torch.inference_mode():
        for game, points in enumerate(games.points):
            game_start = len(relative)
            for start in range(0, len(points), batch_size):
                features, rows = games.batch(game, points[start:start + batch_size])
                origin, face, angle, bounds, blend_angle, blend_bounds = measurement_geometry(rows, features[:, -1])
                global_pos = rows[:, PRED_START:POSE_START].reshape(len(rows), len(PREDICTIONS), 2)
                bounded = polar_to_global(origin, blend_angle, blend_bounds[:, 2])
                parts = [global_pos, bounded[:, None]]
                if model is not None:
                    if model.anchor == "blend":
                        angle, bounds = blend_angle, blend_bounds
                    radius = model(torch.from_numpy(features), torch.from_numpy(bounds)).numpy()
                    estimate = polar_to_global(origin, angle, radius)
                    parts.append(estimate[:, None])
                global_pos = np.concatenate(parts, axis=1)

                # Compare in observer coordinates without injecting true observer pose.
                # Rotate the target truth vector into the estimated axes for an equivalent Euclidean norm.
                true_angle = np.deg2rad(rows[:, 1]) + face
                true_rel = rows[:, :1] * np.column_stack((np.cos(true_angle), np.sin(true_angle)))
                relative.append(np.linalg.norm(global_pos - origin[:, None] - true_rel[:, None], axis=-1))
                if detailed:
                    distances.append(rows[:, 0])
                    motions.append(rows[:, 4] == 1)
                    world.append(np.linalg.norm(global_pos - rows[:, None, 2:4], axis=-1))
            if detailed and model is not None:
                batches = relative[game_start:]
                index = names.index("bounded_blend")
                game_counts.append(len(points))
                game_gains.append(sum(float((part[:, index] - part[:, -1]).sum(dtype=np.float64))
                                      for part in batches))
    relative = np.concatenate(relative)
    if not np.isfinite(relative).all():
        raise ValueError("non-finite prediction on evaluation cohort")
    if not detailed:
        return dict(zip(names, np.mean(relative, axis=0, dtype=np.float64).tolist()))
    world = np.concatenate(world)
    distances = np.concatenate(distances)
    motions = np.concatenate(motions)
    groups = {"all": np.ones(len(relative), dtype=bool), "motion": motions, "no_motion": ~motions}
    distance_bin = np.searchsorted([10, 20, 30, 40, 60], distances, side="right")
    groups.update({label: distance_bin == index for index, label in enumerate(DISTANCE)})
    result = {}
    for label, mask in groups.items():
        if not mask.any():
            continue
        result[label] = {"n": int(mask.sum())}
        for field, errors in (("relative", relative), ("global", world)):
            error = errors[mask]
            result[label][field] = {
                name: {"mean": float(np.mean(error[:, index], dtype=np.float64)),
                       "p90": float(np.quantile(error[:, index], .9))}
                for index, name in enumerate(names)
            }
    if game_gains:
        gains, counts = np.array(game_gains), np.array(game_counts)
        sample = np.random.default_rng(42).integers(len(gains), size=(10000, len(gains)))
        bootstrap = gains[sample].sum(axis=1) / counts[sample].sum(axis=1)
        result["all"]["gru_gain_over_bounded_blend"] = {
            "games_improved": int((gains > 0).sum()),
            "mean_gain_ci95_m": np.quantile(bootstrap, [.025, .975]).tolist(),
        }
    return result


def train(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_data = Games(args.dataset, "train")
    valid = Games(args.dataset, "validation", args.validation_stride)
    model = RangeGRU(args.hidden, args.anchor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    samples = np.array([len(points) for points in train_data.points])
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        losses = []
        for game in rng.choice(len(samples), args.batches, p=samples / samples.sum()):
            endpoints = train_data.points[game][rng.integers(samples[game], size=args.batch_size)]
            features, rows = train_data.batch(game, endpoints)
            _, _, _, bounds, _, blend_bounds = measurement_geometry(rows, features[:, -1])
            predicted = model(torch.from_numpy(features), torch.from_numpy(blend_bounds if args.anchor == "blend" else bounds))
            loss = nn.functional.smooth_l1_loss(predicted, torch.from_numpy(rows[:, 0]), beta=.2)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = score(model, valid, args.batch_size * 4, detailed=False)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "mean_relative_m": metrics})
        print(f"Epoch {epoch}: relative cm " + " | ".join(f"{key} {100*value:.3f}" for key, value in metrics.items()), flush=True)
        if metrics["gru"] < best:
            best = metrics["gru"]
            torch.save({"hidden": args.hidden, "epoch": epoch, "anchor": args.anchor, "model": model.state_dict()}, output / "best.pt")
        (output / "validation.json").write_text(json.dumps(history, indent=2) + "\n")


def evaluate(args):
    checkpoint = torch.load(args.evaluate, weights_only=True)
    model = RangeGRU(checkpoint["hidden"], checkpoint["anchor"])
    model.load_state_dict(checkpoint["model"])
    games = Games(args.dataset, args.split, args.stride)
    scores = score(model, games, args.batch_size * 4, detailed=True)
    payload = {"split": args.split, "stride": args.stride, "games": games.names,
               "hidden": checkpoint["hidden"], "epoch": checkpoint["epoch"], "anchor": checkpoint["anchor"], "scores": scores}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    overall = scores["all"]
    print(f'{len(games.names)} games, {overall["n"]:,} paired observations')
    for key, metric in overall["relative"].items():
        print(f'{key:20} relative {100*metric["mean"]:7.3f}/{100*metric["p90"]:.3f} cm mean/p90', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path, help="run directory for training or JSON file for evaluation")
    parser.add_argument("--anchor", choices=("quantized", "blend"), default="quantized")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batches", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-stride", type=int, default=24)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if min(args.hidden, args.epochs, args.batches, args.batch_size, args.threads, args.stride, args.validation_stride) < 1:
        parser.error("all count and stride options must be positive")
    torch.set_num_threads(args.threads)
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
