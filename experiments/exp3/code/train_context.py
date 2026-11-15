#!/usr/bin/env python3
"""Compare reachability, learned displacement, and context on Exp3 tracks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from build_intersections import CONTEXT_SIZE, TRACK_FEATURES
from build_sequences import FEATURE_START, FEATURES, POSE_START, PRED_START
from train_range import DISTANCE, WINDOW, Games, RangeGRU, measurement_geometry, polar_to_global

SET_MODES = {"masked_set", "masked_set_no_unum", "masked_set_role", "masked_set_polar"}
MODES = ("region", "motion", "ball", "players", "unum", "unum_no_motion",
         "masked_set", "masked_set_no_unum", "masked_set_role", "masked_set_polar")
NO_MOTION = {"region", "unum_no_motion", *SET_MODES}
WITH_UNUM = {"unum", "unum_no_motion", "masked_set", "masked_set_polar"}


class ContextGames(Games):
    def __init__(self, dataset, sidecars, split, stride):
        super().__init__(dataset, split, stride)
        self.geometry = [np.load(sidecars / f"{name}-geometry.npy", mmap_mode="r") for name in self.names]
        self.context = [np.load(sidecars / f"{name}-context.npy", mmap_mode="r") for name in self.names]
        self.unum = [np.load(sidecars / f"{name}-unums.npy") for name in self.names]
        for name, data, geometry, context, nums in zip(self.names, self.data, self.geometry, self.context, self.unum):
            if (geometry.shape != (*data.shape[:2], 4) or context.shape != (*data.shape[:2], CONTEXT_SIZE)
                    or nums.shape != (data.shape[0],) or not ((nums >= 1) & (nums <= 11)).all()):
                raise ValueError(f"{name}: invalid sidecar layout")

    def batch(self, game, points):
        features, current = super().batch(game, points)
        tracks, end = points.T
        earlier = end[:, None] + np.arange(1 - WINDOW, 0)
        context = self.context[game][tracks[:, None], earlier].astype(np.float32)
        geometry = self.geometry[game]
        previous = self.data[game][tracks, end - 1, POSE_START:POSE_START + 3].copy()
        prev_geometry, curr_geometry = geometry[tracks, end - 1].copy(), geometry[tracks, end].copy()
        return features, current, context, self.unum[game][tracks].copy(), previous, prev_geometry, curr_geometry


def unit(angle):
    return torch.stack((angle.cos(), angle.sin()), dim=-1)


class ContextGRU(nn.Module):
    def __init__(self, mode, hidden=64):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Embedding(12, 4, padding_idx=0)
        self.encoder = nn.GRU(FEATURES + CONTEXT_SIZE + 4, hidden, batch_first=True)
        self.motion = nn.Linear(hidden, 2)
        self.readout = nn.Sequential(nn.Linear(hidden + 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        for layer in (self.motion, self.readout[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.mode in SET_MODES:
            self.neighbor_encoder = nn.Sequential(nn.Linear(TRACK_FEATURES, 16), nn.SiLU(), nn.Linear(16, 13))

        if self.mode == "masked_set_polar":
            self.bearing = nn.Linear(hidden + 3, 1)
            nn.init.zeros_(self.bearing.weight)
            nn.init.zeros_(self.bearing.bias)

    def set_context(self, context):
        ball = context[..., :TRACK_FEATURES]
        neighbors = context[..., TRACK_FEATURES:].unflatten(-1, (2, 2, TRACK_FEATURES))
        available = neighbors[..., 5:6]
        count = available.sum(dim=-2)
        pooled = (self.neighbor_encoder(neighbors) * available).sum(dim=-2) / count.clamp_min(1)
        grouped = torch.cat((pooled, count / 2), dim=-1).flatten(-2)
        return torch.cat((ball, grouped), dim=-1)

    def forward(self, features, context, unum, origin, previous, old, region):
        if self.mode == "ball":
            context = torch.cat((context[..., :TRACK_FEATURES], torch.zeros_like(context[..., TRACK_FEATURES:])), dim=-1)
        elif self.mode in ("region", "motion"):
            context = torch.zeros_like(context)
        last_ball = context[:, -1, :TRACK_FEATURES]
        if self.mode in SET_MODES:
            context = self.set_context(context)
        context = torch.cat((context, torch.zeros_like(context[:, :1])), dim=1)
        embedding = self.embedding(unum if self.mode in WITH_UNUM else torch.zeros_like(unum))
        identity = embedding[:, None].expand(-1, features.shape[1], -1)
        if self.mode == "masked_set_role":
            last_position = previous[:, :2] + old[:, 3, None] * unit(old[:, 2])
            field_position = last_position / last_position.new_tensor([52.5, 34])
            ball_distance = torch.linalg.vector_norm(last_ball[:, :2], dim=-1, keepdim=True)
            descriptor = torch.cat((field_position, ball_distance, last_ball[:, 5:6]), dim=-1)
            identity = torch.cat((torch.zeros_like(identity[:, :-1]), descriptor[:, None]), dim=1)
        inputs = torch.cat((features, context, identity), dim=-1)
        hidden, _ = self.encoder(inputs)
        low, high, angle, center = region.unbind(dim=-1)
        width = (high - low).clamp(min=.01)
        old_pos = previous[:, :2] + old[:, 3, None] * unit(old[:, 2])
        delta = self.motion(hidden[:, -2]).tanh()
        delta = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp(min=1)
        direction = unit(previous[:, 2])
        sideways = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
        prior = old_pos + 1.5 * (delta[:, :1] * direction + delta[:, 1:] * sideways)
        prior_range = ((prior - origin) * unit(angle)).sum(dim=-1)
        motion_hint = (prior_range - center).clamp(-5, 5) / 3
        if self.mode in NO_MOTION:
            motion_hint = torch.zeros_like(motion_hint)
        state = torch.cat((hidden[:, -1], motion_hint[:, None],
                           ((center - (low + high)/2) / width)[:, None], (width/3)[:, None]), dim=-1)
        radius = torch.clamp(center + self.readout(state).squeeze(-1).tanh() * width, low, high)
        if self.mode == "masked_set_polar":
            return radius, prior, self.bearing(state).squeeze(-1).tanh()
        return radius, prior


def model_batch(model, batch):
    features, rows, context, unum, previous, old, region = batch
    origin = rows[:, POSE_START:POSE_START + 2]
    inputs = map(torch.from_numpy, (features, context, unum, origin, previous, old, region))
    radius, prior, *bearing = model(*inputs)
    angle = torch.from_numpy(region[:, 2])
    if bearing:
        # The selected reachability ray already lies inside the raw one-degree cell.
        raw = torch.from_numpy(rows[:, POSE_START + 2]) + torch.atan2(
            torch.from_numpy(features[:, -1, 1]), torch.from_numpy(features[:, -1, 2]))
        difference = angle - raw
        difference = torch.atan2(difference.sin(), difference.cos())
        half_step = torch.deg2rad(angle.new_tensor(.5))
        angle = angle + (difference + half_step * bearing[0]).clamp(-half_step, half_step) - difference
    direction = unit(angle)
    return radius[:, None] * direction, prior


def relative_truth(rows):
    true_angle = np.deg2rad(rows[:, 1]) + rows[:, POSE_START + 2]
    return rows[:, :1] * np.stack((np.cos(true_angle), np.sin(true_angle)), axis=-1)


def load_old(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = RangeGRU(state["hidden"], state["anchor"])
    model.load_state_dict(state["model"])
    return model.eval()


def benchmark(games, models, old, batch_size):
    labels = ["visual", "bounded_blend", "step02", "intersection", *models]
    rel, glob, distances, motions = [], [], [], []
    game_stats = {}
    with torch.inference_mode():
        for index, points in enumerate(games.points):
            sums = np.zeros(len(labels), dtype=np.float64)
            for start in range(0, len(points), batch_size):
                batch = games.batch(index, points[start:start + batch_size])
                features, rows, *_, geometry = batch
                origin, _, angle, bounds, blend_angle, blend_bounds = measurement_geometry(rows, features[:, -1])
                if old.anchor == "blend":
                    angle, bounds = blend_angle, blend_bounds
                old_range = old(torch.from_numpy(features), torch.from_numpy(bounds)).numpy()
                baseline = polar_to_global(origin, angle, old_range)
                bounded = polar_to_global(origin, blend_angle, blend_bounds[:, 2])
                intersection = polar_to_global(origin, geometry[:, 2], geometry[:, 3])
                predictions = np.stack((
                    rows[:, PRED_START+2:PRED_START+4], bounded, baseline, intersection,
                    *(model_batch(model, batch)[0].numpy() + origin for model in models.values()),
                ), axis=1)
                actual = relative_truth(rows)
                error = np.linalg.norm(predictions - origin[:, None] - actual[:, None], axis=-1)
                if not np.isfinite(error).all():
                    raise ValueError("non-finite prediction on scoring cohort")
                sums += error.sum(axis=0, dtype=np.float64)
                rel.append(error)
                glob.append(np.linalg.norm(predictions - rows[:, None, 2:4], axis=-1))
                distances.append(rows[:, 0])
                motions.append(rows[:, 4] == 1)
            game_stats[games.names[index]] = {"n": len(points), "sum_relative_m": dict(zip(labels, sums.tolist()))}
    n = sum(game["n"] for game in game_stats.values())
    relative, world, distance, motion = map(np.concatenate, (rel, glob, distances, motions))
    bins = np.searchsorted([10, 20, 30, 40, 60], distance, side="right")
    groups = {"all": np.ones(n, dtype=bool), "motion_available": motion, "motion_missing": ~motion}
    groups.update({label: bins == i for i, label in enumerate(DISTANCE)})
    results = {}
    for label, mask in groups.items():
        if not mask.any():
            continue
        results[label] = {"n": int(mask.sum())}
        for frame, errors in (("relative", relative), ("global", world)):
            data = errors[mask]
            mean = data.mean(axis=0, dtype=np.float64)
            p90 = np.quantile(data, .9, axis=0)
            results[label][frame] = {name: {"mean": float(a), "p90": float(b)}
                                     for name, a, b in zip(labels, mean, p90)}
    return {"scores": results, "games": game_stats}


def validation_mean(model, validation, batch_size):
    total = count = 0
    with torch.inference_mode():
        for game, points in enumerate(validation.points):
            for start in range(0, len(points), batch_size):
                batch = validation.batch(game, points[start:start + batch_size])
                estimate, _ = model_batch(model, batch)
                error = np.linalg.norm(estimate.numpy() - relative_truth(batch[1]), axis=-1)
                total += float(error.sum(dtype=np.float64))
                count += len(error)
    return total / count


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    games = ContextGames(args.dataset, args.sidecars, "train", 1)
    validation = ContextGames(args.dataset, args.sidecars, "validation", args.stride)
    model = ContextGRU(args.mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    samples = np.array([len(points) for points in games.points])
    args.output.mkdir(parents=True, exist_ok=False)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for game in rng.choice(len(samples), size=args.batches, p=samples/samples.sum()):
            endpoints = games.points[game][rng.integers(samples[game], size=args.batch_size)]
            batch = games.batch(game, endpoints)
            rows = batch[1]
            estimate, prior = model_batch(model, batch)
            target = torch.from_numpy(relative_truth(rows))
            error = torch.linalg.vector_norm(estimate - target, dim=-1)
            loss = nn.functional.smooth_l1_loss(error, torch.zeros_like(error), beta=.2)
            if args.mode not in NO_MOTION:
                actual = torch.from_numpy(rows[:, 2:4])
                loss += .12 * nn.functional.smooth_l1_loss(prior, actual, beta=.2)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        metric = validation_mean(model, validation, args.batch_size * 4)
        history.append({"epoch": epoch, "mean_relative_m": metric})
        print(f"Epoch {epoch} | {args.mode} {100*metric:.2f} cm", flush=True)
        if metric < best:
            best = metric
            torch.save({"model": model.state_dict(), "mode": args.mode, "epoch": epoch}, args.output / "best.pt")
        (args.output / "validation.json").write_text(json.dumps(history, indent=2) + "\n")


def evaluate(args):
    old = load_old(args.baseline)
    games = ContextGames(args.dataset, args.sidecars, args.split, args.stride)
    models, epochs = {}, {}
    for path in args.evaluate:
        state = torch.load(path, map_location="cpu", weights_only=True)
        mode = state["mode"]
        model = ContextGRU(mode)
        model.load_state_dict(state["model"])
        models[mode], epochs[mode] = model.eval(), state["epoch"]
    result = benchmark(games, models, old, args.batch_size * 4)
    result.update(split=args.split, stride=args.stride, epochs=epochs)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f'{len(games.names)} {args.split} games, {result["scores"]["all"]["n"]:,} observations')
    for name, stat in result["scores"]["all"]["relative"].items():
        print(f'{name:20} relative {100*stat["mean"]:.2f}/{100*stat["p90"]:.2f} cm mean/p90')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("sidecars", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=MODES, default="region")
    parser.add_argument("--evaluate", nargs="*", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batches", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.stride, args.epochs, args.batches, args.batch_size, args.threads) < 1:
        parser.error("count and stride arguments must be positive")
    torch.set_num_threads(args.threads)
    if args.evaluate is None:
        train(args)
    else:
        evaluate(args)
