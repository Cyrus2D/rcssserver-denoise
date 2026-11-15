#!/usr/bin/env python3
"""Compare causal one-cycle forecasts and fresh quantized position updates."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from build_sequences import AGE, BOUNDS_START, FEATURES, POSE_START, PRED_START
from replay_normal import quantize_distance, rotate
from train_context import ContextGRU, ContextGames, model_batch, relative_truth
from train_range import DISTANCE, WINDOW

STRENGTHS = (.25, .5, 1.)
FAMILIES = ("cv", "single", "mixture")


class Forecast(nn.Module):
    def __init__(self, components):
        super().__init__()
        self.encoder = nn.GRU(FEATURES, 48, batch_first=True)
        self.head = nn.Linear(48, components * 5)
        self.components = components

    def forward(self, features):
        state, _ = self.encoder(features)
        raw = self.head(state[:, -1]).reshape(-1, self.components, 5)
        mean = 1.2 * raw[..., :2].tanh()
        scale = .04 + .66 * raw[..., 2:4].sigmoid()
        weight = raw[..., 4].log_softmax(dim=1)
        return mean, scale, weight


def forecast_support(games, game, points):
    tracks, end = points.T
    data = games.data[game]
    # Age counts uninterrupted valid observations, independently of truth.
    current = data[tracks, end]
    previous = data[tracks, end - 1]
    return (current[:, AGE] > WINDOW) & np.isfinite(previous[:, :4]).all(axis=1)


def train(args, components):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    train = ContextGames(args.dataset, args.sidecars, "train", 1)
    validation = ContextGames(args.dataset, args.sidecars, "validation", 24)
    train_points = [p[forecast_support(train, g, p)] for g, p in enumerate(train.points)]
    valid_points = [p[forecast_support(validation, g, p)] for g, p in enumerate(validation.points)]
    counts = np.array([len(p) for p in train_points])
    model = Forecast(components)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)

    def loss(store, game, points):
        prior_points = points.copy()
        prior_points[:, 1] -= 1
        features, prior = store.batch(game, prior_points)[:2]
        tracks, end = points.T
        target = store.data[game][tracks, end, 2:4]
        movement = rotate(target - prior[:, 2:4], -prior[:, POSE_START + 2])
        mean, scale, weight = model(torch.from_numpy(features))
        z = (torch.from_numpy(movement)[:, None] - mean) / scale
        log_density = weight - (.5 * z.square() + scale.log()).sum(dim=-1) - math.log(2 * math.pi)
        return -torch.logsumexp(log_density, dim=1).mean()

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for game in rng.choice(len(counts), args.batches, p=counts / counts.sum()):
            points = train_points[game][rng.integers(counts[game], size=args.batch_size)]
            value = loss(train, game, points)
            optimizer.zero_grad()
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        total = n = 0
        with torch.inference_mode():
            for game, points in enumerate(valid_points):
                for start in range(0, len(points), 1024):
                    batch = points[start:start + 1024]
                    total += float(loss(validation, game, batch)) * len(batch)
                    n += len(batch)
        metric = total / n
        history.append(metric)
        print(f"{components} components epoch {epoch}: validation NLL {metric:.5f}", flush=True)
        if metric < best:
            best = metric
            torch.save(model.state_dict(), args.output / f"forecast-{components}.pt")
    (args.output / f"forecast-{components}-history.json").write_text(json.dumps(history) + "\n")


def candidates(rows, features, current):
    """Build an approximate observation cell in the observer's current frame."""
    face = rows[:, POSE_START + 2]
    raw_bearing = np.arctan2(features[:, 1], features[:, 2])
    raw_range = features[:, 0] * 60
    lo, hi = rows[:, BOUNDS_START:BOUNDS_START + 2].T
    radius = lo[:, None, None] + (hi - lo)[:, None, None] * np.linspace(.01, .99, 15)
    angle = raw_bearing[:, None, None] + np.deg2rad([-.45, 0., .45])[None, :, None]
    focus = features[:, 14, None, None] * 10
    # This dataset stores sine of the focus-relative bearing, so the exact
    # cosine is known only for zero focus. Keep the stored bounds otherwise.
    valid = (focus != 0) | (np.abs(quantize_distance(radius, radius) - raw_range[:, None, None]) < .02)
    valid = np.broadcast_to(valid, (len(rows), 3, 15)).reshape(-1, 45).copy()
    valid |= ~valid.any(axis=1, keepdims=True)
    world_angle = angle + face[:, None, None]
    rel = (radius[..., None] * np.concatenate((np.cos(world_angle)[..., None],
                                               np.sin(world_angle)[..., None]), axis=-1)).reshape(-1, 45, 2)
    basis = rotate(rel - current[:, None], -(raw_bearing + face)[:, None])
    spread = np.stack((.08 + .20 * (hi - lo), .06 + .006 * raw_range), axis=-1)
    log_measurement = np.where(valid, -.5 * ((basis / spread[:, None])**2).sum(axis=-1), -np.inf)
    return rel, log_measurement, raw_bearing + face, lo, hi


def centroid(grid, log_weight):
    weight = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
    return (weight[..., None] * grid).sum(axis=1) / weight.sum(axis=1, keepdims=True)


def update(grid, log_measurement, base_centroid, current, angle, lo, hi, prior, strength):
    correction = centroid(grid, log_measurement + strength * prior) - base_centroid
    proposal = current + correction
    radius = np.linalg.norm(proposal, axis=1).clip(lo, hi)
    rel = rotate(proposal, -angle)
    bearing = angle + np.arctan2(rel[:, 1], rel[:, 0]).clip(-np.deg2rad(.5), np.deg2rad(.5))
    return radius[:, None] * np.stack((np.cos(bearing), np.sin(bearing)), axis=1)


def compare_batch(games, game, points, tracker, models):
    batch = games.batch(game, points)
    features, rows = batch[:2]
    current = model_batch(tracker, batch)[0].numpy()
    truth = relative_truth(rows)
    predictions = {"visual": rows[:, PRED_START+2:PRED_START+4] - rows[:, POSE_START:POSE_START+2],
                   "tracker": current}
    for family in FAMILIES:
        for strength in STRENGTHS:
            predictions[f"{family}_{strength}"] = current.copy()
    support = forecast_support(games, game, points)
    usable = points[support]
    forecasts = {}
    nll = {}
    coverage = {}
    if len(usable):
        previous_points = usable.copy()
        previous_points[:, 1] -= 1
        previous_batch = games.batch(game, previous_points)
        history, previous = previous_batch[:2]
        anchor = previous[:, POSE_START:POSE_START+2] + model_batch(tracker, previous_batch)[0].numpy()
        face = previous[:, POSE_START+2]
        origin = rows[support, POSE_START:POSE_START+2]
        world_truth = rows[support, 2:4]
        truth_displacement = rotate(world_truth - previous[:, 2:4], -face)
        angle = np.arctan2(history[:, -1, 1], history[:, -1, 2])
        visual_velocity = (previous[:, PRED_START+4:PRED_START+6] -
                           games.data[game][usable[:, 0], usable[:, 1] - 2, PRED_START+2:PRED_START+4])
        measured = rotate(history[:, -1, 5:7], angle)
        cv = np.where(history[:, -1, 7:8] > 0, measured, rotate(visual_velocity, -face))
        cv /= np.maximum(1, np.linalg.norm(cv, axis=1, keepdims=True) / 1.05)
        parameters = {"cv": (cv[:, None], np.full((len(cv), 1, 2), .22), np.zeros((len(cv), 1)))}
        for name, model in models.items():
            parameters[name] = tuple(value.numpy() for value in model(torch.from_numpy(history)))
        grid, observation, angle, lo, hi = candidates(rows[support], features[support, -1], current[support])
        base_centroid = centroid(grid, observation)
        grid_local = rotate(grid + origin[:, None] - anchor[:, None], -face[:, None])
        anchor_error = (.10 + .009 * history[:, -1, 0] * 60)[:, None, None]
        for name, (mean, scale, weight) in parameters.items():
            forecast = anchor + rotate((np.exp(weight)[..., None] * mean).sum(axis=1), face)
            forecasts[name] = np.linalg.norm(forecast - world_truth, axis=-1)
            z = (truth_displacement[:, None] - mean) / scale
            log_density = weight - (.5 * z**2 + np.log(scale)).sum(axis=-1) - math.log(2 * math.pi)
            nll[name] = -np.logaddexp.reduce(log_density, axis=1)
            cdf = .5 * (1 + torch.erf(torch.from_numpy(z / math.sqrt(2))).numpy())
            marginal = (np.exp(weight)[..., None] * cdf).sum(axis=1)
            coverage[name] = (marginal >= .1) & (marginal <= .9)
            sigma = np.sqrt(scale**2 + anchor_error**2)
            z = (grid_local[:, :, None] - mean[:, None]) / sigma[:, None]
            log_density = weight[:, None] - (.5 * z**2 + np.log(sigma[:, None])).sum(axis=-1)
            prior = np.logaddexp.reduce(log_density, axis=-1)
            for strength in STRENGTHS:
                predictions[f"{name}_{strength}"][support] = update(
                    grid, observation, base_centroid, current[support], angle, lo, hi, prior, strength)
    errors = {key: np.linalg.norm(value - truth, axis=1) for key, value in predictions.items()}
    return errors, forecasts, nll, support, rows[:, 0], rows[:, 4] == 1, coverage


def summarize(values):
    return {"n": len(values), "mean": float(values.mean(dtype=np.float64)),
            "p90": float(np.quantile(values, .9))}


@torch.inference_mode()
def evaluate(args, split, selected=None):
    games = ContextGames(args.dataset, args.sidecars, split, 24 if split == "validation" else 1)
    checkpoint = torch.load(args.tracker, weights_only=True)
    tracker = ContextGRU(checkpoint["mode"]).eval()
    tracker.load_state_dict(checkpoint["model"])
    models = {}
    for name, components in (("single", 1), ("mixture", 3)):
        model = Forecast(components).eval()
        model.load_state_dict(torch.load(args.output / f"forecast-{components}.pt", weights_only=True))
        models[name] = model
    records = []
    for game, points in enumerate(games.points):
        for start in range(0, len(points), 1024):
            records.append(compare_batch(games, game, points[start:start + 1024], tracker, models))
        print(f"{split}: {game + 1}/{len(games.points)} games", flush=True)
    def combine(index):
        keys = next(record[index] for record in records if record[index])
        return {key: np.concatenate([r[index][key] for r in records if key in r[index]]) for key in keys}
    errors, forecast, nll = (combine(i) for i in range(3))
    support, distance, motion = (np.concatenate([r[i] for r in records]) for i in range(3, 6))
    if not all(np.isfinite(x).all() for collection in (errors, forecast, nll) for x in collection.values()):
        raise ValueError("non-finite forecast metrics")
    if selected is None:
        selected = {family: min((f"{family}_{s}" for s in STRENGTHS),
                                key=lambda key: errors[key].mean()) for family in FAMILIES}
        selected["winner"] = min(("tracker", *selected.values()), key=lambda key: errors[key].mean())
    groups = {"all": np.ones(len(support), bool), "forecast_eligible": support,
              "motion_available": motion, "motion_missing": ~motion}
    bins = np.searchsorted([10, 20, 30, 40, 60], distance, side="right")
    groups.update({name: bins == i for i, name in enumerate(DISTANCE)})
    return {"selected": selected, "tracker": checkpoint["mode"],
            "forecast": {key: summarize(value) for key, value in forecast.items()},
            "displacement_nll": {key: float(value.mean()) for key, value in nll.items()},
            "displacement_marginal_80_coverage": {key: value.mean(axis=0).tolist()
                                                  for key, value in combine(6).items()},
            "scores": {name: {key: summarize(value[mask]) for key, value in errors.items()}
                       for name, mask in groups.items() if mask.any()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "sidecars", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--train", type=int, choices=(1, 3))
    parser.add_argument("--tracker", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.train:
        train(args, args.train)
    elif args.tracker:
        validation = evaluate(args, "validation")
        print("Selected on validation:", validation["selected"], flush=True)
        test = evaluate(args, "test", validation["selected"])
        (args.output / "metrics.json").write_text(json.dumps({"validation": validation, "test": test}, indent=2) + "\n")
    else:
        parser.error("use --train or supply --tracker for evaluation")
