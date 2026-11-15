#!/usr/bin/env python3
"""Build causal reachable-region and context sidecars for the Exp3 sequences."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_sequences import AGE, AGENT, KEY, POSE, FEATURE_START
from train_range import measurement_geometry

AXES = np.array([[1, 0], [0, 1], [2**-.5, 2**-.5], [2**-.5, -2**-.5]], np.float32)
ANGLES = np.deg2rad(np.linspace(-.5, .5, 5, dtype=np.float32))
CONTEXT_FIELDS = ["trk_pos_x", "trk_pos_y", "trk_vel_x", "trk_vel_y", "trk_pos_count"]
TRACK_FEATURES = 7
CONTEXT_SIZE = 5 * TRACK_FEATURES


def unit(angle):
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def intersect(sequence, speed_limit):
    """Clip sampled sector rays by expanded support strips from the previous sector."""
    tracks, steps, fields = sequence.shape
    origin, _, angle, original, bounded_angle, bounded = measurement_geometry(
        sequence.reshape(-1, fields), sequence[..., FEATURE_START:].reshape(tracks * steps, -1))
    origins = origin.reshape(tracks, steps, 2)
    angles = angle.reshape(tracks, steps)
    center = bounded_angle.reshape(tracks, steps)
    bounds = original.reshape(tracks, steps, 3)
    base = bounded[:, 2].reshape(tracks, steps)
    valid = sequence[..., AGE] > 0
    contiguous = sequence[..., AGE] > 1
    sidecar = np.full((tracks, steps, 4), np.nan, np.float32)
    prior_lo = np.zeros((tracks, len(AXES)), np.float32)
    prior_hi = np.zeros_like(prior_lo)
    fallbacks = narrowed = continued = 0
    for t in range(steps):
        ray = np.concatenate((angles[:, t, None] + ANGLES, center[:, t, None]), axis=1)
        ray_units = unit(ray)
        projections = ray_units @ AXES.T
        positions = origins[:, t] @ AXES.T
        radial_low, radial_high = bounds[:, t, :2].T
        near_zero = np.abs(projections) < 1e-7
        slope = np.where(near_zero, 1, projections)
        # Approximate the two observer-pose uncertainties using a translation
        # allowance and distance-dependent bearing allowance at each endpoint.
        prior_range = base[:, t - 1] if t else base[:, t]
        motion = speed_limit + .12 + .0044 * (prior_range + base[:, t])
        lower = prior_lo - motion[:, None] - positions
        upper = prior_hi + motion[:, None] - positions
        lower_ray = lower[:, None] / slope
        upper_ray = upper[:, None] / slope
        parallel_inside = (lower[:, None] <= 0) & (upper[:, None] >= 0)
        reach_lo = np.where(near_zero, np.where(parallel_inside, -np.inf, np.inf),
                            np.minimum(lower_ray, upper_ray)).max(axis=-1)
        reach_hi = np.where(near_zero, np.where(parallel_inside, np.inf, -np.inf),
                            np.maximum(lower_ray, upper_ray)).min(axis=-1)
        low = np.maximum(radial_low[:, None], np.where(contiguous[:, t, None], reach_lo, -np.inf))
        high = np.minimum(radial_high[:, None], np.where(contiguous[:, t, None], reach_hi, np.inf))
        feasible = low <= high
        empty = valid[:, t] & ~feasible.any(axis=-1)
        low = np.where(empty[:, None], radial_low[:, None], low)
        high = np.where(empty[:, None], radial_high[:, None], high)
        feasible = low <= high

        # Use the feasible point closest to the bounded blend. Keep all feasible
        # rays in the propagated envelope, including those not selected here.
        vector = base[:, t, None] * unit(center[:, t])
        radial = np.clip((ray_units * vector[:, None]).sum(axis=-1), low, high)
        delta = radial[..., None] * ray_units - vector[:, None]
        cost = np.where(feasible, (delta * delta).sum(axis=-1), np.inf)
        chosen = cost.argmin(axis=-1)
        idx = np.arange(tracks)
        selected_low, selected_high = low[idx, chosen], high[idx, chosen]
        sidecar[:, t] = np.stack((selected_low, selected_high, ray[idx, chosen], radial[idx, chosen]), axis=-1)
        low_proj = low[..., None] * projections
        high_proj = high[..., None] * projections
        # A quarter-degree ray spacing gives a half-step sampling allowance.
        padding = radial_high[:, None] * np.deg2rad(.125)
        prior_lo = np.where(feasible[..., None], np.minimum(low_proj, high_proj), np.inf).min(axis=1) + positions - padding
        prior_hi = np.where(feasible[..., None], np.maximum(low_proj, high_proj), -np.inf).max(axis=1) + positions + padding
        prior_lo = np.where(valid[:, t, None], prior_lo, 0)
        prior_hi = np.where(valid[:, t, None], prior_hi, 0)
        fallbacks += int(empty.sum())
        continued += int(contiguous[:, t].sum())
        narrowed += int((contiguous[:, t] & (selected_high - selected_low + .01 < radial_high - radial_low)).sum())
    return sidecar, {"continued": continued, "narrowed": narrowed, "fallback": fallbacks}


def context_features(observer, sequence):
    agent = pd.read_csv(observer / "agent_cycle.csv.gz", usecols=AGENT)
    agent = agent[agent.game_mode.eq("play_on") & agent.decision_made.eq(1) & agent.stopped.eq(0)
                  & agent.see_used.eq(1) & agent[POSE].notna().all(axis=1)].sort_values("cycle")
    objects = pd.read_csv(observer / "objects.csv.gz", usecols=KEY + ["object_id", "dist", "track_match"] + CONTEXT_FIELDS,
                          dtype={"track_match": str})
    objects = objects[objects.stopped.eq(0) & objects.cycle.isin(agent.cycle)]
    ids = sorted(objects.object_id.unique())
    if len(ids) != 22 or ids[0] != 0:
        raise ValueError(f"{observer}: expected one ball and 21 players")
    objects = objects.set_index(["cycle", "object_id"], verify_integrity=True)
    objects = objects.reindex(pd.MultiIndex.from_product([agent.cycle, ids], names=["cycle", "object_id"]))
    identified = (objects.index.get_level_values("object_id") == 0) | objects.track_match.eq("unum")
    values = objects[CONTEXT_FIELDS].where(identified, np.nan).to_numpy(np.float32).reshape(len(agent), len(ids), -1)
    # Check the cycle/player ordering against the original sequence features.
    raw = objects.dist.to_numpy().reshape(len(agent), len(ids))[:, 1:].T
    check = sequence[:, :len(agent), FEATURE_START] * 60
    observed = np.isfinite(check)
    if not np.allclose(raw[observed], check[observed], atol=.002, rtol=0):
        raise ValueError(f"{observer}: player/cycle order differs from sequence data")
    output = np.zeros((*sequence.shape[:2], CONTEXT_SIZE), np.float16)
    same_side = (np.array(ids[1:])[:, None] <= 11) == (np.array(ids[1:])[None, :] <= 11)
    same_side &= ~np.eye(21, dtype=bool)
    opposite = (np.array(ids[1:])[:, None] <= 11) != (np.array(ids[1:])[None, :] <= 11)

    for start in range(0, len(agent), 256):
        end = min(start + 256, len(agent))
        tracks = values[start:end]
        target_pos = tracks[:, 1:, :2]
        target_vel = tracks[:, 1:, 2:4]
        displacement = tracks[:, None, :, :2] - target_pos[:, :, None]
        vel = tracks[:, None, :, 2:4] - target_vel[:, :, None]
        age = np.broadcast_to(tracks[:, None, :, 4], displacement.shape[:-1])
        distance = (displacement[:, :, 1:] ** 2).sum(axis=-1)
        distance = np.where(np.isfinite(distance), distance, np.inf)
        neighbors = []
        for allowed in (same_side, opposite):
            masked = np.where(allowed, distance, np.inf)
            nearest = np.argsort(masked, axis=-1, kind="stable")[..., :2]
            neighbors.append(np.where(np.take_along_axis(masked, nearest, axis=-1) < np.inf,
                                      nearest + 1, -1))
        index = np.concatenate((np.zeros((*distance.shape[:2], 1), dtype=int), *neighbors), axis=-1)
        d = np.take_along_axis(displacement, np.maximum(index, 0)[..., None], axis=2)
        v = np.take_along_axis(vel, np.maximum(index, 0)[..., None], axis=2)
        ages = np.take_along_axis(age, np.maximum(index, 0), axis=2)
        has_pos = (index >= 0) & np.isfinite(d).all(axis=-1)
        has_vel = has_pos & np.isfinite(v).all(axis=-1)
        # Rotate context vectors into the observer's looking frame at that cycle.
        facing = sequence[:, start:end, 17].T[:, :, None]
        def rotate(vector):
            x, y = vector[..., 0], vector[..., 1]
            return np.stack((x * np.cos(facing) + y * np.sin(facing),
                             -x * np.sin(facing) + y * np.cos(facing)), axis=-1)
        scaled = np.concatenate((
            np.where(has_pos[..., None], rotate(d) / 20, 0),
            np.where(has_vel[..., None], rotate(v) / 2, 0),
            np.minimum(10, np.where(has_pos & np.isfinite(ages), ages, 10))[..., None] / 10,
            has_pos[..., None], has_vel[..., None],
        ), axis=-1)
        output[:, start:end] = scaled.reshape(end - start, 21, CONTEXT_SIZE).transpose(1, 0, 2)
    return output, (np.array(ids[1:]) - 1) % 11 + 1


def build_game(args):
    game, sequence_root, source, output = args
    name, number = game["name"], game["observer_unum"]
    sequence = np.load(sequence_root / f"{name}.npy", mmap_mode="r")
    folder = source / name
    meta = json.loads((folder / f"l_{number}/meta.json").read_text())
    speed = max(float(meta["server_param"]["player_speed_max"]),
                *(float(p["player_speed_max"]) for p in meta["player_types"]))
    geometry, counts = intersect(sequence, speed)
    context, unums = zip(*(context_features(folder / f"{side}_{number}", sequence[21*i:21*(i+1)])
                           for i, side in enumerate("lr")))
    np.save(output / f"{name}-geometry.npy", geometry, allow_pickle=False)
    np.save(output / f"{name}-context.npy", np.concatenate(context), allow_pickle=False)
    np.save(output / f"{name}-unums.npy", np.concatenate(unums), allow_pickle=False)
    return name, counts


def build(dataset, source, output, workers, limit):
    manifest = json.loads((dataset / "manifest.json").read_text())
    games = manifest["games"]
    if limit:
        games = [g for g in games if g["split"] == "train"][:limit]
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(g, dataset, source, output) for g in games]
    result = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (name, counts) in enumerate(pool.map(build_game, jobs), 1):
            result[name] = counts
            if index == 1 or index % 10 == 0 or index == len(games):
                print(f"Built intersections and context for {index}/{len(games)} games", flush=True)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build(args.dataset, args.source, args.output, args.workers, args.limit)
