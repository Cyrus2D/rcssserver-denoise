#!/usr/bin/env python3
"""Build causal player tracks and baseline predictions from Exp3 logs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cycle", "stopped"]
POSE = ["est_pos_x", "est_pos_y", "est_face"]
VISUAL = ["ls_pos_x", "ls_pos_y"]
VELOCITY = ["ls_vel_x", "ls_vel_y"]
AGENT = KEY + POSE + ["focus_dist", "focus_dir", "decision_made", "game_mode", "see_used"]
OBJECT = KEY + VISUAL + VELOCITY + [
    "object_id", "seen", "match_method", "dist", "dir", "dist_chg", "dir_chg",
    "truth_valid", "tru_on_field", "true_dist", "true_dir", "tru_pos_x", "tru_pos_y",
]
FEATURES = 16
PREDICTIONS = ("raw", "visual", "constant_velocity", "blend", "quantized")
PRED_START, POSE_START, BOUNDS_START, AGE, FEATURE_START = 5, 15, 18, 21, 22
FIELDS = FEATURE_START + FEATURES


def norm(vector):
    return np.sqrt(np.sum(vector * vector, axis=-1))


def cap_velocity(velocity):
    return velocity / np.maximum(1, norm(velocity)[..., None] / 1.5)


def zero_focus_bins():
    result = {}
    for k in range(-230, 65):
        observed = int(np.rint(10 * np.exp(k / 10)))
        lower, upper = np.maximum(0, np.exp((k + np.array([-0.5, 0.5])) / 10) - 1e-10)
        if observed in result:
            lower = result[observed][0]
        result[observed] = lower, upper
    result[0] = 0, result[0][1]
    return result


BINS = zero_focus_bins()


def range_bounds(distance, direction, focus_dist, focus_dir):
    lo, hi = np.zeros_like(distance), np.zeros_like(distance)
    index = np.rint(np.nan_to_num(distance) * 10).astype(int)
    valid = np.isfinite(distance)
    zero_focus = (focus_dist == 0)[:, None] & valid
    if zero_focus.any():
        observed, inverse = np.unique(index[zero_focus], return_inverse=True)
        bounds = np.array([BINS.get(i, (np.nan, np.nan)) for i in observed])
        lo[zero_focus], hi[zero_focus] = bounds[inverse].T
    rows, cols = np.where(valid & ~zero_focus)
    if len(rows):
        d, fd = distance[rows, cols], focus_dist[rows]
        radius = .065 * (d + fd) + .15
        samples = np.maximum(0, d[:, None] + radius[:, None] * np.linspace(-1, 1, 81))
        cos = np.cos(direction[rows, cols] - focus_dir[rows])[:, None]
        focus_range = np.sqrt(np.maximum(0, samples**2 + fd[:, None]**2 - 2 * samples * fd[:, None] * cos))
        quantize = lambda x: np.exp(np.rint(np.log(x + 1e-10) * 10) / 10)
        measured = np.maximum(0, (samples - focus_range + quantize(samples) + quantize(focus_range)) / 2)
        matches = np.rint(measured * 10) == index[rows, cols, None]
        lo[rows, cols] = np.min(np.where(matches, samples, np.inf), axis=1)
        hi[rows, cols] = np.max(np.where(matches, samples, -np.inf), axis=1)
    fallback = valid & (~np.isfinite(lo) | ~np.isfinite(hi))
    lo[fallback] = np.maximum(0, distance[fallback] * .94 - .05)
    hi[fallback] = distance[fallback] * 1.06 + .05
    return lo, hi, int(fallback.sum())


def make_tracks(path):
    meta = json.loads((path / "meta.json").read_text())
    config, server = meta["config"], meta["server_param"]
    if (not meta["final"] or meta["status"] != "ok" or meta["schema_version"] != 2
            or config["gaussian_see"] or config["use_fullstate"] or not config["debug_fullstate"]
            or server["quantize_step"] != "0.1" or server["see_noise_free"] != "false"):
        raise ValueError(f"{path}: expected complete quantized logs with reference truth")
    a = pd.read_csv(path / "agent_cycle.csv.gz", usecols=AGENT)
    a = a[a["game_mode"].eq("play_on") & a["decision_made"].eq(1) & a["stopped"].eq(0)
          & a["see_used"].eq(1) & a[POSE].notna().all(axis=1)].set_index("cycle", verify_integrity=True)
    a = a.sort_index()
    o = pd.read_csv(path / "objects.csv.gz", usecols=OBJECT, dtype={"match_method": str})
    o = o[o["object_id"].ne(0) & o["stopped"].eq(0) & o["cycle"].isin(a.index)]
    ids = sorted(o["object_id"].unique())
    if len(ids) != 21:
        raise ValueError(f"{path}: expected 21 player IDs")
    o = o.set_index(["cycle", "object_id"], verify_integrity=True)
    o = o.reindex(pd.MultiIndex.from_product([a.index, ids]))
    def values(columns):
        return o[columns].to_numpy(dtype=np.float32).reshape(len(a), len(ids), len(columns))

    raw = values(["dist", "dir", "dist_chg", "dir_chg"])
    visual, velocity = values(VISUAL), values(VELOCITY)
    truth = values(["true_dist", "true_dir", "tru_pos_x", "tru_pos_y"])
    truth[values(["truth_valid", "tru_on_field"]).min(axis=-1) != 1] = np.nan
    seen = (o["seen"].eq(1) & o["match_method"].eq("unum")).to_numpy().reshape(len(a), len(ids))
    valid = seen & np.isfinite(raw[..., :2]).all(axis=-1) & np.isfinite(visual).all(axis=-1)
    raw[~valid] = np.nan
    self_pos = a[POSE[:2]].to_numpy(dtype=np.float32)
    self_face = np.deg2rad(a[POSE[2]].to_numpy(dtype=np.float32))
    focus_dist = a["focus_dist"].fillna(0).to_numpy()
    focus_dir = np.deg2rad(a["focus_dir"].fillna(0).to_numpy())
    direction = np.deg2rad(raw[..., 1])
    angle = direction + self_face[:, None]
    unit = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    side = np.stack((-unit[..., 1], unit[..., 0]), axis=-1)
    lo, hi, fallback = range_bounds(raw[..., 0], direction, focus_dist, focus_dir)
    raw_pos = self_pos[:, None, :] + raw[..., :1] * unit
    has_vel = valid & np.isfinite(velocity).all(axis=-1)
    cycles = a.index.to_numpy()
    data = np.full((len(a), len(ids), FIELDS), np.nan, np.float32)
    data[..., :4] = truth
    data[..., 4] = has_vel
    data[..., 5:7] = raw_pos
    data[..., 7:9] = visual
    data[..., 15:17] = self_pos[:, None, :]
    data[..., 17] = self_face[:, None]
    ages = np.zeros(len(ids), dtype=int)
    speed = np.zeros((len(ids), 2), dtype=np.float32)
    filtered = np.zeros_like(speed)
    old_visual = np.zeros_like(speed)
    visual_speed = np.zeros_like(speed)
    gaps = np.r_[True, np.diff(cycles) != 1]
    for t in range(len(a)):
        present = valid[t]
        continuing = present & (ages > 0) & ~gaps[t]
        ages = np.where(present, np.where(continuing, ages + 1, 1), 0)
        origin = self_pos[t]
        direction_t = unit[t]
        prior = filtered + speed
        prior = np.where(continuing[:, None], prior, visual[t])
        radial_prior = np.sum((prior - origin) * direction_t, axis=-1)
        center = (lo[t] + hi[t]) / 2
        width = hi[t] - lo[t]
        clipped = np.clip(radial_prior, lo[t], hi[t])
        weight = width**2 / (width**2 + .18**2 * 12)
        estimate = np.where(continuing, center + weight * (clipped - center), center)
        position = origin + estimate[:, None] * direction_t
        movement = np.where(continuing[:, None], position - filtered, 0)
        measured_velocity = np.where(has_vel[t, :, None], velocity[t], movement)
        speed = cap_velocity(np.where(continuing[:, None], .65 * speed + .35 * measured_velocity,
                                      np.where(has_vel[t, :, None], velocity[t], 0)))
        # Independent CV uses previous visual position and its own smoothed velocity.
        delta_visual = np.where(continuing[:, None], visual[t] - old_visual, 0)
        cv = np.where(continuing[:, None], old_visual + visual_speed, visual[t])
        blend = .65 * visual[t] + .35 * cv
        visual_speed = cap_velocity(np.where(
            continuing[:, None], .65 * visual_speed + .35 * np.where(has_vel[t, :, None], velocity[t], delta_visual),
            np.where(has_vel[t, :, None], velocity[t], 0),
        ))
        own_move = self_pos[t] - self_pos[t - 1] if not gaps[t] else np.zeros(2)
        vel = np.nan_to_num(velocity[t], nan=0)
        features = np.column_stack((
            raw[t, :, 0] / 60, np.sin(direction[t]), np.cos(direction[t]),
            direction_t @ own_move, side[t] @ own_move,
            np.sum(vel * direction_t, axis=-1), np.sum(vel * side[t], axis=-1), has_vel[t],
            np.nan_to_num(raw[t, :, 2], nan=0), np.nan_to_num(raw[t, :, 3], nan=0) / 10,
            (estimate - center) / np.maximum(.01, width), width / 3,
            np.clip(radial_prior - center, -8, 8) / 3, np.minimum(ages, 10) / 10,
            np.full(len(ids), focus_dist[t] / 10), np.sin(direction[t] - focus_dir[t]),
        ))
        data[t, :, 9:11] = cv
        data[t, :, 11:13] = blend
        data[t, :, 13:15] = position
        data[t, :, 18:21] = np.column_stack((lo[t], hi[t], estimate))
        data[t, :, 21] = ages
        data[t, :, 22:] = np.where(present[:, None], features, np.nan)
        filtered = np.where(present[:, None], position, 0)
        old_visual = np.where(present[:, None], visual[t], 0)
    return np.ascontiguousarray(data.transpose(1, 0, 2)), fallback


def build_game(task):
    game, split, output, observer = task
    parts, fallback = [], 0
    for side in "lr":
        data, misses = make_tracks(game / f"{side}_{observer}")
        parts.append(data)
        fallback += misses
    steps = max(part.shape[1] for part in parts)
    result = np.concatenate([np.pad(part, ((0, 0), (0, steps - part.shape[1]), (0, 0)),
                                    constant_values=np.nan) for part in parts])
    np.save(output / f"{game.name}.npy", result, allow_pickle=False)
    return {"name": game.name, "split": split, "observer_unum": observer,
            "steps": result.shape[1], "range_fallback": fallback}


def build(source, output, workers, limit):
    games = sorted(path for path in source.iterdir() if path.is_dir() and (path / "l_1/meta.json").is_file())
    if limit:
        games = games[:limit]
    if len(games) < 5:
        raise ValueError("need at least five games")
    rng = np.random.default_rng(42)
    rng.shuffle(games)
    holdout = max(1, len(games) // 10)
    splits = ["test"] * holdout + ["validation"] * holdout + ["train"] * (len(games) - 2 * holdout)
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(game, split, output, int(rng.integers(1, 12))) for game, split in zip(games, splits)]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, record in enumerate(pool.map(build_game, jobs), 1):
            records.append(record)
            if index == 1 or index % 10 == 0 or index == len(games):
                print(f"Built {index}/{len(games)} games", flush=True)
    (output / "manifest.json").write_text(json.dumps({"version": 1, "games": records}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-games", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_games is not None and args.limit_games < 5):
        parser.error("workers must be positive and limit-games must be at least 5")
    build(args.source, args.output, args.workers, args.limit_games)
