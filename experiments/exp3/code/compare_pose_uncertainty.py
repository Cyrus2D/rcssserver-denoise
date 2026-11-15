#!/usr/bin/env python3
"""Ablate observer-pose uncertainty in the fresh-observation likelihood."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_sequences import range_bounds
from compare_directional import bounded_velocity, public_speed, step13
from replay_normal import quantize_distance, unit
from train_normal import Store, errors, load_models, observer

FIELDS = ["est_pos_err_x", "est_pos_err_y", "est_face_err"]
METHODS = ("step13", "fixed", "pose_half", "pose_full")
RANGES = ("under_10m", "10_to_25m", "at_least_25m")


def pose_errors(path, cycles):
    log = pd.read_csv(path, usecols=["cycle", "stopped", *FIELDS])
    log = log[log.stopped.eq(0) & log.cycle.isin(cycles)]
    return log.set_index("cycle", verify_integrity=True).reindex(cycles)[FIELDS].to_numpy()


def candidates(raw, pose):
    measured, angle = raw.T
    bearing = np.deg2rad(angle)
    focus = pose[:, 5]
    focus_bearing = np.deg2rad(pose[:, 6])
    lo, hi, _ = range_bounds(measured[:, None], bearing[:, None], focus, focus_bearing)
    radius = lo + (hi - lo) * np.linspace(.001, .999, 15)
    rays = bearing[:, None] + np.deg2rad([-.45, 0, .45])
    focus_dist = np.sqrt(np.maximum(0, radius[:, None] ** 2 + focus[:, None, None] ** 2
                         - 2 * radius[:, None] * focus[:, None, None]
                         * np.cos(rays - focus_bearing[:, None])[..., None]))
    valid = np.abs(quantize_distance(radius[:, None], focus_dist) - measured[:, None, None]) < .02
    valid |= ~valid.any(axis=(1, 2))[:, None, None]
    offsets = radius[:, None, :, None] * unit(rays + np.deg2rad(pose[:, 2, None]))[:, :, None, :]
    return offsets.reshape(len(raw), 45, 2), valid.reshape(len(raw), 45)


def estimate(raw, pose, uncertainty, prior):
    offsets, valid = candidates(raw, pose)
    points = pose[:, None, :2] + offsets
    residual = points - prior[:, None]
    dx, dy = residual[..., 0], residual[..., 1]
    jx, jy = -offsets[..., 1], offsets[..., 0]
    # A first-order yaw perturbation rotates the candidate around the observer.
    base = (.35 + .015 * raw[:, 0, None]) ** 2
    out = []
    for scale in (0, .5, 1):
        sx, sy, yaw = (scale * uncertainty).T[:, :, None] ** 2
        a, b, c = base + sx + jx**2 * yaw, jx * jy * yaw, base + sy + jy**2 * yaw
        det = a * c - b**2
        cost = (c * dx**2 - 2 * b * dx * dy + a * dy**2) / det + np.log(det)
        weights = np.where(valid, -.5 * cost, -np.inf)
        weights = np.exp(weights - weights.max(axis=1, keepdims=True))
        out.append((points * weights[..., None]).sum(axis=1) / weights.sum(axis=1, keepdims=True))
    return out


def evaluate(args, split, selection=None):
    store = Store(args.associated, args.events, split)
    models, calibration = load_models(args.runs)
    config = json.loads((args.runs / "audited-validation.json").read_text())["config"]
    games = json.loads((args.events / "manifest.json").read_text())["games"]
    numbers = {game["name"]: game["observer_unum"] for game in games}
    records = []
    for i, (name, rows, cycles) in enumerate(store.games):
        side = "lr"[i % 2]
        number = numbers[name]
        with np.load(args.events / f"{name}-{side}.npz", allow_pickle=False) as events:
            pose = events["pose"]
            if not np.array_equal(events["cycle"], cycles):
                raise ValueError(f"{name}-{side}: misaligned pose")
        unc = pose_errors(args.source / name / f"{side}_{number}" / "agent_cycle.csv.gz", cycles)
        unc[:, 2] = np.deg2rad(unc[:, 2])
        known_unc = np.isfinite(unc).all(axis=1)
        unc = np.clip(np.abs(unc), 0, [3, 3, np.deg2rad(15)])
        reference = step13(store, i, models, calibration, config)
        mask = np.isfinite(rows[..., :6]).all(axis=-1)
        data = rows[mask]
        support = mask & (rows[..., 18] > 0) & (rows[..., 12] > 0) & known_unc[None]
        support[:, 0] = False
        tr, t = np.nonzero(support)
        kin = np.load(args.kinematics / f"{name}-{side}.npy", mmap_mode="r")[tr, t - 1]
        ok = ((cycles[t] - cycles[t - 1] == 1) & np.isfinite(kin).all(axis=1) & (kin[:, 2] <= 6)
              & np.isfinite(reference[tr, t - 1]).all(axis=1))
        tr, t = tr[ok], t[ok]
        support.fill(False)
        support[tr, t] = True
        raw = rows[tr, t, 12:14]
        prior = reference[tr, t - 1] + bounded_velocity(kin[ok, :2], public_speed(args.source, name, number, side))
        candidates_world = estimate(raw, pose[t], unc[t], prior)
        bucket = np.full(rows.shape[:2], -1, np.int8)
        bucket[tr, t] = np.searchsorted([10, 25], raw[:, 0], side="right")
        score = {"step13": errors(data, observer(data, reference[mask]))[0],
                 "librcsc": errors(data, observer(data, data[:, 4:6]))[0]}
        for method, estimate_world in zip(METHODS[1:], candidates_world):
            pred = reference[mask]
            pred[support[mask]] = estimate_world
            score[method] = errors(data, observer(data, pred))[0]
        records.append((bucket[mask], score))
        print(f"{split}: {i + 1}/{len(store.games)} observers", flush=True)
    bins = np.concatenate([bucket for bucket, _ in records])
    scores = {method: np.concatenate([score[method] for _, score in records])
              for method in ("librcsc", *METHODS)}
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError("invalid estimate")
    selected = scores["step13"].copy()
    if selection is None:
        selection = [min(METHODS, key=lambda method: scores[method][bins == b].mean())
                     if (bins == b).sum() >= 100 else "step13" for b in range(len(RANGES))]
    for b, method in enumerate(selection):
        selected[bins == b] = scores[method][bins == b]
    scores["selected"] = selected
    groups = {"all": np.ones(len(bins), bool), "supported": bins >= 0,
              **{name: bins == b for b, name in enumerate(RANGES)}}
    results = {}
    for name, mask in groups.items():
        if mask.any():
            results[name] = {m: {"n": int(mask.sum()), "mean_m": float(v[mask].mean(dtype=np.float64)),
                                 "p90_m": float(np.quantile(v[mask], .9))}
                             for m, v in scores.items()}
    return {"selection": dict(zip(RANGES, selection)), "metrics": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "events", "associated", "kinematics", "runs", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    validation = evaluate(args, "validation")
    print("Selected on validation:", validation["selection"], flush=True)
    test = evaluate(args, "test", list(validation["selection"].values()))
    args.output.write_text(json.dumps({"validation": validation, "test": test}, indent=2) + "\n")
