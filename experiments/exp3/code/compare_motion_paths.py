#!/usr/bin/env python3
"""Compare stale motion hypotheses, learned trust and hindsight smoothing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_directional import bounded_velocity, public_speed, step13
from replay_normal import MAX_AGE
from train_normal import Store, errors, load_models, observer

ONLINE = ("step13", "coast", "stop", "trajectory", "mixture")
GROUPS = ((1, 2), (3, 5), (6, MAX_AGE))


def paths(rows, cycles, reference, kin, speed_limit):
    fresh = (rows[..., 18] > 0) & np.isfinite(reference).all(axis=-1)
    index = np.arange(len(cycles))
    segment = np.cumsum(np.r_[0, np.diff(cycles) != 1])
    last = np.maximum.accumulate(np.where(fresh, index, -1), axis=1)
    following = np.minimum.accumulate(np.where(fresh, index, len(index))[:, ::-1], axis=1)[:, ::-1]
    target, current = np.nonzero((rows[..., 17] >= 1) & (rows[..., 17] <= MAX_AGE)
                                & (last >= 1) & np.isfinite(rows[..., :6]).all(axis=-1))
    start = last[target, current]
    earlier = last[target, start - 1]
    keep = (earlier >= 0) & (segment[current] == segment[np.maximum(earlier, 0)])
    target, current, start, earlier = (x[keep] for x in (target, current, start, earlier))
    dt = cycles[start] - cycles[earlier]
    keep = dt <= MAX_AGE
    target, current, start, earlier, dt = (x[keep] for x in (target, current, start, earlier, dt))
    age = cycles[current] - cycles[start]
    anchor = reference[target, start]
    older = reference[target, earlier]
    velocity = (anchor - older) / dt[:, None]
    prev_earlier = last[target, np.maximum(earlier - 1, 0)]
    third = (earlier > 0) & (prev_earlier >= 0) & (segment[np.maximum(prev_earlier, 0)] == segment[start])
    third &= cycles[start] - cycles[np.maximum(prev_earlier, 0)] <= MAX_AGE
    third_pos = reference[target[third], prev_earlier[third]]
    long_velocity = ((anchor[third] - third_pos)
                     / (cycles[start[third]] - cycles[prev_earlier[third]])[:, None])
    recent = velocity.copy()
    velocity[third] = .55 * velocity[third] + .45 * long_velocity
    velocity = bounded_velocity(velocity, speed_limit)

    wm = kin[target, start]
    have_wm = np.isfinite(wm).all(axis=-1) & (wm[:, 2] <= 6)
    wm_vel = bounded_velocity(np.where(have_wm[:, None], wm[:, :2], velocity), speed_limit)
    coast = anchor + ((1 - .4**age) / .6)[:, None] * wm_vel
    trajectory = anchor + age[:, None] * velocity
    baseline = reference[target, current]
    # Maintain separate stop, coasting and straight-motion paths. Their
    # mixture favors agreement with observed displacement before the gap.
    past_speed = np.linalg.norm(velocity, axis=-1)
    disagreement = np.linalg.norm(recent - wm_vel, axis=-1)
    logits = np.stack((-past_speed / .3, -disagreement / .4, -dt / 10), axis=-1)
    weights = np.exp(logits)
    mixture = (np.stack((anchor, coast, trajectory), axis=1) * weights[..., None]).sum(axis=1)
    mixture /= weights.sum(axis=1, keepdims=True)

    end = following[target, current]
    bounded_end = np.minimum(end, len(cycles) - 1)
    hindsight = (end < len(cycles)) & (segment[bounded_end] == segment[current])
    hindsight &= (cycles[bounded_end] - cycles[start]) <= MAX_AGE
    smoothed = baseline.copy()
    start_time = cycles[start[hindsight]]
    fraction = (cycles[current[hindsight]] - start_time) / (cycles[end[hindsight]] - start_time)
    smoothed[hindsight] = anchor[hindsight] + fraction[:, None] * (
        reference[target[hindsight], end[hindsight]] - anchor[hindsight])

    data = rows[target, current]
    distance = np.linalg.norm(baseline - data[:, 14:16], axis=-1)
    separation = np.linalg.norm(mixture - baseline, axis=-1)
    x = np.stack((np.ones(len(age)), age / 12, np.minimum(distance, 70) / 70,
                  past_speed, disagreement, np.minimum(separation, 8) / 8,
                  have_wm, third, (age / 12) * past_speed), axis=1)
    return target, current, hindsight, x, (baseline, coast, anchor, trajectory, mixture, smoothed)


def collect(args, split):
    store = Store(args.associated, args.events, split)
    models, calibration = load_models(args.runs)
    config = json.loads((args.runs / "audited-validation.json").read_text())["config"]
    games = json.loads((args.events / "manifest.json").read_text())["games"]
    numbers = {game["name"]: game["observer_unum"] for game in games}
    records = []
    for i, (name, rows, cycles) in enumerate(store.games):
        side = "lr"[i % 2]
        kin = np.load(args.kinematics / f"{name}-{side}.npy", mmap_mode="r")
        reference = step13(store, i, models, calibration, config)
        tr, t, hindsight, features, preds = paths(
            rows, cycles, reference, kin, public_speed(args.source, name, numbers[name], side),
        )
        data = rows[tr, t]
        scores = np.column_stack([errors(data, observer(data, pos))[0] for pos in preds])
        paired = np.isfinite(rows[..., :6]).all(axis=-1)
        baseline_rows = rows[paired]
        base = errors(baseline_rows, observer(baseline_rows, reference[paired]))[0]
        librcsc = errors(baseline_rows, observer(baseline_rows, baseline_rows[:, 4:6]))[0]
        order = np.flatnonzero(paired.ravel())
        paired_index = np.searchsorted(order, tr * rows.shape[1] + t)
        if not np.isfinite(scores).all() or not np.allclose(base[paired_index], scores[:, 0], atol=1e-5):
            raise ValueError(f"{name}-{side}: invalid path score or alignment")
        records.append((name, data[:, 17], hindsight, features, scores, base, librcsc, paired_index))
        print(f"{split}: {i + 1}/{len(store.games)} observers", flush=True)
    return records


def fit_gate(records):
    x = np.concatenate([r[3] for r in records])
    scores = np.concatenate([r[4][:, :len(ONLINE)] for r in records])
    delta = scores[:, 1:] - scores[:, :1]
    return np.linalg.solve(x.T @ x + 50 * np.eye(x.shape[1]), x.T @ delta)


def gated(features, weights):
    delta = features @ weights
    return np.argmin(np.column_stack((np.zeros(len(delta)), delta)), axis=1)


def compare(records, weights, selection=None):
    age = np.concatenate([r[1] for r in records])
    hindsight = np.concatenate([r[2] for r in records])
    scores = np.concatenate([r[4] for r in records])
    x = np.concatenate([r[3] for r in records])
    learned = scores[np.arange(len(x)), gated(x, weights)]
    scores_online = np.column_stack((scores[:, :len(ONLINE)], learned))
    methods = (*ONLINE, "learned")
    masks = [(age >= lo) & (age <= hi) for lo, hi in GROUPS]
    if selection is None:
        selection = [methods[np.argmin(scores_online[mask].mean(axis=0))] for mask in masks]
    selected = scores[:, 0].copy()
    for method, mask in zip(selection, masks):
        selected[mask] = scores_online[mask, methods.index(method)]
    cols = {**dict(zip(methods, scores_online.T)), "selected": selected}
    groups = {"supported": np.ones(len(age), bool), "hindsight_eligible": hindsight,
              **{f"{lo}-{hi}": mask for (lo, hi), mask in zip(GROUPS, masks)}}
    metrics = {}
    for name, mask in groups.items():
        values = {**cols, "hindsight": scores[:, -1]} if name == "hindsight_eligible" else cols
        metrics[name] = {key: summarize(value[mask]) for key, value in values.items()}
    total, offset = [], 0
    for _, ages, _, _, _, base, _, indices in records:
        result = base.copy()
        result[indices] = selected[offset:offset + len(ages)]
        total.append(result)
        offset += len(ages)
    metrics["all"] = {"step13": summarize(np.concatenate([r[5] for r in records])),
                      "librcsc": summarize(np.concatenate([r[6] for r in records])),
                      "selected": summarize(np.concatenate(total))}
    return {"selection": selection, "metrics": metrics}


def summarize(value):
    return {"n": len(value), "mean_m": float(value.mean(dtype=np.float64)),
            "p90_m": float(np.quantile(value, .9))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "events", "associated", "kinematics", "runs", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    validation = collect(args, "validation")
    names = sorted({r[0] for r in validation})
    fit_set, select_set = names[:5], names[5:]
    weights = fit_gate([r for r in validation if r[0] in fit_set])
    chosen = compare([r for r in validation if r[0] in select_set], weights)
    print("Selected on validation:", chosen["selection"], flush=True)
    test = compare(collect(args, "test"), weights, chosen["selection"])
    args.output.write_text(json.dumps({"fit_games": fit_set, "selection_games": select_set,
                                       "gate_weights": weights.tolist(), "validation": chosen, "test": test},
                                      indent=2) + "\n")
