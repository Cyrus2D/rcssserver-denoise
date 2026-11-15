#!/usr/bin/env python3
"""Test velocity derived from quantized visual motion fields in normal view."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_sequences import range_bounds
from compare_directional import bounded_velocity, public_speed, step13
from replay_normal import MAX_AGE, unit
from train_normal import Store, errors, load_models, observer

METHODS = ("step13", "prior", "midpoint", "clipped", "clipped_blend")
GAPS = ((1, 1), (2, 2), (3, 5), (6, MAX_AGE))


def velocity_bounds(events, pose):
    """Invert the server's rounded radial rate and angular rate."""
    distance, bearing, dist_change, dir_change = events.T
    bearing = np.deg2rad(bearing)
    low, high, _ = range_bounds(distance[:, None], bearing[:, None], pose[:, 5], np.deg2rad(pose[:, 6]))
    radius = np.stack((low.ravel(), high.ravel()))
    radial = dist_change / distance
    tangent = np.deg2rad(dir_change)
    rates = np.stack((np.stack((radial - .01, radial + .01)),
                      np.stack((tangent - np.deg2rad(.05), tangent + np.deg2rad(.05)))), axis=-1)
    products = radius[:, None, :, None] * rates[None]
    return products.min(axis=(0, 1)), products.max(axis=(0, 1))


def velocities(events, pose, prior, limit):
    low, high = velocity_bounds(events, pose)
    radial = unit(np.deg2rad(pose[:, 2] + events[:, 1]))
    tangent = np.stack((-radial[:, 1], radial[:, 0]), axis=-1)
    basis = np.stack((radial, tangent), axis=1)
    own = pose[:, 3:5]
    prior_components = np.einsum("nki,ni->nk", basis, prior - own)
    midpoint = (low + high) / 2
    clipped = np.where(np.isfinite(prior_components), np.clip(prior_components, low, high), midpoint)
    return tuple(bounded_velocity(own + np.einsum("nki,nk->ni", basis, v), limit)
                 for v in (prior_components, midpoint, clipped))


def variants(rows, cycles, pose, events, ids, kinematics, reference, limit):
    # Only exact uniform numbers can seed a velocity. Anonymous updates
    # terminate propagation, as does a break in the PlayOn cycle sequence.
    time, event = np.nonzero(np.isin(events[..., 0], ids) & (events[..., 2] > 0)
                             & np.isfinite(events[..., 2:6]).all(axis=-1))
    track = np.searchsorted(ids, events[time, event, 0])
    valid = (time > 0) & (rows[track, time, 18] == 1)
    time, event, track = time[valid], event[valid], track[valid]
    prior = kinematics[track, time - 1].copy()
    valid = (cycles[time] - cycles[time - 1] == 1) & np.isfinite(prior).all(axis=-1) & (prior[:, 2] <= 6)
    time, event, track, prior = time[valid], event[valid], track[valid], prior[valid, :2]
    prior, midpoint, clipped = velocities(events[time, event, 2:6], pose[time], prior, limit)

    position = {name: reference.copy() for name in METHODS[1:]}
    supported = np.zeros(rows.shape[:2], bool)
    for age in range(1, MAX_AGE + 1):
        current = np.minimum(time + age, len(cycles) - 1)
        mask = ((cycles[current] - cycles[time] == age) & (rows[track, current, 17] == age)
                & np.isfinite(reference[track, time]).all(axis=-1))
        t, tr, start = current[mask], track[mask], time[mask]
        anchor = reference[tr, start]
        supported[tr, t] = True
        # One post-decay velocity is applied next cycle, with a .4 decay
        # thereafter. This intentionally models coasting, with no dash model.
        travel = (1 - .4 ** age) / (1 - .4)
        for name, velocity in (("prior", prior), ("midpoint", midpoint), ("clipped", clipped)):
            position[name][tr, t] = anchor + travel * velocity[mask]
        position["clipped_blend"][tr, t] = .5 * (reference[tr, t] + position["clipped"][tr, t])
    return position, supported


def evaluate(args, split, selection=None):
    store = Store(args.associated, args.events, split)
    models, calibration = load_models(args.runs)
    config = json.loads((args.runs / "audited-validation.json").read_text())["config"]
    games = json.loads((args.events / "manifest.json").read_text())["games"]
    numbers = {game["name"]: game["observer_unum"] for game in games}
    scored = []
    for index, (name, rows, cycles) in enumerate(store.games):
        side = "lr"[index % 2]
        with np.load(args.events / f"{name}-{side}.npz") as event_file:
            pose, events, ids = (event_file[k] for k in ("pose", "events", "ids"))
            if not np.array_equal(event_file["cycle"], cycles):
                raise ValueError(f"{name}-{side}: event cycle alignment mismatch")
        kin = np.load(args.kinematics / f"{name}-{side}.npy", mmap_mode="r")
        reference = step13(store, index, models, calibration, config)
        alternatives, available = variants(
            rows, cycles, pose, events, ids, kin, reference,
            public_speed(args.source, name, numbers[name], side),
        )
        paired = np.isfinite(rows[..., :6]).all(axis=-1)
        data = rows[paired]
        pos = {"librcsc": rows[..., 4:6], "step13": reference, **alternatives}
        score = {key: errors(data, observer(data, val[paired]))[0] for key, val in pos.items()}
        if not all(np.isfinite(value).all() for value in score.values()):
            raise ValueError(f"{name}-{side}: non-finite prediction")
        scored.append((name, data[:, 17], available[paired], score))
        print(f"{split}: {index + 1}/{len(store.games)} observers", flush=True)

    if selection is None:
        selection = []
        for lo, hi in GAPS:
            totals = np.zeros(len(METHODS))
            count = 0
            for _, age, available, score in scored:
                mask = available & (age >= lo) & (age <= hi)
                count += mask.sum()
                totals += [score[key][mask].sum(dtype=np.float64) for key in METHODS]
            selection.append(METHODS[np.argmin(totals)] if count >= 100 else "step13")

    metrics = {}
    for group in ("all", "supported", *(f"{lo}-{hi}" for lo, hi in GAPS)):
        collected = {name: [] for name in ("librcsc", *METHODS, "selected")}
        for _, age, available, score in scored:
            chosen = score["step13"].copy()
            for (lo, hi), method in zip(GAPS, selection):
                update = available & (age >= lo) & (age <= hi)
                chosen[update] = score[method][update]
            mask = np.ones_like(available) if group == "all" else available.copy()
            if "-" in group:
                lo, hi = map(int, group.split("-"))
                mask &= (age >= lo) & (age <= hi)
            for key, value in {**score, "selected": chosen}.items():
                collected[key].append(value[mask])
        metrics[group] = {}
        for method, values in collected.items():
            data = np.concatenate(values)
            if len(data):
                metrics[group][method] = {
                    "n": len(data), "mean_m": float(data.mean(dtype=np.float64)),
                    "p90_m": float(np.quantile(data, .9)),
                }
    return {"selection": selection, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "events", "associated", "kinematics", "runs", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    validation = evaluate(args, "validation")
    print("Selected on validation:", validation["selection"], flush=True)
    test = evaluate(args, "test", validation["selection"])
    args.output.write_text(json.dumps({"validation": validation, "test": test}, indent=2) + "\n")
