#!/usr/bin/env python3
"""Measure one-cycle-delayed fusion with a second observer on the same side."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_normal_events import observations
from compare_directional import public_speed, step13
from compare_velocity_bins import velocities
from replay_normal import unit
from train_normal import Store, errors, load_models, observer

POSE = ["est_pos_x", "est_pos_y", "est_face", "est_vel_x", "est_vel_y", "focus_dist", "focus_dir"]
METHODS = ("step13", "teammate", "blend25", "blend50")
AGES = ("fresh", "age_1_2", "age_3_plus")


def second_observer(path, cycles, ids, side, speed, receiver_pose, packet):
    meta = json.loads((path / "meta.json").read_text())
    config = meta["config"]
    if not meta["final"] or config["use_fullstate"] or config["gaussian_see"]:
        raise ValueError(f"{path}: need completed quantized sender observations")
    agent = pd.read_csv(path / "agent_cycle.csv.gz",
                        usecols=["cycle", "stopped", "game_mode", "decision_made", "see_used", *POSE])
    send = agent[agent.cycle.isin(cycles - 1) & agent.stopped.eq(0) & agent.game_mode.eq("play_on")
                 & agent.decision_made.eq(1) & agent.see_used.eq(1) & agent[POSE[:3]].notna().all(axis=1)]
    if send.cycle.duplicated().any():
        raise ValueError(f"{path}: duplicate sender cycles")
    own, opponent = meta["team_name"], meta["opponent_team_name"]
    teams = (own, opponent) if side == "l" else (opponent, own)
    events, seen = observations(path, send.cycle.to_numpy(), *teams)
    if seen != len(send):
        raise ValueError(f"{path}: missing sender raw observations")
    raw_index, slot = np.nonzero(np.isin(events[..., 0], ids) & (events[..., 2] > 0))
    raw = events[raw_index, slot]
    pose = send[POSE].fillna(0).to_numpy()[raw_index]
    xy = pose[:, :2] + raw[:, 2, None] * unit(np.deg2rad(pose[:, 2] + raw[:, 3]))
    moving = np.isfinite(raw[:, 4:6]).all(axis=1)
    _, midpoint, _ = velocities(raw[moving, 2:6], pose[moving], np.zeros((moving.sum(), 2)), speed)
    xy[moving] += midpoint
    t = np.searchsorted(cycles, send.cycle.to_numpy()[raw_index] + 1)
    tr = np.searchsorted(ids, raw[:, 0])
    if packet:
        # Cycle-based round robin uses sender measurements alone.
        priority = (raw[:, 0] - 1 - send.cycle.to_numpy()[raw_index]) % 22
        order = np.lexsort((priority, raw_index))
        _, first = np.unique(raw_index[order], return_index=True)
        keep = order[first]
        # Match OnePlayerMessage's field bounds and coordinate grid.
        offset = np.array([52.5, 34.0])
        bounds = offset - .01
        xy = np.clip(np.rint((np.clip(xy, -bounds, bounds) + offset) / .63),
                     0, [167, 108]) * .63 - offset
        # Estimated range approximates server audibility at the send cycle.
        previous = np.maximum(t[keep] - 1, 0)
        separation = np.linalg.norm(pose[keep, :2] - receiver_pose[previous, :2], axis=1)
        audio_limit = float(meta["server_param"]["audio_cut_dist"])
        keep = keep[(cycles[t[keep]] - cycles[previous] == 1) & (separation <= audio_limit)]
        tr, t, xy = tr[keep], t[keep], xy[keep]
    result = np.full((len(ids), len(cycles), 2), np.nan, np.float32)
    result[tr, t] = xy
    return result


def run(args, split, selection=None):
    store = Store(args.associated, args.events, split)
    models, calibration = load_models(args.runs)
    config = json.loads((args.runs / "audited-validation.json").read_text())["config"]
    games = json.loads((args.events / "manifest.json").read_text())["games"]
    numbers = {game["name"]: game["observer_unum"] for game in games}
    buckets, records = [], []
    for i, (game, rows, cycles) in enumerate(store.games):
        side = "lr"[i % 2]
        with np.load(args.events / f"{game}-{side}.npz") as event_file:
            ids, pose = event_file["ids"], event_file["pose"]
        primary = numbers[game]
        sender = primary % 11 + 1
        remote = second_observer(args.source / game / f"{side}_{sender}", cycles, ids, side,
                                 public_speed(args.source, game, primary, side), pose, args.packet)
        reference = step13(store, i, models, calibration, config)
        paired = np.isfinite(rows[..., :6]).all(axis=-1)
        data = rows[paired]
        ref, other = reference[paired], remote[paired]
        available = np.isfinite(other).all(axis=-1)
        other[~available] = ref[~available]
        age = data[:, 17]
        buckets.append(np.where(available, np.where(age == 0, 0, np.where(age <= 2, 1, 2)), -1))
        scores = {}
        for method, position in (
            ("step13", ref), ("librcsc", data[:, 4:6]), ("teammate", other),
            ("blend25", .75 * ref + .25 * other), ("blend50", .5 * (ref + other)),
        ):
            scores[method] = errors(data, observer(data, position))[0]
        records.append(scores)
        print(f"{split}: {i + 1}/{len(store.games)} observers", flush=True)
    bucket = np.concatenate(buckets)
    scores = {method: np.concatenate([r[method] for r in records]) for method in records[0]}
    if selection is None:
        selection = [min(METHODS, key=lambda m: scores[m][bucket == b].mean())
                     if (bucket == b).sum() >= 100 else "step13" for b in range(len(AGES))]
    selected = scores["step13"].copy()
    for b, method in enumerate(selection):
        selected[bucket == b] = scores[method][bucket == b]
    scores["selected"] = selected
    groups = {"all": np.ones(len(bucket), bool), "available": bucket >= 0,
              **{name: bucket == b for b, name in enumerate(AGES)}}
    metrics = {name: {m: {"n": int(mask.sum()), "mean_m": float(v[mask].mean(dtype=np.float64)),
                         "p90_m": float(np.quantile(v[mask], .9))}
                      for m, v in scores.items()}
               for name, mask in groups.items() if mask.any()}
    return {"selection": dict(zip(AGES, selection)), "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "events", "associated", "runs", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--packet", action="store_true", help="one quantized target per sender cycle")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    validation = run(args, "validation")
    print("Selected on validation:", validation["selection"], flush=True)
    test = run(args, "test", list(validation["selection"].values()))
    args.output.write_text(json.dumps({"validation": validation, "test": test}, indent=2) + "\n")
