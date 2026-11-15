#!/usr/bin/env python3
"""Measure eligibility of the eight-consecutive-observation model in normal games."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_sequences import AGE, make_tracks
from train_range import WINDOW


def inspect(task):
    game, number, dataset = task
    parts = []
    totals = Counter()
    for side in "lr":
        observer = game / f"{side}_{number}"
        agent = pd.read_csv(observer / "agent_cycle.csv.gz", usecols=["game_mode", "decision_made", "see_used"])
        play = agent[agent.game_mode.eq("play_on") & agent.decision_made.eq(1)]
        data, _ = make_tracks(observer)
        if dataset is not None:
            parts.append(data)
        age = data[..., AGE]
        totals["decisions"] += len(play)
        totals["with_see"] += int(play.see_used.eq(1).sum())
        totals["fresh_known_targets"] += int((age >= 1).sum())
        totals["eight_consecutive_targets"] += int((age >= WINDOW).sum())
        totals["observer_cycle_target_slots"] += len(play) * data.shape[0]
    steps = 0
    if dataset is not None:
        steps = max(part.shape[1] for part in parts)
        padded = [np.pad(part, ((0, 0), (0, steps-part.shape[1]), (0, 0)),
                         constant_values=np.nan) for part in parts]
        np.save(dataset / f"{game.name}.npy", np.concatenate(padded), allow_pickle=False)
    return dict(totals), steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-output", type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    games = sorted(path for path in args.source.iterdir() if (path / "l_1/meta.json").is_file())
    if not 1 <= args.games <= len(games) or args.workers < 1:
        parser.error("invalid game or worker count")
    rng = np.random.default_rng(42)
    rng.shuffle(games)
    if args.dataset_output is not None:
        args.dataset_output.mkdir(parents=True, exist_ok=False)
    jobs = [(game, int(rng.integers(1, 12)), args.dataset_output) for game in games[:args.games]]
    manifest = []
    total = Counter()
    details = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (game, number, _), (values, steps) in zip(jobs, pool.map(inspect, jobs)):
            manifest.append({"name": game.name, "observer_unum": number, "steps": steps, "split": "test"})
            total.update(values)
            details[game.name] = {"observer_unum": number, **values}
    if args.dataset_output is not None:
        (args.dataset_output / "manifest.json").write_text(
            json.dumps({"version": 1, "games": manifest}, indent=2) + "\n")
    result = {"games": details, "totals": dict(total)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["totals"])


if __name__ == "__main__":
    main()
