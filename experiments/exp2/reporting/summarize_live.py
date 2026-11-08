#!/usr/bin/env python3
"""Measure self localization against fullstate, weighting each player-cycle equally."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cycle", "stopped"]
EST_POS = ["est_pos_x", "est_pos_y"]
TRUTH = ["tru_pos_x", "tru_pos_y"]
CYCLE_COLUMNS = KEY + EST_POS + TRUTH + ["decision_made", "game_mode", "truth_valid", "est_pos_count"]
AGE_LABELS = ("all", "age 0", "age 1", "age 2", "age 3+")


def distance(frame, first, second):
    return np.linalg.norm(frame[first].to_numpy() - frame[second].to_numpy(), axis=1)


def analyze_game(paths):
    effective, paired = defaultdict(list), defaultdict(list)
    statuses = Counter()
    coverage = Counter()
    manifest = []
    for path in paths:
        meta_path = path.with_name("meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            manifest.append((meta["side"], meta["final"], meta["status"],
                             meta["score_left"], meta["score_right"]))
        frame = pd.read_csv(path, usecols=CYCLE_COLUMNS)
        frame = frame[frame["decision_made"].eq(1)].set_index(KEY, verify_integrity=True)
        play = frame.index.get_level_values("stopped") == 0
        play = play & frame["game_mode"].eq("play_on")
        scored = play & frame["truth_valid"].eq(1) & frame[EST_POS + TRUTH].notna().all(axis=1)
        model_log = path.with_name("self_localization.csv")
        mode = "off"
        if model_log.exists():
            decisions = pd.read_csv(model_log).set_index(KEY, verify_integrity=True)
            modes = decisions["mode"].unique()
            if len(modes) != 1 or modes[0] not in {"apply", "shadow"}:
                raise ValueError(f"invalid mode in {model_log}")
            mode = modes[0]
            statuses.update({(mode, status): n for status, n in decisions["status"].value_counts().items()})
            ready = decisions["status"].eq("ready")
            if not decisions["applied"].eq((ready & (mode == "apply")).astype(int)).all():
                raise ValueError(f"unexpected application in {model_log}")
            observed = decisions.loc[ready].join(frame)
            if not (observed["game_mode"].eq("play_on")
                    & (observed.index.get_level_values("stopped") == 0)).all():
                raise ValueError(f"missing PlayOn decision in {model_log}")
            position = ["model_x", "model_y"] if mode == "apply" else ["baseline_x", "baseline_y"]
            if not (distance(observed, position, EST_POS) <= 0.0001).all():
                raise ValueError(f"{mode} did not produce the expected world-model position in {model_log}")
            observed = observed[observed["truth_valid"].eq(1) & observed[TRUTH].notna().all(axis=1)]
            data = np.column_stack((
                observed["pos_age"], 100 * distance(observed, ["baseline_x", "baseline_y"], TRUTH),
                100 * distance(observed, ["model_x", "model_y"], TRUTH), observed["inference_us"],
            ))
            if not np.isfinite(data).all():
                raise ValueError(f"invalid ready-cycle data in {model_log}")
            paired[mode].append(data.astype(np.float32))
        effective[mode].append(np.column_stack((
            frame.loc[scored, "est_pos_count"], 100 * distance(frame.loc[scored], EST_POS, TRUTH),
        )).astype(np.float32))
        coverage[mode, "players"] += 1
        coverage[mode, "play_on"] += int(play.sum())
        coverage[mode, "truth"] += int((play & frame["truth_valid"].eq(1)).sum())
        coverage[mode, "scored"] += int(scored.sum())
    return ({mode: np.concatenate(values) for mode, values in effective.items()},
            {mode: np.concatenate(values) for mode, values in paired.items()}, statuses, coverage, manifest)


def stats(values):
    return {"n": len(values), "mean": float(values.mean(dtype=np.float64)),
            "p90": float(np.quantile(values, 0.90)), "p99": float(np.quantile(values, 0.99))}


def by_age(data):
    yield AGE_LABELS[0], data[:, 1:]
    for age, label in enumerate(AGE_LABELS[1:]):
        mask = data[:, 0] == age if age < 3 else data[:, 0] >= age
        if mask.any():
            yield label, data[mask, 1:]


def game_interval(deltas, counts):
    """Bootstrap games, retaining the sample-weighted estimator within each draw."""
    deltas, counts = np.asarray(deltas), np.asarray(counts)
    if len(deltas) < 2:
        return None
    rng = np.random.default_rng(42)
    samples = rng.integers(len(deltas), size=(10000, len(deltas)))
    estimates = deltas[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def summarize(root, workers):
    games = defaultdict(list)
    for path in sorted(set(root.rglob("agent_cycle.csv.gz")) | set(root.rglob("agent_cycle.csv"))):
        games[path.parent.parent].append(path)
    if not games:
        raise ValueError(f"no agent_cycle files under {root}")
    effective, paired = defaultdict(list), defaultdict(list)
    statuses, coverage = Counter(), Counter()
    metadata, per_game = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for game, (used, candidates, counts, logged, manifest) in zip(games, pool.map(analyze_game, games.values())):
            metadata[game.name] = manifest
            per_game[game.name] = {
                "effective": {mode: [len(data), float(data[:, 1].sum(dtype=np.float64))]
                              for mode, data in used.items()},
                "paired": {mode: [len(data), float((data[:, 1] - data[:, 2]).sum(dtype=np.float64))]
                           for mode, data in candidates.items()},
            }
            for mode, data in used.items():
                effective[mode].append(data)
            for mode, data in candidates.items():
                paired[mode].append(data)
            statuses.update(counts)
            coverage.update(logged)

    summary = {"games": len(games), "metadata": metadata, "per_game": per_game,
               "coverage": {}, "statuses": {}, "effective": {}, "paired": {}}
    for (mode, metric), value in sorted(coverage.items()):
        summary["coverage"].setdefault(mode, {})[metric] = value
    for (mode, status), count in sorted(statuses.items()):
        summary["statuses"].setdefault(mode, {})[status] = count

    print(f"Read {sum(len(paths) for paths in games.values()):,} player logs across {len(games)} games.")
    print("Position used on all scored PlayOn cycles (mean/p90 cm)")
    for mode, parts in sorted(effective.items()):
        cohorts = {}
        for label, data in by_age(np.concatenate(parts)):
            if len(data):
                result = cohorts[label] = stats(data[:, 0])
                print(f"  {mode:6} {label:6} {result['n']:>10,} cycles  {result['mean']:.3f}/{result['p90']:.3f}")
        summary["effective"][mode] = cohorts

    print("\nSelf localization decision counts by mode and status")
    for (mode, status), count in sorted(statuses.items()):
        print(f"  {mode:6} {status:17} {count:>10,}")

    print("\nMode    Visual age       Paired   librcsc mean/p90 cm     Model mean/p90 cm   Infer p90 us")
    for mode, parts in sorted(paired.items()):
        cohorts = {}
        for label, data in by_age(np.concatenate(parts)):
            if not len(data):
                continue
            baseline, model, elapsed = data.T
            ref, nn, timing = stats(baseline), stats(model), stats(elapsed)
            cohorts[label] = {"baseline": ref, "model": nn, "timing_us": timing,
                              "improved_share": float(np.mean(model < baseline)),
                              "worse_by_10cm_share": float(np.mean(model - baseline > 10))}
            print(f"{mode:8}{label:11}{len(data):>11,}{ref['mean']:>11.3f}/{ref['p90']:.3f}"
                  f"{nn['mean']:>15.3f}/{nn['p90']:.3f}{timing['p90']:>16.1f}")
        game_pairs = [entry["paired"][mode] for entry in per_game.values()
                      if mode in entry["paired"] and entry["paired"][mode][0]]
        if game_pairs:
            counts, deltas = zip(*game_pairs)
            summary["paired"][mode] = {
                "ages": cohorts, "games_improved": sum(delta > 0 for delta in deltas),
                "games_scored": len(deltas), "mean_gain_ci_cm": game_interval(deltas, counts),
            }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json", type=Path, help="write machine-readable statistics")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = summarize(args.logs, args.workers)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
