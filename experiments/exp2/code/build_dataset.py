#!/usr/bin/env python3
"""Build the exp2 self-localization database from finalized exp1 logs."""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from schema import FEATURES, PHYSICS_KEYS, ROW, SCHEMA_VERSION, STATE

ROOT = Path("/home/nader/workspace/denoise/logs")
ESTIMATE = ("est_pos_x", "est_pos_y", "est_vel_x", "est_vel_y", "est_body", "est_neck")
TRUTH = ("tru_pos_x", "tru_pos_y", "tru_vel_x", "tru_vel_y", "tru_body", "tru_neck")
AGENT_COLUMNS = (
    "cycle", "stopped", "game_mode", "decision_made", "truth_valid", *ESTIMATE, *TRUTH,
    "body_cmd", "dash_power", "dash_dir", "turn_moment", "turn_neck_moment",
    "sb_effort", "sb_received", "sb_speed_mag", "sb_speed_dir", "sb_neck_dir",
    "sb_coll_ball", "sb_coll_player", "sb_coll_post", "see_used", "view_width",
    "est_pos_count", "est_face_count", "rawfix_pos_x", "rawfix_pos_y", "rawfix_face",
    "rawfix_pos_err_x", "rawfix_pos_err_y", "rawfix_face_err",
)
LANDMARK_COLUMNS = ("cycle", "stopped", "kind", "dist", "dir", "land_x", "land_y")
LANDMARK_FEATURES = ("land_count", "land_mean_dist", "land_min_dist",
                     "land_bearing_spread", "land_span", "line_count")


def source_file(directory, table):
    compressed = directory / f"{table}.csv.gz"
    return compressed if compressed.is_file() else directory / f"{table}.csv"


def read_table(directory, table, columns):
    return pd.read_csv(source_file(directory, table), usecols=columns)


def view_geometry(directory, cycles):
    rows = read_table(directory, "landmarks", LANDMARK_COLUMNS)
    rows = rows.loc[(rows["stopped"] == 0) & rows["cycle"].isin(cycles)]
    line_counts = rows.loc[rows["kind"] == "line"].groupby("cycle").size()
    points = rows.loc[rows["kind"].isin(("flag", "goal"))].dropna(subset=["dist", "dir", "land_x", "land_y"])
    radians = np.deg2rad(points["dir"])
    points = points.assign(sin=np.sin(radians), cos=np.cos(radians))
    geometry = points.groupby("cycle").agg(
        land_count=("dist", "size"), land_mean_dist=("dist", "mean"), land_min_dist=("dist", "min"),
        sin=("sin", "mean"), cos=("cos", "mean"),
        min_x=("land_x", "min"), max_x=("land_x", "max"), min_y=("land_y", "min"), max_y=("land_y", "max"),
    ).reindex(cycles)
    geometry["land_bearing_spread"] = (1 - np.hypot(geometry["sin"], geometry["cos"])).clip(0, 1)
    geometry["land_span"] = np.hypot(geometry["max_x"] - geometry["min_x"], geometry["max_y"] - geometry["min_y"])
    geometry["line_count"] = line_counts.reindex(cycles)
    return geometry.loc[:, LANDMARK_FEATURES].fillna(0)


def player_states(rows, columns):
    result = rows.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    result[:, 4:] = np.deg2rad(result[:, 4:])
    return result


def extract_agent(directory):
    meta = json.loads((directory / "meta.json").read_text())
    if meta["schema_version"] != 2 or not meta["final"] or meta["status"] != "ok":
        raise ValueError(f"{directory}: need a finalized schema-v2 log")
    if meta["config"]["use_fullstate"] is not False:
        raise ValueError(f"{directory}: the logged world model must use reference-only fullstate")
    physics = {key: float(meta["server_param"][source]) for key, source in PHYSICS_KEYS.items()}
    rows = read_table(directory, "agent_cycle", AGENT_COLUMNS)
    keep = (rows["game_mode"] == "play_on") & (rows["decision_made"] == 1) & (rows["stopped"] == 0)
    rows = rows.loc[keep].set_index("cycle").sort_index()
    if not rows.index.is_unique:
        raise ValueError(f"{directory}: duplicate play_on cycles")
    cycles = rows.index.to_numpy(dtype=np.int32)
    prior = rows.reindex(cycles - 1)
    sb = rows["sb_received"].eq(1)
    see = rows["see_used"].eq(1)
    fix_pos = see & rows[["rawfix_pos_x", "rawfix_pos_y"]].notna().all(axis=1)
    fix_face = see & rows["rawfix_face"].notna()
    features = {
        "dash_power": prior["dash_power"].where(prior["body_cmd"] == "dash"),
        "dash_dir": np.deg2rad(prior["dash_dir"]),
        "turn": np.deg2rad(prior["turn_moment"].where(prior["body_cmd"] == "turn")),
        "neck_turn": np.deg2rad(prior["turn_neck_moment"]),
        "effort": prior["sb_effort"].fillna(1),
        "dash": prior["body_cmd"].eq("dash"), "turn_cmd": prior["body_cmd"].eq("turn"),
        "tackle": prior["body_cmd"].eq("tackle"),
        "speed": rows["sb_speed_mag"].where(sb),
        "speed_dir": np.deg2rad(rows["sb_speed_dir"].where(sb)),
        "neck": np.deg2rad(rows["sb_neck_dir"].where(sb)),
        "has_speed": sb & rows[["sb_speed_mag", "sb_speed_dir"]].notna().all(axis=1),
        "has_neck": sb & rows["sb_neck_dir"].notna(),
        "collision": sb & rows[["sb_coll_ball", "sb_coll_player", "sb_coll_post"]].eq(1).any(axis=1),
        "see": see, "fix_pos": fix_pos, "fix_face": fix_face,
        "fix_x": rows["rawfix_pos_x"].where(fix_pos), "fix_y": rows["rawfix_pos_y"].where(fix_pos),
        "fix_angle": np.deg2rad(rows["rawfix_face"].where(fix_face)),
        "fix_err_x": rows["rawfix_pos_err_x"].where(fix_pos),
        "fix_err_y": rows["rawfix_pos_err_y"].where(fix_pos),
        "fix_err_angle": np.deg2rad(rows["rawfix_face_err"].where(fix_face)),
        "pos_age": rows["est_pos_count"].fillna(1000),
        "face_age": rows["est_face_count"].fillna(1000),
        **{width: rows["view_width"].eq(width) for width in ("narrow", "normal", "wide")},
    }
    geometry = view_geometry(directory, cycles).where(see, 0)
    features.update({name: geometry[name] for name in LANDMARK_FEATURES})
    data = np.empty(len(rows), dtype=ROW)
    data["cycle"] = cycles
    for i, name in enumerate(FEATURES):
        data["features"][:, i] = features[name].fillna(0).to_numpy(dtype=np.float32)
    if not np.isfinite(data["features"]).all():
        raise ValueError(f"{directory}: non-finite input feature")
    data["truth"] = player_states(rows, TRUTH)
    data["truth"][~rows["truth_valid"].eq(1).to_numpy()] = np.nan
    data["librcsc"] = player_states(rows, ESTIMATE)

    # Only the seed must have a complete librcsc state. Missing truth never
    # creates a boundary or determines which rows the recurrent model sees.
    starts = np.r_[0, np.flatnonzero(np.diff(cycles) != 1) + 1, len(cycles)]
    valid_seed = np.isfinite(data["librcsc"]).all(axis=1)
    segments = []
    for start, end in zip(starts[:-1], starts[1:]):
        seeds = np.flatnonzero(valid_seed[start:end])
        if len(seeds) and start + int(seeds[0]) + 1 < end:
            segments.append((start + int(seeds[0]), end))
    return data, segments, physics, f'{meta["side"]}_{meta["unum"]}'


def build_game(task):
    game, agents, split, directory = task
    arrays, segments = [], []
    physics = None
    offset = 0
    for agent in agents:
        data, intervals, parameters, identity = extract_agent(agent)
        if physics is not None and physics != parameters:
            raise ValueError(f"{game}: mixed server physics within a game")
        physics = parameters
        for start, end in intervals:
            arrays.append(data[start:end])
            next_offset = offset + int(end - start)
            segments.append({"start": offset, "end": next_offset, "agent": identity})
            offset = next_offset
    if not arrays:
        raise ValueError(f"{game}: no contiguous play_on data with a usable seed")
    records = np.concatenate(arrays)
    filename = f"{game}.npy"
    np.save(directory / filename, records, allow_pickle=False)
    return {"id": game, "split": split, "file": filename, "rows": len(records),
            "segments": segments, "agents": len(agents),
            "truth_rows": int(np.isfinite(records["truth"]).all(axis=1).sum())}, physics


def discover(root, limit):
    games = []
    for game in sorted(root.iterdir()):
        if game.is_dir():
            agents = [path for path in sorted(game.iterdir())
                      if path.is_dir() and source_file(path, "agent_cycle").is_file()]
            if agents:
                games.append((game.name, agents))
    games = games[:limit]
    if len(games) < 3:
        raise ValueError("Need at least three games for disjoint train, validation, and test splits")
    return games


def build(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose an unused output directory")
    if output == source or output.is_relative_to(source):
        raise ValueError("The output directory must be outside the input logs")
    games = discover(source, args.limit_games)
    random.Random(args.seed).shuffle(games)
    holdout = max(1, round(len(games) * 0.1))
    assignments = (["train"] * (len(games) - 2 * holdout) + ["validation"] * holdout + ["test"] * holdout)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        tasks = [(name, agents, split, staging) for (name, agents), split in zip(games, assignments)]
        manifest = {"schema_version": SCHEMA_VERSION, "features": FEATURES, "state": STATE,
                    "source": str(source), "split_seed": args.seed, "physics": None, "games": []}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            for i, (entry, physics) in enumerate(pool.map(build_game, tasks), 1):
                if manifest["physics"] is not None and manifest["physics"] != physics:
                    raise ValueError("Mixed server physics; build a separate dataset for each configuration")
                manifest["physics"] = physics
                manifest["games"].append(entry)
                if i == 1 or i % 10 == 0 or i == len(tasks):
                    print(f"Built {i}/{len(tasks)} games", flush=True)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"{output} appeared during the build")
        staging.rename(output)
    print(f"Dataset ready at {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "exp1")
    parser.add_argument("--output", type=Path, default=ROOT / "exp2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--limit-games", type=int, help="Use only the first N games for a smoke run")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_games is not None and args.limit_games < 3):
        parser.error("use at least one worker and at least three games")
    try:
        build(args)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
