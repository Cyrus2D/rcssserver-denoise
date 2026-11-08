#!/usr/bin/env python3
"""Index the flags and goals seen at each Exp2 cycle."""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from build_dataset import ROOT, source_file
from landmark_slots import LANDMARK_COUNT, LANDMARK_VERSION
from schema import FEATURE_INDEX as F

COLUMNS = ("cycle", "stopped", "kind", "landmark_id", "dist", "dir")


def build_game(task):
    dataset, source, staging, game = task
    base = np.load(dataset / game["file"], mmap_mode="r", allow_pickle=False)
    slots = np.zeros((len(base), LANDMARK_COUNT, 2), dtype=np.float32)
    slots[..., 0] = -1
    by_agent = defaultdict(list)
    for span in game["segments"]:
        by_agent[span["agent"]].append(np.arange(span["start"], span["end"]))
    duplicate_readings = 0
    for agent, spans in by_agent.items():
        directory = source / game["id"] / agent
        rows = np.concatenate(spans)
        cycles = pd.Series(rows, index=base["cycle"][rows])
        if not cycles.index.is_unique:
            raise ValueError(f"{directory}: duplicate cycles in the dataset")
        observations = pd.read_csv(source_file(directory, "landmarks"), usecols=COLUMNS)
        observations = observations.loc[(observations["stopped"] == 0)
                                        & observations["kind"].isin(("flag", "goal"))]
        observations = observations.assign(row=observations["cycle"].map(cycles)).dropna(
            subset=["row", "landmark_id", "dist", "dir"],
        )
        indices = observations["row"].to_numpy(dtype=np.intp)
        observations = observations.loc[base["features"][indices, F["see"]] > 0.5]
        if not observations["landmark_id"].between(0, LANDMARK_COUNT - 1).all():
            raise ValueError(f"{directory}: invalid flag or goal ID")
        if not (observations["dist"] >= 0).all() or not np.isfinite(observations[["dist", "dir"]]).all().all():
            raise ValueError(f"{directory}: invalid distance or bearing")
        duplicate_readings += int(observations.duplicated(["row", "landmark_id"]).sum())
        observations = observations.drop_duplicates(["row", "landmark_id"], keep="last")
        indices = observations["row"].to_numpy(dtype=np.intp)
        landmarks = observations["landmark_id"].to_numpy(dtype=np.intp)
        slots[indices, landmarks, 0] = observations["dist"].to_numpy()
        slots[indices, landmarks, 1] = np.deg2rad(observations["dir"].to_numpy())
    np.save(staging / game["file"], slots, allow_pickle=False)
    return {"file": game["file"], "rows": len(slots), "visible_slots": int((slots[..., 0] >= 0).sum()),
            "duplicate_readings": duplicate_readings}


def build(args):
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output")
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    splits = sorted(set(args.split or ["train", "validation"]))
    games = [entry for entry in manifest["games"] if entry["split"] in splits]
    if {entry["split"] for entry in games} != set(splits):
        raise ValueError("At least one selected split is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "index"
        staging.mkdir()
        tasks = [(dataset, Path(manifest["source"]), staging, game) for game in games]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            entries = []
            for i, entry in enumerate(pool.map(build_game, tasks), 1):
                entries.append(entry)
                if i == 1 or i % 10 == 0 or i == len(games):
                    print(f"Indexed {i}/{len(games)} games", flush=True)
        index = {"schema_version": LANDMARK_VERSION, "landmarks": LANDMARK_COUNT, "splits": splits,
                 "dataset_fingerprint": hashlib.sha256(manifest_bytes).hexdigest(), "games": entries}
        (staging / "manifest.json").write_text(json.dumps(index, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"{output} appeared during the build")
        staging.rename(output)
    print(f"Landmark slots: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "exp2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), action="append")
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output is None:
        args.output = args.dataset / "landmark-slots"
    try:
        build(args)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
