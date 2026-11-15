#!/usr/bin/env python3
"""Read causal librcsc velocities and ages for the normal-view holdouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["trk_vel_x", "trk_vel_y", "trk_vel_count"]


def build(job):
    source, events, output, game = job
    name, number = game["name"], game["observer_unum"]
    for side in "lr":
        with np.load(events / f"{name}-{side}.npz", allow_pickle=False) as raw:
            cycles, ids = raw["cycle"], raw["ids"]
        path = source / name / f"{side}_{number}"
        objects = pd.read_csv(path / "objects.csv.gz",
                              usecols=["cycle", "stopped", "object_id", "track_match"] + FIELDS,
                              dtype={"track_match": str})
        objects = objects[objects.stopped.eq(0) & objects.object_id.isin(ids) & objects.cycle.isin(cycles)]
        objects = objects.set_index(["cycle", "object_id"], verify_integrity=True)
        indexed = objects.reindex(pd.MultiIndex.from_product([cycles, ids], names=["cycle","object_id"]))
        exact = indexed["track_match"].eq("unum")
        kinematics = indexed[FIELDS].where(exact, np.nan).to_numpy(np.float32)
        np.save(output / f"{name}-{side}.npy", kinematics.reshape(len(cycles), len(ids), 3).transpose(1,0,2),
                allow_pickle=False)
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("events", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    data = json.loads((args.events / "manifest.json").read_text())
    games = [g for g in data["games"] if g["split"] in {"validation", "test"}]
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(args.source, args.events, args.output, g) for g in games]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        names = list(pool.map(build, jobs))
    (args.output / "manifest.json").write_text(json.dumps(names) + "\n")
    print(f"Built kinematics for {len(names)} games.")


if __name__ == "__main__":
    main()
