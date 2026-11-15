#!/usr/bin/env python3
"""Add the server quantization interval to each available Exp4 ball sighting."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_sequences import range_bounds


def build(job):
    source, output = job
    with np.load(output / "timing.npz", allow_pickle=False) as data:
        cycles = data["cycles"]
    a = pd.read_csv(source / "agent_cycle.csv.gz", usecols=["cycle", "stopped", "focus_dist", "focus_dir"])
    focus = a[a.stopped.eq(0)].set_index("cycle", verify_integrity=True).reindex(cycles).fillna(0)
    inputs = np.load(output / "input.npy", mmap_mode="r", allow_pickle=False)[:, 0]
    bearing = np.arctan2(inputs[:, 14], inputs[:, 13])
    available = (inputs[:, 13:15] != 0).any(axis=1)
    distance = np.where(available, inputs[:, 12] * 60, np.nan)
    lo, hi, fallback = range_bounds(distance[:, None], bearing[:, None],
                                     focus.focus_dist.to_numpy(), np.deg2rad(focus.focus_dir.to_numpy()))
    np.save(output / "ball_bounds.npy", np.column_stack((lo, hi, available)).astype(np.float32), allow_pickle=False)
    return int(available.sum()), fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    games = json.loads((args.dataset / "manifest.json").read_text())["games"]
    jobs = [(args.source / g["name"] / f"{side}_{g['observer_unum']}", args.dataset / f"{g['name']}-{side}")
            for g in games for side in "lr"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result = list(pool.map(build, jobs))
    print("Available and approximate fallback intervals:", np.sum(result, axis=0).tolist())
