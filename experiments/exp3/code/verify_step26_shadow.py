#!/usr/bin/env python3
"""Compare shadow positions with the Python checkpoint on the same live logs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_intersections import context_features, intersect
from build_sequences import AGE, AGENT, FEATURE_START, POSE, POSE_START, make_tracks
from train_context import ContextGRU, model_batch
from train_range import WINDOW

CANDIDATE = ["denoise_pos_x", "denoise_pos_y"]


@torch.inference_mode()
def check(path, model):
    meta = json.loads((path / "meta.json").read_text())
    if not meta["final"] or meta["status"] != "ok":
        raise ValueError(f"{path}: incomplete dataset")
    sequence, _ = make_tracks(path)
    context, unums = context_features(path, sequence)
    speed = max(float(meta["server_param"]["player_speed_max"]),
                *(float(p["player_speed_max"]) for p in meta["player_types"]))
    geometry, _ = intersect(sequence, speed)
    agent = pd.read_csv(path / "agent_cycle.csv.gz", usecols=AGENT)
    cycles = agent.loc[agent.game_mode.eq("play_on") & agent.decision_made.eq(1)
                       & agent.stopped.eq(0) & agent.see_used.eq(1)
                       & agent[POSE].notna().all(axis=1), "cycle"].sort_values()
    keys = ["cycle", "stopped", "object_id"]
    fields = ["match_method", "denoise_applied", *CANDIDATE]
    objects = pd.read_csv(path / "objects.csv.gz", usecols=keys + fields)
    if objects.denoise_applied.eq(1).any():
        raise ValueError(f"{path}: shadow logs required")
    objects = objects[objects.stopped.eq(0) & objects.object_id.ne(0)]
    ids = sorted(objects.object_id.unique())
    ordered = objects.set_index(["cycle", "object_id"]).reindex(pd.MultiIndex.from_product([cycles, ids]))
    logged = ordered[CANDIDATE].where(ordered.match_method.eq("unum"), np.nan)
    logged = logged.to_numpy().reshape(len(cycles), len(ids), 2).transpose(1, 0, 2)
    eligible = sequence[..., AGE] >= WINDOW
    available = np.isfinite(logged).all(axis=-1)
    errors = []
    for positions in np.array_split(np.argwhere(eligible & available),
                                    max(1, int((eligible & available).sum()) // 1024 + 1)):
        if not len(positions):
            continue
        tr, end = positions.T
        times = end[:, None] + np.arange(1 - WINDOW, 1)
        current = sequence[tr, end]
        batch = (sequence[tr[:, None], times, FEATURE_START:], current,
                 context[tr[:, None], times[:, :-1]].astype(np.float32), unums[tr],
                 sequence[tr, end - 1, POSE_START:POSE_START + 3],
                 geometry[tr, end - 1], geometry[tr, end])
        relative, _ = model_batch(model, batch)
        world = relative.numpy() + current[:, POSE_START:POSE_START + 2]
        errors.append(np.linalg.norm(world - logged[tr, end], axis=-1))
    if not errors:
        raise ValueError(f"{path}: no paired shadow predictions")
    error = np.concatenate(errors)
    return {"paired": len(error), "python_eligible": int(eligible.sum()),
            "live_available": int(available.sum()), "mean_m": float(error.mean()),
            "p99_m": float(np.quantile(error, .99)), "max_m": float(error.max())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("observers", type=Path, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["mode"] != "masked_set_polar":
        parser.error("expected a Step 26 checkpoint")
    model = ContextGRU(checkpoint["mode"]).eval()
    model.load_state_dict(checkpoint["model"])
    print(json.dumps({str(path): check(path, model) for path in args.observers}, indent=2))
