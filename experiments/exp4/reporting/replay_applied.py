#!/usr/bin/env python3
"""Replay ball experts on recorded pre-apply snapshots and audit control geometry.

Uses the eight games selected by replay_control.py. Ages and the observer
kickability cue come from end-of-cycle logs, so this is an approximate replay,
not an exact reconstruction of every live input. Fullstate is scoring only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from build_dataset import observer
from train_state import Model, Stream, error, predict


def create(job):
    path, output = job
    output.mkdir()
    original = pd.read_csv
    def reader(p, *args, **kwargs):
        frame = original(p, *args, **kwargs)
        if str(p).endswith("objects.csv.gz"):
            columns = ["exp4_vel_x", *["exp4_stock_" + f for f in ("pos_x", "pos_y", "vel_x", "vel_y", "body")]]
            extra = original(p, usecols=columns)
            ready = extra.exp4_vel_x.notna()
            for field in ("pos_x", "pos_y", "vel_x", "vel_y", "body"):
                frame.loc[ready, "trk_" + field] = extra.loc[ready, "exp4_stock_" + field]
        return frame
    pd.read_csv = reader
    try:
        observer(path, output)
    finally:
        pd.read_csv = original
    return output


def geometry(path):
    meta = json.loads((path / "meta.json").read_text())
    s = meta["server_param"]
    default = sum(float(s[k]) for k in ("player_size", "ball_size", "kickable_margin"))
    radii = {p["id"]: p["player_size"] + p["kickable_margin"] + float(s["ball_size"]) for p in meta["player_types"]}
    agent = pd.read_csv(path / "agent_cycle.csv.gz", usecols=["cycle", "stopped", "game_mode", "decision_made"])
    agent = agent[(agent.game_mode == "play_on") & (agent.stopped == 0) & (agent.decision_made == 1)]
    cols = """cycle stopped object_id player_type truth_valid tru_on_field dist_chg
exp4_stock_pos_x exp4_stock_pos_y exp4_stock_vel_x exp4_stock_vel_y exp4_pos_x exp4_pos_y exp4_vel_x exp4_vel_y
denoise_pos_x denoise_pos_y denoise_applied tru_pos_x tru_pos_y tru_vel_x tru_vel_y""".split()
    objects = pd.read_csv(path / "objects.csv.gz", usecols=cols).merge(agent, on=["cycle", "stopped"])
    ball = objects[objects.object_id == 0].set_index("cycle")
    players = objects[(objects.object_id != 0) & (objects.truth_valid == 1) & (objects.tru_on_field == 1) & (objects.denoise_applied == 1)]
    players = players[players.cycle.map(ball.exp4_pos_x).notna()]
    def distance(p, bp):
        return np.hypot(players[p + "_x"] - players.cycle.map(ball[bp + "_x"]),
                        players[p + "_y"] - players.cycle.map(ball[bp + "_y"]))
    area = players.player_type.map(radii).fillna(default)
    old = distance("exp4_stock_pos", "exp4_stock_pos") <= area
    new = distance("denoise_pos", "exp4_pos") <= area
    truth = distance("tru_pos", "tru_pos") <= area
    control = dict(n=len(players), true_in_range=int(truth.sum()),
                   false_in_range_before=int((old & ~truth).sum()), false_in_range_after=int((new & ~truth).sum()),
                   missed_before=int((~old & truth).sum()), missed_after=int((~new & truth).sum()),
                   changed=int((old != new).sum()), fixed=int(((old != truth) & (new == truth)).sum()),
                   broken=int(((old == truth) & (new != truth)).sum()))
    b = ball[(ball.exp4_vel_x.notna()) & (ball.truth_valid == 1) & (ball.tru_on_field == 1)]
    target = b[["tru_vel_x", "tru_vel_y"]].values
    e0 = np.linalg.norm(b[["exp4_stock_vel_x", "exp4_stock_vel_y"]].fillna(0).values - target, axis=1)
    e1 = np.linalg.norm(b[["exp4_vel_x", "exp4_vel_y"]].values - target, axis=1)
    groups = {k: [int(mask.sum()), float(e0[mask].sum()), float(e1[mask].sum())]
              for k, mask in {"raw_motion": b.dist_chg.notna(), "no_raw_motion": b.dist_chg.isna()}.items()}
    return control, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", type=Path)
    p.add_argument("control_results", type=Path)
    p.add_argument("runs", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    games = json.loads(args.control_results.read_text())["games"]
    torch.set_num_threads(2)
    model = Model("ball_experts").eval()
    model.load_state_dict(torch.load(args.runs / "ball_experts.pt", weights_only=True, map_location="cpu"))
    scored = []
    control_total = {}
    groups_total = {}
    with tempfile.TemporaryDirectory(prefix="exp4-apply-replay-") as tmp:
        with ProcessPoolExecutor(8) as pool:
            paths = list(pool.map(create, [(args.logs / game / "l_2", Path(tmp) / game) for game in games]))
        with torch.inference_mode():
            for path in paths:
                source = args.logs / path.name / "l_2"
                data = Stream(path, True, 1)
                with np.load(path / "timing.npz") as t:
                    cycles = t["cycles"]
                cols = "cycle stopped object_id exp4_vel_x exp4_vel_y exp4_next_pos_x exp4_next_pos_y exp4_next_vel_x exp4_next_vel_y".split()
                live = pd.read_csv(source / "objects.csv.gz", usecols=cols)
                live = live[(live.stopped == 0) & (live.object_id == 0)].set_index("cycle").reindex(cycles)
                values, separation = [], []
                for start in range(0, len(data.points), 1024):
                    points = data.points[start:start + 1024]
                    batch = data.batch(points, True)
                    pred, target = predict(model, batch)
                    actual = live.iloc[points[:, 0]][cols[3:]].to_numpy()
                    separation.append(np.linalg.norm((pred.numpy()[:, 2:] - actual).reshape(-1, 3, 2), axis=2))
                    values.append(np.column_stack((error(torch.from_numpy(batch[3]), target, True).numpy(),
                                                   error(pred, target, True).numpy())))
                a = np.concatenate(values)
                scored.append(dict(game=path.name, n=len(a), mean=a.mean(axis=0, dtype=np.float64).tolist(),
                                   mean_cpp_output_distance_by_pair=np.nanmean(np.concatenate(separation), axis=0).tolist()))
                control, groups = geometry(source)
                for k, v in control.items():
                    control_total[k] = control_total.get(k, 0) + v
                for k, v in groups.items():
                    groups_total[k] = np.array(groups_total.get(k, [0, 0, 0])) + v
                print("replayed", path.name, flush=True)
    grouped = {k: dict(n=int(v[0]), mean_before=float(v[1] / v[0]), mean_after=float(v[2] / v[0]))
               for k, v in groups_total.items() if v[0]}
    result = dict(games=scored, geometry=control_total, ball_velocity_groups=grouped,
                  limitation="Pre-apply values reconstructed; estimate ages and observer kickability are end-of-cycle logged values.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
