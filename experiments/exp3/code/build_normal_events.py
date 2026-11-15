#!/usr/bin/env python3
"""Store normal-view player measurements separately from evaluation labels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

AGENT = "cycle stopped game_mode decision_made see_used est_pos_x est_pos_y est_face est_vel_x est_vel_y focus_dist focus_dir".split()
POSE = AGENT[5:]
OBJECT = "cycle stopped object_id truth_valid tru_on_field tru_pos_x tru_pos_y true_dist true_dir true_focus_dist track_match trk_pos_x trk_pos_y".split()
PLAYER = re.compile(r'\(\(([pP])(?: "([^"]+)")?(?: (\d+))?(?: goalie)?\) ([^()]*)\)')
SEE = re.compile(r'^\(see (\d+) ')


def observations(path, cycles, team_left, team_right):
    events = np.full((len(cycles), 24, 6), np.nan, dtype=np.float32)
    indices = {cycle: i for i, cycle in enumerate(cycles)}
    messages = {}
    with gzip.open(path / "raw_stream.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dir"] != "recv" or row["stopped"] != "0":
                continue
            match = SEE.match(row["msg"])
            if match and int(match[1]) in indices:
                messages[int(match[1])] = row["msg"]
    for cycle, message in messages.items():
        measurements = []
        for kind, team, unum, fields in PLAYER.findall(message):
            tokens = fields.split()
            if len(tokens) < 2:
                continue
            side = 1 if team == team_left else -1 if team == team_right else 0
            identity = int(unum) + (11 if side == -1 else 0) if unum and side else 0
            motion = [np.nan, np.nan]
            if len(tokens) >= 4:
                try:
                    motion = [float(tokens[2]), float(tokens[3])]
                except ValueError:
                    pass
            measurements.append([identity, side, float(tokens[0]), float(tokens[1]), *motion])
        if len(measurements) > events.shape[1]:
            raise ValueError(f"{path}: too many visible players")
        events[indices[cycle], :len(measurements)] = np.asarray(measurements).reshape(-1, 6)
    return events, len(messages)


def observer(path):
    meta = json.loads((path / "meta.json").read_text())
    config = meta["config"]
    if not (meta["final"] and meta["status"] == "ok" and meta["schema_version"] == 2
            and not config["use_fullstate"] and config["debug_fullstate"]
            and not config["gaussian_see"] and not config["see_noise_free"]
            and float(meta["server_param"]["quantize_step"]) == .1):
        raise ValueError(f"{path}: need finalized default-quantized reference logs")
    a = pd.read_csv(path / "agent_cycle.csv.gz", usecols=AGENT)
    a = a[a.game_mode.eq("play_on") & a.decision_made.eq(1) & a.stopped.eq(0)
          & a[POSE[:3]].notna().all(axis=1)].sort_values("cycle")
    cycles = a.cycle.to_numpy(np.int32)
    if len(cycles) != len(np.unique(cycles)):
        raise ValueError(f"{path}: duplicate PlayOn cycles")
    own_team, their_team = meta["team_name"], meta["opponent_team_name"]
    left, right = (own_team, their_team) if meta["side"] == "l" else (their_team, own_team)
    events, seen = observations(path, cycles[a.see_used.eq(1).to_numpy()], left, right)
    if seen != int(a.see_used.eq(1).sum()):
        raise ValueError(f"{path}: see messages missing from raw stream ({seen})")
    all_events = np.full((len(a), *events.shape[1:]), np.nan, np.float32)
    all_events[a.see_used.eq(1)] = events
    o = pd.read_csv(path / "objects.csv.gz", usecols=OBJECT, dtype={"track_match":str})
    o = o[o.object_id.ne(0) & o.stopped.eq(0) & o.cycle.isin(cycles)]
    ids = np.array(sorted(o.object_id.unique()), dtype=np.int32)
    if len(ids) != 21:
        raise ValueError(f"{path}: expected 21 target slots")
    o = o.set_index(["cycle", "object_id"], verify_integrity=True).reindex(pd.MultiIndex.from_product([cycles,ids]))
    truth = o[["tru_pos_x","tru_pos_y","true_dist","true_dir","true_focus_dist"]].to_numpy(np.float32)
    truth[~(o.truth_valid.eq(1)&o.tru_on_field.eq(1)).to_numpy()] = np.nan
    baseline = o[["trk_pos_x","trk_pos_y"]].where(o.track_match.eq("unum"),np.nan).to_numpy(np.float32)
    return cycles, a[POSE].fillna(0).to_numpy(np.float32), all_events, ids, truth.reshape(-1,21,5), baseline.reshape(-1,21,2)


def make_game(job):
    game, split, number, output = job
    count = 0
    for side in "lr":
        data = observer(game / f"{side}_{number}")
        np.savez(output / f"{game.name}-{side}.npz", **dict(zip(("cycle","pose","events","ids","truth","wm"),data)))
        count += len(data[0])
    return {"name":game.name,"split":split,"observer_unum":number,"decisions":count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    games = sorted(path for path in a.source.iterdir() if (path / "l_1/meta.json").is_file())
    if a.limit is not None:
        games = games[:a.limit]
    if len(games)<5 or a.workers<1:
        p.error("at least five games and one worker are required")
    rng = np.random.default_rng(42)
    rng.shuffle(games)
    heldout=max(1,len(games)//10)
    splits=["test"]*heldout+["validation"]*heldout+["train"]*(len(games)-2*heldout)
    a.output.mkdir(parents=True,exist_ok=False)
    jobs=[(g,s,int(rng.integers(1,12)),a.output) for g,s in zip(games,splits)]
    records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, record in enumerate(pool.map(make_game,jobs),1):
            records.append(record)
            if i==1 or i%10==0 or i==len(jobs):
                print(f"Built {i}/{len(jobs)} normal-view games",flush=True)
    (a.output/"manifest.json").write_text(json.dumps({"version":1,"games":records},indent=2)+"\n")


if __name__=="__main__":
    main()
