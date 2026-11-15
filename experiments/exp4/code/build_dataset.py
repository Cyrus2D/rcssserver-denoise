#!/usr/bin/env python3
"""Build causal state inputs and separate fullstate labels from Exp3 games."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["cycle", "stopped"]
POSE = "est_pos_x est_pos_y est_vel_x est_vel_y est_body est_face est_kickable focus_dist focus_dir".split()
AGENT = KEY + ["game_mode", "decision_made", "see_used", "tru_pos_x", "tru_pos_y"] + POSE
STATE = "trk_pos_x trk_pos_y trk_vel_x trk_vel_y trk_body trk_pos_count trk_vel_count trk_body_count".split()
RAW = "dist dir dist_chg dir_chg body_rel".split()
TRUTH = "tru_pos_x tru_pos_y tru_vel_x tru_vel_y tru_body".split()
OBJECT = KEY + ["object_id", "track_match", "match_method", "truth_valid", "tru_on_field", "player_type"] + STATE + RAW + TRUTH
FEATURES = 28
CONTEXT = 40
HISTORY = 8


def unit(degrees):
    rad = np.deg2rad(degrees)
    return np.stack((np.cos(rad), np.sin(rad)), axis=-1)


def nearby(state, team):
    pos, vel, direction, age = state[..., :2], state[..., 2:4], state[..., 4], state[..., 5]
    t, n = pos.shape[:2]
    output = np.zeros((t, n, CONTEXT), np.float16)
    holder = np.zeros((t, n), np.float32)
    # Ball groups are relative to observer. Player groups are relative to target.
    ids = np.arange(n)
    same = team[:, None] == team[None, :]
    allowed = (ids[None] > 0) & (ids[:, None] != ids[None])
    for start in range(0, t, 256):
        end = min(t, start + 256)
        xy, speed = pos[start:end], vel[start:end]
        d = xy[:, None] - xy[:, :, None]
        dv = speed[:, None] - speed[:, :, None]
        dist = np.linalg.norm(d, axis=-1)
        valid = np.isfinite(dist) & allowed
        indexes = []
        for relation in (same, ~same):
            dist_group = np.where(valid & relation, dist, np.inf)
            choice = np.argsort(dist_group, axis=-1, kind="stable")[..., :2]
            indexes.append(np.where(np.take_along_axis(dist_group, choice, axis=2) < np.inf, choice, 0))
        index = np.concatenate(indexes, axis=2)
        dp = np.take_along_axis(d, index[..., None], axis=2)
        vv = np.take_along_axis(dv, index[..., None], axis=2)
        row = np.arange(end - start)[:, None, None]
        body = direction[start:end][row, index]
        has_body = np.isfinite(body)
        has_pos = index > 0
        has_vel = has_pos & np.isfinite(vv).all(axis=-1)
        a = np.nan_to_num(age[start:end][row, index], nan=10)
        slots = np.concatenate((np.where(has_pos[..., None], dp / 20, 0),
                                np.where(has_vel[..., None], vv / 2, 0),
                                np.where((has_pos & has_body)[..., None], unit(body), 0),
                                np.minimum(a, 10)[..., None] / 10, has_pos[..., None],
                                has_vel[..., None], (has_pos & has_body)[..., None]), axis=-1)
        output[start:end] = slots.reshape(end - start, n, CONTEXT)
        # Estimated proximity to the ball is available in the observation WM.
        holder[start:end] = np.nan_to_num(np.linalg.norm(xy - xy[:, :1], axis=-1), nan=150)
    return output, holder


def observer(path, output):
    meta = json.loads((path / "meta.json").read_text())
    c, s = meta["config"], meta["server_param"]
    if not (meta["final"] and meta["status"] == "ok" and not c["gaussian_see"] and not c["use_fullstate"]
            and c["debug_fullstate"] and c["full_view"] and not c["see_noise_free"]
            and s["say_msg_size"] == "0" and s["see_no_fade_unum"] == "true"):
        raise ValueError(f"{path}: expected original quantized Exp3 reference logs")
    reverse = meta["side"] == "r"
    self_id = meta["unum"] + (11 if reverse else 0)
    a = pd.read_csv(path / "agent_cycle.csv.gz", usecols=AGENT)
    a = a[a.game_mode.eq("play_on") & a.decision_made.eq(1) & a.see_used.eq(1)
          & a.stopped.eq(0) & a[POSE[:2]].notna().all(axis=1)].set_index("cycle", verify_integrity=True).sort_index()
    o = pd.read_csv(path / "objects.csv.gz", usecols=OBJECT)
    o = o[o.stopped.eq(0) & o.cycle.isin(a.index)].set_index(["cycle", "object_id"], verify_integrity=True)
    o = o.reindex(pd.MultiIndex.from_product([a.index, range(23)], names=["cycle", "object_id"]))
    state = o[STATE].to_numpy(np.float32).reshape(len(a), 23, -1)
    track = (o.track_match.eq("unum") | o.track_match.eq("ball")).to_numpy().reshape(len(a), 23)
    state[~track] = np.nan
    state[:, self_id, :5] = a[POSE[:5]].to_numpy(np.float32)
    state[:, self_id, 5:] = 0
    raw = o[RAW].to_numpy(np.float32).reshape(len(a), 23, -1)
    identified = (o.match_method.eq("unum") | (o.index.get_level_values("object_id") == 0)).to_numpy()
    raw[~identified.reshape(len(a), 23)] = np.nan
    truth = o[TRUTH].to_numpy(np.float32).reshape(len(a), 23, -1)
    truth[~(o.truth_valid.eq(1) & o.tru_on_field.eq(1)).to_numpy().reshape(len(a), 23)] = np.nan
    truth[:, self_id] = np.nan
    agent = a[POSE].to_numpy(np.float32)
    if reverse:
        state[..., :4] *= -1
        truth[..., :4] *= -1
        state[..., 4] += 180
        truth[..., 4] += 180
        agent[:, :4] *= -1
        agent[:, 5] += 180
    team = (np.arange(23) <= 11) != reverse
    team[0] = True
    context, to_ball = nearby(state, team)
    pos, vel, body = state[..., :2], state[..., 2:4], state[..., 4]
    have_pos = np.isfinite(pos).all(axis=-1)
    have_vel = np.isfinite(vel).all(axis=-1)
    have_body = np.isfinite(body)
    own = lambda v: np.broadcast_to(v[:, None], (len(a), 23, v.shape[-1]))
    estimated_area = sum(float(s[k]) for k in ("player_size", "kickable_margin", "ball_size"))
    dist_to_player = np.where(have_pos, to_ball, 150)
    dist_to_player[:, 0] = 150
    near = dist_to_player.min(axis=1)
    features = np.concatenate((
        pos / 55, vel / 2, unit(body), np.minimum(10, state[..., 5:8]) / 10,
        have_pos[..., None], have_vel[..., None], have_body[..., None],
        raw[..., :1] / 60, unit(raw[..., 1]), raw[..., 2:3], raw[..., 3:4] / 10,
        np.isfinite(raw[..., 2:3]), np.isfinite(raw[..., 4:5]),
        own(agent[:, :2] / 55), own(agent[:, 2:4] / 2), own(unit(agent[:, 5])),
        np.broadcast_to(team, have_pos.shape)[..., None], np.minimum(to_ball, 30)[..., None] / 30,
        own(np.minimum(near, 30)[:, None] / 30),
    ), axis=-1)
    assert features.shape[-1] == FEATURES
    # Estimated proximity is a soft ownership proxy. Include the observer's
    # own kickability, and use exact truth only for the evaluation categories.
    controlled = (near <= estimated_area) | (agent[:, 6] == 1)
    radii = {p["id"]: p["player_size"] + p["kickable_margin"] + float(s["ball_size"]) for p in meta["player_types"]}
    radii = o.player_type.map(radii).fillna(estimated_area).to_numpy(np.float32).reshape(len(a), 23)
    truth_players = truth[..., :2].copy()
    truth_players[:, self_id] = a[["tru_pos_x", "tru_pos_y"]].to_numpy() * (-1 if reverse else 1)
    true_dist = np.linalg.norm(truth_players - truth[:, :1, :2], axis=-1)
    true_control = (true_dist[:, 1:] <= radii[:, 1:]).any(axis=1)
    np.save(output / "input.npy", np.nan_to_num(features).astype(np.float32), allow_pickle=False)
    np.save(output / "context.npy", context, allow_pickle=False)
    np.save(output / "state.npy", state, allow_pickle=False)
    np.save(output / "truth.npy", truth, allow_pickle=False)
    np.savez(output / "timing.npz", cycles=a.index.to_numpy(np.int32), controlled=controlled,
             true_control=true_control, ball_decay=float(s["ball_decay"]), self_id=self_id)
    return len(a)


def build(job):
    source, output, game = job
    name, number = game["name"], game["observer_unum"]
    counts = {}
    for side in "lr":
        target = output / f"{name}-{side}"
        target.mkdir()
        counts[side] = observer(source / name / f"{side}_{number}", target)
    return {**game, "rows": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "split_manifest", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    manifest = json.loads(args.split_manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(args.source, args.output, g) for g in manifest["games"]]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(build, jobs), 1):
            records.append(record)
            if i == 1 or i % 10 == 0:
                print(f"Built {i}/{len(jobs)} games", flush=True)
    (args.output / "manifest.json").write_text(json.dumps({"games": records}, indent=2) + "\n")
