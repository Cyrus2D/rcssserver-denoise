#!/usr/bin/env python3
"""Summarize one schema-v2 denoise experiment from its per-agent logs."""

import argparse
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd

from . import denoise_hearing as hearing

SELF = {
    "n_est_pos_d": ("Self position", "m"),
    "n_est_vel_d": ("Self velocity", "m/cycle"),
    "n_est_body": ("Self body", "°"),
    "n_est_face": ("Self face", "°"),
    "n_est_neck": ("Self neck", "°"),
    "n_rawfix_pos_d": ("Visual self fix", "m"),
    "n_rawfix_face": ("Visual self face", "°"),
    "n_sb_speed_mag": ("Sense-body speed", "m/cycle"),
    "n_sb_speed_dir": ("Sense-body speed direction", "°"),
    "see_used": ("See used", "%"),
}

RAW = {
    "n_dist": ("Raw distance", "m"),
    "n_dist_rel": ("Raw relative distance", "%"),
    "n_dir": ("Raw bearing", "°"),
    "n_dist_chg": ("Raw radial velocity", "m/cycle"),
    "n_dir_chg": ("Raw angular velocity", "°/cycle"),
    "n_body_rel": ("Raw body", "°"),
    "n_head_rel": ("Raw head", "°"),
    "n_ls_rpos_d": ("Visual relative position", "m"),
    "n_ls_pos_d": ("Visual global position", "m"),
    "n_ls_vel_d": ("Visual global velocity", "m/cycle"),
    "n_ls_rvel_d": ("Visual relative velocity", "m/cycle"),
}

TRACK = {
    "n_trk_pos_d": ("Track position", "m"),
    "n_trk_vel_d": ("Track velocity", "m/cycle"),
    "n_trk_body": ("Track body", "°"),
    "n_trk_face": ("Track face", "°"),
}
HEARD = {
    "n_trk_heard_pos_d": ("Stored heard position", "m"),
    "n_trk_heard_vel_d": ("Stored heard velocity", "m/cycle"),
}
PAYLOAD = {
    "n_ball_pos_d": ("Decoded ball position", "m"),
    "n_ball_vel_d": ("Decoded ball velocity", "m/cycle"),
    "n_goalie_pos_d": ("Decoded goalie position", "m"),
}
OBJECT_RATES = {
    "seen": ("Target seen", "%"),
    "track_found": ("Track found", "%"),
    "trk_is_ghost": ("Ghost among tracked targets", "%"),
    "team_visible": ("Team visible among seen players", "%"),
    "unum_visible": ("Uniform number visible among seen players", "%"),
    "raw_velocity_available": ("Velocity fields among seen targets", "%"),
    "raw_body_available": ("Body field among seen players", "%"),
    "raw_head_available": ("Head field among seen players", "%"),
}
PREDICTIONS = {
    "self_pos": ("Self position", "m"),
    "self_body": ("Self body", "°"),
    "ball_pos": ("Ball position", "m"),
    "ball_vel": ("Ball velocity", "m/cycle"),
}
NEXT = {
    f"{method}_{field}": (f"{label} ({method})", unit)
    for field, (label, unit) in PREDICTIONS.items()
    for method in ("prediction", "persistence")
}
RATES = {"see_used", "heard_better", "heard_worse", *OBJECT_RATES}
METRICS = SELF | RAW | TRACK | HEARD | PAYLOAD | OBJECT_RATES | NEXT | hearing.PAYLOAD_METRICS | hearing.CANDIDATE_METRICS
AGES = ("0", "1", "2", "3–5", "6–10", "11+", "Unknown")
VIEWS = ("Narrow", "Normal", "Wide", "Other")
DISTANCES = ("0–3", "3–10", "10–20", "20–40", "40–60", "60+", "Unknown")
TARGETS = ("Ball", "Teammate", "Opponent")
VISION = ("Seen", "No see used", "Outside cone", "In cone, unseen", "Unknown cone")
HEARING = ("Fresh", "Older", "No recorded report")
COMMANDS = ("None", "Dash", "Turn", "Kick", "Other")
MATCHES = ("Ball", "Known identity", "Nearest truth", "No track")
SEE = ("No see used", "See used")
HEARD_DISTANCES = ("<10", "10–39", "40+", "Unknown")
HEARD_AGES = ("0", "1–2", "3–5", "6+", "Unknown")


@dataclass(frozen=True)
class Panel:
    name: str
    dimensions: tuple
    metrics: tuple

    @property
    def size(self):
        return math.prod(len(labels) for labels in self.dimensions)

    @property
    def groups(self):
        return product(*self.dimensions)


def panel(name, dimensions, metrics):
    return Panel(name, dimensions, tuple(metrics))


SELF_ANGLES = ("n_est_pos_d", "n_est_vel_d", "n_est_body", "n_est_face", "n_est_neck")
HEADLINE_TRACK = ("n_trk_pos_d", "n_trk_vel_d", "n_trk_body", "n_trk_face")
TRACK_RATES = HEADLINE_TRACK + ("track_found", "seen")
FRESH = ("n_rawfix_pos_d", "n_est_pos_d")
PANELS = {
    p.name: p for p in (
        panel("self", (), SELF),
        panel("self_view", (VIEWS,), SELF),
        panel("self_see", (SEE,), SELF_ANGLES),
        panel("self_position_age", (AGES,), SELF_ANGLES),
        panel("self_velocity_age", (AGES,), ("n_est_vel_d",)),
        panel("self_face_age", (AGES,), ("n_est_face", "n_est_body")),
        panel("self_paired_fresh", (), FRESH + ("n_rawfix_face", "n_est_face")),
        panel("self_paired_by_view", (VIEWS,), FRESH + ("n_rawfix_face", "n_est_face")),
        panel("objects", (TARGETS,), RAW | TRACK | HEARD | OBJECT_RATES),
        panel("object_distance", (TARGETS, DISTANCES), RAW | TRACK | OBJECT_RATES),
        panel("object_view", (TARGETS, VIEWS), TRACK | OBJECT_RATES),
        panel("object_visibility", (TARGETS, VISION), TRACK_RATES),
        panel("object_visual_age", (TARGETS, AGES), TRACK_RATES),
        panel("object_visual_age_under_50m", (TARGETS, AGES), ("n_trk_pos_d",)),
        panel("object_position_age", (TARGETS, AGES), HEADLINE_TRACK),
        panel("object_velocity_age", (TARGETS, AGES), ("n_trk_vel_d",)),
        panel("object_body_age", (TARGETS, AGES), ("n_trk_body",)),
        panel("object_face_age", (TARGETS, AGES), ("n_trk_face",)),
        panel("object_view_visual_age", (TARGETS, VIEWS, AGES), TRACK_RATES),
        panel("object_distance_visual_age", (TARGETS, DISTANCES, AGES), TRACK_RATES),
        panel("object_track_identity", (TARGETS, MATCHES), HEADLINE_TRACK),
        panel("object_hearing", (TARGETS, HEARING), TRACK | HEARD),
        panel("object_hearing_age_distance", (TARGETS, HEARD_AGES, HEARD_DISTANCES, HEARING),
              ("n_trk_pos_d", "n_trk_vel_d", "n_trk_body", "n_trk_face", "n_trk_heard_pos_d", "n_trk_heard_vel_d")),
        panel("object_reacquired", (TARGETS, AGES), ("n_trk_pos_d", "n_trk_vel_d")),
        panel("object_reacquired_previous", (TARGETS, AGES), ("n_trk_pos_d", "n_trk_vel_d")),
        panel("object_paired_fresh", (TARGETS,), ("n_ls_pos_d", "n_trk_pos_d", "n_ls_vel_d", "n_trk_vel_d")),
        panel("landmark_distance", (DISTANCES,), ("n_dist", "n_dir")),
        panel("heard_payload", (), PAYLOAD),
        panel("heard_message_payload", (hearing.TARGETS, hearing.MESSAGE_TYPES), hearing.PAYLOAD_METRICS),
        panel("heard_candidate_context", (hearing.TARGETS, hearing.CONTEXTS), hearing.CANDIDATE_METRICS),
        panel("heard_candidate_distance", (hearing.TARGETS, hearing.CONTEXTS, hearing.DISTANCES), hearing.CANDIDATE_METRICS),
        panel("heard_candidate_message", (hearing.TARGETS, hearing.REFERENCES, hearing.MESSAGE_TYPES),
              hearing.CANDIDATE_METRICS),
        panel("next_cycle", (), NEXT),
        panel("next_cycle_command", (COMMANDS,), NEXT),
        panel("next_cycle_view", (VIEWS,), NEXT),
        panel("next_cycle_ball_visual_age", (AGES,),
              tuple(key for key in NEXT if "_ball_" in key)),
    )
}

# Zero is a separate bin. Positive histogram bins have at most 2.9% width
# relative to their lower bound between 1e-6 and 1e6. Counts and means are exact
# apart from floating-point summation, including when workers are merged.
EDGES = np.geomspace(1e-6, 1e6, 1001)
HIST_SIZE = len(EDGES) + 2


class Summary:
    def __init__(self):
        self.coverage = Counter()
        self.settings = Counter()
        self.eligible = {}
        self.stats = {}

    def add(self, name, values, *dimensions):
        spec = PANELS[name]
        count = len(next(iter(values.values())))
        groups = (
            np.ravel_multi_index(dimensions, tuple(map(len, spec.dimensions)))
            if dimensions else np.zeros(count, dtype=np.intp)
        )
        eligible = np.bincount(groups, minlength=spec.size)
        if name not in self.eligible:
            self.eligible[name] = eligible
        else:
            self.eligible[name] += eligible
        for metric in spec.metrics:
            value = values[metric]
            valid = np.isfinite(value)
            if not valid.any():
                continue
            group = groups[valid]
            value = value[valid]
            absolute = np.abs(value)
            totals = np.vstack((
                np.bincount(group, minlength=spec.size),
                np.bincount(group, weights=absolute, minlength=spec.size),
                np.bincount(group, weights=value, minlength=spec.size),
            ))
            hist = None
            if metric not in RATES:
                bins = np.where(absolute == 0, 0, np.searchsorted(EDGES, absolute, side="right") + 1)
                hist = np.bincount(group * HIST_SIZE + bins, minlength=spec.size * HIST_SIZE)
                hist = hist.reshape(spec.size, HIST_SIZE)
            self.merge_stats((name, metric), totals, hist)

    def merge_stats(self, key, totals, hist):
        if key in self.stats:
            previous, previous_hist = self.stats[key]
            previous += totals
            if hist is not None:
                previous_hist += hist
        else:
            self.stats[key] = (totals, hist)

    def merge(self, other):
        self.coverage.update(other.coverage)
        self.settings.update(other.settings)
        for key, value in other.eligible.items():
            if key in self.eligible:
                self.eligible[key] += value
            else:
                self.eligible[key] = value
        for key, (totals, hist) in other.stats.items():
            self.merge_stats(key, totals, hist)

    def rows(self):
        for name, spec in PANELS.items():
            if name not in self.eligible:
                continue
            for i, group in enumerate(spec.groups):
                eligible = int(self.eligible[name][i])
                if eligible == 0:
                    continue
                for metric in spec.metrics:
                    label, unit = METRICS[metric]
                    totals, hist = self.stats.get((name, metric), (None, None))
                    n = int(totals[0, i]) if totals is not None else 0
                    quantiles = (
                        [percentile(hist[i], n, q) for q in (0.5, 0.9)]
                        if n and hist is not None else [None, None]
                    )
                    yield {
                        "panel": name, "group": " / ".join(group) or "All",
                        "metric": metric, "label": label, "unit": unit, "eligible": eligible,
                        "n": n, "coverage_pct": 100 * n / eligible,
                        "mean": totals[1, i] / n if n else None,
                        "median": quantiles[0], "p90": quantiles[1],
                        "bias": totals[2, i] / n if n and metric not in RATES else None,
                    }


def percentile(hist, n, q):
    cumulative = np.cumsum(hist)
    rank = q * n
    index = int(np.searchsorted(cumulative, rank))
    if index == 0:
        return 0.0
    if index == 1:
        return EDGES[0] / 2
    if index > len(EDGES):
        return math.inf
    within = (rank - cumulative[index - 1]) / hist[index]
    return EDGES[index - 2] * (EDGES[index - 1] / EDGES[index - 2]) ** within


KEYS = ("cycle", "stopped")
AGENT = (
    *KEYS, "game_mode", "decision_made", "truth_valid", "view_width", "body_cmd",
    "est_pos_count", "est_vel_count", "est_face_count", *SELF,
    "est_pos_x", "est_pos_y", "est_body", "tru_pos_x", "tru_pos_y", "tru_body",
    "next_self_pos_x", "next_self_pos_y", "next_self_body", "next_ball_pos_x",
    "next_ball_pos_y", "next_ball_vel_x", "next_ball_vel_y",
)
OBJECT = (
    *KEYS, "object_id", "is_teammate", "truth_valid", "tru_on_field", "true_dist",
    "in_cone", "seen", "match_ambiguous", "match_method", "track_found", "track_match", "is_goalie",
    "trk_is_ghost", "trk_pos_count", "trk_seen_pos_count", "trk_vel_count",
    "trk_body_count", "trk_face_count", "trk_heard_pos_count", "team_visible", "unum_visible",
    "dist_chg", "dir_chg", "body_rel", "head_rel", "trk_pos_x", "trk_pos_y",
    "trk_vel_x", "trk_vel_y", "tru_pos_x", "tru_pos_y", "tru_vel_x", "tru_vel_y",
    *RAW, *TRACK, *HEARD,
)
CATEGORICAL = {"game_mode", "view_width", "body_cmd", "match_method", "track_match", "kind", "sender_kind"}
TEXT_COLUMNS = {"raw_msg", "decoded"}


def read_table(directory, name, columns):
    path = directory / f"{name}.csv.gz"
    if not path.exists():
        path = directory / f"{name}.csv"
    try:
        return pd.read_csv(path, usecols=columns, keep_default_na=False, na_values=[""],
                           dtype={key: "category" if key in CATEGORICAL else str if key in TEXT_COLUMNS
                                  else np.float64 for key in columns})
    except (ValueError, OSError, EOFError) as error:
        raise ValueError(f"{path}: {error}") from error


def has_table(directory, name):
    return any((directory / f"{name}{suffix}").is_file() for suffix in (".csv.gz", ".csv"))


def select(values, keys):
    return {key: values[key].to_numpy() for key in keys}


def age(values):
    return bins(np.where(values >= 1000, np.nan, values), [1, 2, 3, 6, 11])


def bins(values, limits):
    return np.searchsorted(limits, values, side="right") + np.isnan(values)


def row_key(frame):
    cycle, stopped = (frame[key].to_numpy(dtype=np.int64) for key in KEYS)
    if np.any((stopped < 0) | (stopped >= 2**20)):
        raise ValueError("Invalid stopped counter")
    return cycle * 2**20 + stopped


def mask_pair(values, first, second):
    paired = np.isfinite(values[first]) & np.isfinite(values[second])
    return {key: np.where(paired, values[key], np.nan) for key in (first, second)}


def summarize_self(summary, agent):
    values = select(agent, SELF)
    values["see_used"] = values["see_used"] * 100
    view = agent["view"]
    pos, vel, face = (age(agent[key].to_numpy()) for key in ("est_pos_count", "est_vel_count", "est_face_count"))
    paired = mask_pair(values, *FRESH) | mask_pair(values, "n_rawfix_face", "n_est_face")
    summary.add("self", values)
    summary.add("self_view", values, view)
    summary.add("self_see", values, agent["visual"])
    summary.add("self_position_age", values, pos)
    summary.add("self_velocity_age", values, vel)
    summary.add("self_face_age", values, face)
    summary.add("self_paired_fresh", paired)
    summary.add("self_paired_by_view", paired, view)


def summarize_objects(summary, objects, agent_index, view, visual):
    values = select(objects, RAW | TRACK | HEARD)
    target = np.where(objects["object_id"] == 0, 0, np.where(objects["is_teammate"] == 1, 1, 2))
    dist = objects["true_dist"].to_numpy()
    distance = bins(dist, [3, 10, 20, 40, 60])
    seen = objects["seen"].to_numpy() == 1
    cone = objects["in_cone"].to_numpy()
    tracked = objects["track_found"].to_numpy() == 1
    raw_match = (objects["match_ambiguous"].to_numpy() != 1)
    for key in RAW:
        values[key] = np.where(seen & raw_match, values[key], np.nan)
    values["n_dist_rel"] *= 100
    for key in ("seen", "track_found"):
        values[key] = objects[key].to_numpy() * 100
    values["trk_is_ghost"] = np.where(tracked, objects["trk_is_ghost"] * 100, np.nan)
    for key in ("team_visible", "unum_visible"):
        values[key] = np.where(seen & (target != 0), objects[key] * 100, np.nan)
    for key, col in (("raw_body_available", "body_rel"), ("raw_head_available", "head_rel")):
        values[key] = np.where(seen & (target != 0), objects[col].notna() * 100, np.nan)
    values["raw_velocity_available"] = np.where(
        seen, (objects["dist_chg"].notna() & objects["dir_chg"].notna()) * 100, np.nan,
    )
    visibility = np.select([seen, ~visual.astype(bool), cone == 0, cone == 1], [0, 1, 2, 3], default=4)
    visual_count = objects["trk_seen_pos_count"].to_numpy()
    visual_age = age(visual_count)
    hearing_count = objects["trk_heard_pos_count"].to_numpy()
    hearing = np.select([hearing_count == 0, (hearing_count > 0) & (hearing_count < 1000)], [0, 1], default=2)
    hear_age = bins(np.where(visual_count >= 1000, np.nan, visual_count), [1, 3, 6])
    hear_dist = bins(dist, [10, 40])
    identity = np.select(
        [target == 0, objects["track_match"] == "unum", objects["track_match"] == "nearest"],
        [0, 1, 2], default=3,
    )
    summary.coverage["ambiguous_raw_matches"] += int(np.count_nonzero(seen & ~raw_match))
    summary.coverage["seen_objects"] += int(np.count_nonzero(seen))
    summary.coverage["object_rows"] += len(objects)
    summary.add("objects", values, target)
    summary.add("object_distance", values, target, distance)
    summary.add("object_view", values, target, view)
    summary.add("object_visibility", values, target, visibility)
    summary.add("object_visual_age", values, target, visual_age)
    under_50m = (dist >= 0) & (dist < 50)
    summary.add("object_visual_age_under_50m",
                {"n_trk_pos_d": values["n_trk_pos_d"][under_50m]},
                target[under_50m], visual_age[under_50m])
    summary.add("object_position_age", values, target, age(objects["trk_pos_count"].to_numpy()))
    summary.add("object_velocity_age", values, target, age(objects["trk_vel_count"].to_numpy()))
    summary.add("object_body_age", values, target, age(objects["trk_body_count"].to_numpy()))
    summary.add("object_face_age", values, target, age(objects["trk_face_count"].to_numpy()))
    summary.add("object_view_visual_age", values, target, view, visual_age)
    summary.add("object_distance_visual_age", values, target, distance, visual_age)
    summary.add("object_track_identity", values, target, identity)
    summary.add("object_hearing", values, target, hearing)
    summary.add("object_hearing_age_distance", values, target, hear_age, hear_dist, hearing)
    paired = mask_pair(values, "n_ls_pos_d", "n_trk_pos_d") | mask_pair(values, "n_ls_vel_d", "n_trk_vel_d")
    summary.add("object_paired_fresh", paired, target)

    # Use the same object and an immediately preceding, eligible play_on cycle.
    ids = objects["object_id"].to_numpy(dtype=np.int64)
    keys = agent_index * 32 + ids
    previous = pd.Index(keys).get_indexer(keys - (2**20 * 32))
    current = np.flatnonzero(seen & (visual_age == 0) & (previous >= 0)
                             & (objects["stopped"].to_numpy() == 0))
    before = previous[current]
    comparable = ~seen[before] & raw_match[current]
    before, current = before[comparable], current[comparable]
    paired = {
        key: np.where(np.isfinite(values[key][before]) & np.isfinite(values[key][current]),
                      values[key][current], np.nan)
        for key in ("n_trk_pos_d", "n_trk_vel_d")
    }
    previous_values = {key: np.where(np.isfinite(val), values[key][before], np.nan)
                       for key, val in paired.items()}
    summary.add("object_reacquired", paired, target[current], visual_age[before])
    summary.add("object_reacquired_previous", previous_values, target[current], visual_age[before])


def predict(summary, agent, ball):
    agent = agent[agent["stopped"] == 0].set_index("cycle")
    ball = ball[ball["stopped"] == 0].set_index("cycle")
    starts = agent.index[agent.index.isin(agent.index - 1)]
    now = agent.loc[starts]
    next_agent = agent.loc[starts + 1]
    now_ball = ball.reindex(starts)
    next_ball = ball.reindex(starts + 1)
    summary.coverage["next_cycle_pairs"] += len(starts)
    values = {}

    for name, source, baseline, actual, predicted, persistence, truth in (
        ("self_pos", now, now, next_agent, "next_self_pos", "est_pos", "tru_pos"),
        ("self_body", now, now, next_agent, "next_self_body", "est_body", "tru_body"),
        ("ball_pos", now, now_ball, next_ball, "next_ball_pos", "trk_pos", "tru_pos"),
        ("ball_vel", now, now_ball, next_ball, "next_ball_vel", "trk_vel", "tru_vel"),
    ):
        if name == "self_body":
            prediction = (source[predicted].to_numpy() - actual[truth].to_numpy() + 180) % 360 - 180
            held = (baseline[persistence].to_numpy() - actual[truth].to_numpy() + 180) % 360 - 180
        else:
            def error(frame, prefix):
                return np.hypot(*(frame[f"{prefix}_{axis}"].to_numpy() - actual[f"{truth}_{axis}"].to_numpy()
                                  for axis in ("x", "y")))
            prediction = error(source, predicted)
            held = error(baseline, persistence)
        valid = np.isfinite(prediction) & np.isfinite(held)
        values[f"prediction_{name}"] = np.where(valid, prediction, np.nan)
        values[f"persistence_{name}"] = np.where(valid, held, np.nan)

    command = now["body_cmd"].map({"dash": 1, "turn": 2, "kick": 3}).astype(float)
    command = np.where(now["body_cmd"].isna(), 0, command.fillna(4)).astype(int)
    summary.add("next_cycle", values)
    summary.add("next_cycle_command", values, command)
    summary.add("next_cycle_view", values, now["view"])
    summary.add("next_cycle_ball_visual_age", values, age(now_ball["trk_seen_pos_count"].to_numpy()))


SERVER_SETTINGS = ("full_view", "see_noise_free", "see_no_fade_unum", "see_no_fade_all", "say_msg_size", "synch_mode")
CLIENT_SETTINGS = ("version", "gaussian_see", "use_fullstate", "debug_fullstate")


def summarize_agent(summary, directory):
    meta = json.loads((directory / "meta.json").read_text())
    if meta["schema_version"] != 2:
        raise ValueError(f"{directory}: expected schema version 2")
    if not meta["final"] or meta["status"] != "ok":
        raise ValueError(f"{directory}: the agent log did not finalize successfully")
    if meta["config"]["use_fullstate"]:
        raise ValueError(f"{directory}: estimates use fullstate, expected separate reference truth")
    settings = tuple(str(meta[section].get(key, "missing")).lower()
                     for section, keys in (("server_param", SERVER_SETTINGS), ("config", CLIENT_SETTINGS))
                     for key in keys)
    summary.settings[settings] += 1
    summary.coverage["agent_logs"] += 1
    agent = read_table(directory, "agent_cycle", AGENT)
    play_on = (agent["game_mode"] == "play_on") & (agent["decision_made"] == 1)
    summary.coverage["play_on_rows"] += int(play_on.sum())
    summary.coverage["missing_truth"] += int((play_on & (agent["truth_valid"] != 1)).sum())
    summary.coverage["skipped_decisions"] += int((agent["decision_made"] == 0).sum())
    agent = agent.loc[play_on & (agent["truth_valid"] == 1)].copy()
    agent.index = row_key(agent)
    if not agent.index.is_unique:
        raise ValueError(f"{directory}: duplicate agent-cycle keys")
    agent["view"] = agent["view_width"].map({"narrow": 0, "normal": 1, "wide": 2}).astype(float).fillna(3).astype(int)
    agent["visual"] = (agent["see_used"] == 1).astype(int)
    summarize_self(summary, agent)

    objects = read_table(directory, "objects", OBJECT)
    keys = row_key(objects)
    positions = agent.index.get_indexer(keys)
    valid = (positions >= 0) & (objects["truth_valid"] == 1).to_numpy() & (objects["tru_on_field"] == 1).to_numpy()
    objects, keys, positions = objects.loc[valid], keys[valid], positions[valid]
    summarize_objects(summary, objects, keys, agent["view"].to_numpy()[positions],
                      agent["visual"].to_numpy()[positions])
    predict(summary, agent, objects[objects["object_id"] == 0])

    if has_table(directory, "landmarks"):
        landmarks = read_table(directory, "landmarks", (*KEYS, "kind", "true_dist", "n_dist", "n_dir"))
        eligible = agent.index.get_indexer(row_key(landmarks)) >= 0
        landmarks = landmarks[eligible & landmarks["kind"].isin(["flag", "goal"])]
        values = select(landmarks, ("n_dist", "n_dir"))
        summary.add("landmark_distance", values, bins(landmarks["true_dist"].to_numpy(), [3, 10, 20, 40, 60]))
        summary.coverage["landmark_logs"] += 1

    if has_table(directory, "hear"):
        heard = read_table(directory, "hear", (*KEYS, "hear_cycle", "sender_kind", "has_content",
                                              *PAYLOAD, *hearing.HEARING_COLUMNS))
        eligible = agent.index.get_indexer(row_key(heard)) >= 0
        heard = heard[eligible & (heard["hear_cycle"] == heard["cycle"])
                      & (heard["sender_kind"] == "teammate") & (heard["has_content"] == 1)]
        summary.add("heard_payload", select(heard, PAYLOAD))
        heard = heard.assign(key=row_key(heard), raw_msg=heard["raw_msg"].fillna(""),
                             decoded=heard["decoded"].fillna(""))
        hearing.summarize_hearing(summary, agent, objects, keys, heard, meta["side"])
        summary.coverage["hear_logs"] += 1


def run_game(directories):
    summary = Summary()
    for directory in directories:
        summarize_agent(summary, directory)
    return summary


def discover(logs, side="both"):
    directories = {path.parent for suffix in (".csv.gz", ".csv")
                   for path in logs.rglob(f"agent_cycle{suffix}")}
    games = {}
    for directory in sorted(directories):
        if side != "both" and not directory.name.startswith(side[0] + "_"):
            continue
        games.setdefault(directory.parent, []).append(directory)
    if not games:
        raise ValueError(f"No agent_cycle logs found under {logs}")
    return games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, help="Experiment log root or a single game directory")
    parser.add_argument("--output", type=Path, help="Output directory, defaults to the Exp1 reports directory")
    parser.add_argument("--side", choices=("both", "left", "right"), default="both",
                        help="Restrict samples to observers on the selected side")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    logs = args.logs.resolve()
    reports = Path(__file__).resolve().parents[1] / "reports"
    output = args.output or reports
    if args.output is None and (logs.name != "exp1" or args.side != "both"):
        suffix = "" if args.side == "both" else f"-{args.side}"
        output /= logs.name + suffix
    try:
        games = discover(logs, args.side)
        summary = Summary()
        print(f"Found {len(games)} games and {sum(map(len, games.values()))} player logs", flush=True)
        if args.workers == 1 or len(games) == 1:
            results = map(run_game, games.values())
            for result in results:
                summary.merge(result)
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(games))) as pool:
                for i, result in enumerate(pool.map(run_game, games.values()), start=1):
                    summary.merge(result)
                    if i == 1 or i % 10 == 0 or i == len(games):
                        print(f"Processed {i}/{len(games)} games", flush=True)
        if len(summary.settings) != 1:
            raise ValueError(f"Mixed experiment settings; run each configuration separately: {summary.settings}")
        if summary.coverage["play_on_rows"] == summary.coverage["missing_truth"]:
            raise ValueError("No eligible play_on decisions with reference truth")
        from .render_denoise_report import write_report
        settings = dict(zip(SERVER_SETTINGS + CLIENT_SETTINGS, next(iter(summary.settings))))
        coverage = dict(summary.coverage) | {
            "games": len(games),
            "observer_side": args.side,
            "logs_per_game": dict(sorted(Counter(map(len, games.values())).items())),
        }
        write_report(output.resolve(), logs, settings, coverage, list(summary.rows()))
        print(f"Report: {output.resolve() / 'report.html'}")
    except (OSError, ValueError, KeyError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()

