#!/usr/bin/env python3
"""Pool player observations and quantization diagnostics across finalized games."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

DISTANCES = ("0–3", "3–10", "10–20", "20–30", "30–40", "40–60", "60+")
KEY = ["cycle", "stopped"]
AGENT = KEY + ["game_mode", "decision_made", "truth_valid", "see_used", "see_count_this_cycle",
               "view_angle_deg", "view_quality", "say_msg"]
OBJECT = KEY + ("object_id truth_valid tru_on_field is_teammate true_dist true_dir seen unum_visible "
                "match_method match_ambiguous dist dir dist_chg dir_chg body_rel head_rel "
                "n_dist_chg n_dir_chg n_body_rel n_head_rel n_ls_rpos_d n_ls_pos_d n_trk_pos_d "
                "n_ls_vel_d n_trk_vel_d").split()
SETTINGS = ("synch_mode", "full_view", "see_no_fade_unum", "see_no_fade_all",
            "see_noise_free", "say_msg_size")


def collect(game):
    total, settings = Counter(), Counter()
    groups = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    metas = sorted(game.glob("*/meta.json"))
    if len(metas) != 22:
        raise ValueError(f"{game}: expected 22 player logs, got {len(metas)}")
    for path in metas:
        meta = json.loads(path.read_text())
        config = meta["config"]
        if (meta["schema_version"] not in (2, 3) or not meta["final"] or meta["status"] != "ok"
                or config["use_fullstate"] or not config["debug_fullstate"]):
            raise ValueError(f"{path}: need finalized schema-v2/v3 logs and reference fullstate")
        settings[tuple(str(meta["server_param"][key]) for key in SETTINGS)
                 + (str(config["gaussian_see"]),)] += 1
        agent = pd.read_csv(path.with_name("agent_cycle.csv.gz"), usecols=AGENT)
        total["skipped"] += int(agent["decision_made"].eq(0).sum())
        agent = agent[agent["game_mode"].eq("play_on") & agent["decision_made"].eq(1)]
        total["play"] += len(agent)
        total["truth"] += int(agent["truth_valid"].eq(1).sum())
        total["see"] += int(agent["see_used"].eq(1).sum())
        total["one_see"] += int(agent["see_count_this_cycle"].eq(1).sum())
        total["full_view"] += int(agent["view_angle_deg"].eq(360).sum())
        total["high_quality"] += int(agent["view_quality"].eq("high").sum())
        total["say"] += int(agent["say_msg"].notna().sum())
        hear = pd.read_csv(path.with_name("hear.csv.gz"), usecols=["sender_kind", "has_content"])
        total["teammate_payload"] += int((hear["sender_kind"].eq("teammate") & hear["has_content"].eq(1)).sum())

        objects = pd.read_csv(path.with_name("objects.csv.gz"), usecols=OBJECT, dtype={"match_method": str})
        objects = objects[objects["object_id"].ne(0) & objects["truth_valid"].eq(1)
                          & objects["tru_on_field"].eq(1) & objects["true_dist"].notna()]
        key = lambda frame: frame["cycle"].to_numpy() * (1 << 20) + frame["stopped"].to_numpy()
        play_keys = key(agent[agent["truth_valid"].eq(1)])
        objects = objects[np.isin(key(objects), play_keys)]
        distance = objects["true_dist"].to_numpy()
        bins = np.searchsorted([3, 10, 20, 30, 40, 60], distance, side="right")
        seen = objects["seen"].eq(1).to_numpy()
        known = seen & objects["unum_visible"].eq(1).to_numpy() & objects["match_method"].eq("unum").to_numpy()
        teammate = objects["is_teammate"].eq(1).to_numpy()
        valid = known & objects["match_ambiguous"].ne(1).to_numpy()
        measured = objects["dist"].to_numpy()
        bearing = (objects["dir"].to_numpy() - objects["true_dir"].to_numpy() + 180) % 360 - 180
        angular = 2 * np.abs(np.sin(np.deg2rad(bearing) / 2))
        position = np.hypot(measured - distance, np.sqrt(np.maximum(0, measured * distance)) * angular)
        metrics = {
            "seen_share": seen, "known_share": known,
            "motion_share": known & objects["dist_chg"].notna().to_numpy() & objects["dir_chg"].notna().to_numpy(),
            "body_share": known & objects["body_rel"].notna().to_numpy(),
            "head_share": known & objects["head_rel"].notna().to_numpy(),
            "track_position": objects["n_trk_pos_d"].to_numpy(),
            "track_velocity": objects["n_trk_vel_d"].to_numpy(),
            "raw_range": np.where(valid, np.abs(measured - distance), np.nan),
            "raw_bearing": np.where(valid, np.abs(bearing), np.nan),
            "raw_position": np.where(valid, position, np.nan),
            "exact_range_position": np.where(valid & np.isfinite(position), distance * angular, np.nan),
            "exact_bearing_position": np.where(valid & np.isfinite(position), np.abs(measured - distance), np.nan),
        }
        for name in ("n_dist_chg", "n_dir_chg", "n_body_rel", "n_head_rel"):
            metrics[name] = np.where(valid, np.abs(objects[name].to_numpy()), np.nan)
        for output, source in (("visual_relative", "n_ls_rpos_d"), ("visual_global", "n_ls_pos_d"),
                               ("visual_velocity", "n_ls_vel_d")):
            metrics[output] = np.where(valid, objects[source].to_numpy(), np.nan)
        paired = valid & objects["n_ls_pos_d"].notna().to_numpy() & objects["n_trk_pos_d"].notna().to_numpy()
        metrics["paired_visual"] = np.where(paired, objects["n_ls_pos_d"], np.nan)
        metrics["paired_track"] = np.where(paired, objects["n_trk_pos_d"], np.nan)

        for target, mask in (("all", np.ones(len(objects), dtype=bool)), ("teammate", teammate), ("opponent", ~teammate)):
            for name, values in metrics.items():
                keep = mask & np.isfinite(values)
                counts = np.bincount(bins[keep], minlength=len(DISTANCES))
                sums = np.bincount(bins[keep], weights=values[keep], minlength=len(DISTANCES))
                for label, n, value in zip(DISTANCES, counts, sums):
                    item = groups[target, label][name]
                    item[0] += int(n)
                    item[1] += float(value)
    return total, settings, {key: dict(value) for key, value in groups.items()}


def analyze(root, workers):
    games = sorted({path.parent.parent for path in root.glob("*/*/meta.json")})
    if not games:
        raise ValueError(f"{root}: no games")
    coverage, settings = Counter(), Counter()
    groups = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (counts, config, stats) in enumerate(pool.map(collect, games), 1):
            coverage.update(counts)
            settings.update(config)
            for key, metrics in stats.items():
                for name, (n, value) in metrics.items():
                    groups[key][name][0] += n
                    groups[key][name][1] += value
            if index == 1 or index % 10 == 0 or index == len(games):
                print(f"{root.name} {index}/{len(games)} games", flush=True)
    return {"source": str(root), "games": len(games), "coverage": dict(coverage),
            "settings": [{"players": n, **dict(zip(SETTINGS + ("gaussian_see",), key))}
                         for key, n in settings.items()],
            "metrics": [{"target": target, "distance": label, "metric": metric,
                         "n": n, "mean": value / n if n else None}
                        for (target, label), values in groups.items()
                        for metric, (n, value) in values.items()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = analyze(args.logs.resolve(), args.workers)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
