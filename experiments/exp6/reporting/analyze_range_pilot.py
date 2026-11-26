#!/usr/bin/env python3
"""Check range routing in the four short stock and updated controls."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

PILOTS = ("stock-gaussian", "new-gaussian", "stock-quantized", "new-quantized")


def audit(pilot, scanner, object_table, output):
    entries = np.array([
        (float(seen), float(mid)) for seen, mid in re.findall(
            r"M_movable_table.emplace_back\(\s*([0-9.]+),\s*([0-9.]+),\s*[0-9.]+\s*\)",
            object_table.read_text())])
    assert len(entries) > 1 and (np.diff(entries[:, 0]) > 0).all()
    results = {}
    for name in PILOTS:
        game = pilot/name/"logs"/name
        files = sorted(game.glob("*/meta.json"))
        assert len(files) == 22
        gaussian = name.endswith("-gaussian")
        raw_path = name == "new-gaussian"
        detail = {"player_checked": 0, "ball_checked": 0, "confirmed": 0,
                  "samples": {}}
        for path in files:
            meta = json.loads(path.read_text())
            cfg = meta["config"]
            assert meta["final"] and meta["status"] == "ok" and cfg["full_view"]
            assert cfg["gaussian_see"] == gaussian and cfg["debug_fullstate"] and not cfg["use_fullstate"]
            assert cfg.get("gaussian_range_localization", False) == raw_path
            raw = pd.read_csv(path.with_name("raw_stream.csv.gz"), usecols=["msg"])
            ack = raw.msg.str.startswith("(ok gaussian_see)").any()
            assert ack == gaussian
            detail["confirmed"] += int(ack)
            key = ["cycle", "stopped"]
            agent = pd.read_csv(path.with_name("agent_cycle.csv.gz"), usecols=key+[
                "game_mode","decision_made","truth_valid","see_used","see_count_this_cycle","view_angle_deg"])
            play = agent[agent.game_mode.eq("play_on") & agent.decision_made.eq(1)]
            assert play[["truth_valid","see_used","see_count_this_cycle"]].eq(1).all().all()
            assert play.view_angle_deg.eq(360).all()
            objects = pd.read_csv(path.with_name("objects.csv.gz"), usecols=key+[
                "object_id","seen","dist","match_method","ls_rpos_x","ls_rpos_y","ls_dist_error"])
            data = objects.merge(play[key],on=key,validate="many_to_one")
            data = data[data.seen.eq(1) & (data.object_id.eq(0) | data.match_method.eq("unum"))]
            data = data.dropna(subset=["dist","ls_rpos_x","ls_rpos_y"])
            ranges = data.dist.to_numpy()
            localized = np.hypot(data.ls_rpos_x, data.ls_rpos_y)
            if raw_path:
                matched = np.abs(localized - ranges) <= .00015
            else:
                def matches(offset):
                    index = np.searchsorted(entries[:,0], ranges-.001+offset)
                    midpoint = entries[np.minimum(index,len(entries)-1),1]
                    return (index < len(entries)) & (np.abs(localized-midpoint) <= .00015)
                # A rounded CSV range can fall on either side of the lookup boundary.
                matched = matches(0.) | matches(-.00005) | matches(.00005)
            assert matched.all(), (path,int((~matched).sum()))
            if raw_path:
                player = data[data.object_id.ne(0)]
                # The pilot used zero focus, so two sigma equals 5% of observed range.
                assert np.allclose(player.ls_dist_error, player.dist*.05, rtol=0, atol=.00006), path
            detail["ball_checked"] += int(data.object_id.eq(0).sum())
            detail["player_checked"] += int(data.object_id.ne(0).sum())
        summary = subprocess.check_output([str(scanner.resolve()),str(game),".0125",".0125",".00125",".00125"],
                                          text=True)
        keep = {"track_position","paired_visual","visual_relative","self_position","self_fix_position","track_velocity"}
        combined = {}
        for line in summary.splitlines():
            fields = line.split("\t")
            if fields[0] == "coverage":
                detail[fields[1]] = int(fields[2])
                continue
            target,_,metric,n,total,_ = fields
            if target not in {"ball","all","self"} or metric not in keep:
                continue
            row = combined.setdefault(target+"_"+metric,[0,0.])
            row[0] += int(n)
            row[1] += float(total)
        detail["samples"] = {name:{"n":n,"mean":total/n} for name,(n,total) in combined.items()}
        results[name] = detail
        print(name,detail["player_checked"],detail["ball_checked"],flush=True)
    output.write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot","scanner","object_table","output"):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    audit(args.pilot,args.scanner,args.object_table,args.output)
