#!/usr/bin/env python3
"""Verify that logged player localization uses the stock movable range table."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def replay(logs, source):
    games = sorted({p.parent.parent for p in logs.glob("*/*/meta.json")})
    entries = np.array([
        [float(x) for x in row]
        for row in re.findall(
            r"M_movable_table.emplace_back\(\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\s*\)",
            source.read_text())
    ])
    if len(entries) < 2 or not (np.diff(entries[:, 0]) > 0).all():
        raise ValueError("Could not read ordered movable distance table")
    records = []
    key = ["cycle", "stopped"]
    tolerance = .00015
    for game in (games[0], games[-1]):
        for observer in ("l_1", "r_1"):
            path = game/observer
            agent = pd.read_csv(path/"agent_cycle.csv.gz", usecols=key+["game_mode","decision_made","truth_valid"])
            valid = agent.game_mode.eq("play_on") & agent.decision_made.eq(1) & agent.truth_valid.eq(1)
            objects = pd.read_csv(path/"objects.csv.gz", usecols=key+[
                "object_id","dist","seen","match_method","ls_rpos_x","ls_rpos_y"])
            data = objects.merge(agent.loc[valid,key],on=key,validate="many_to_one")
            data = data[data.object_id.ne(0) & data.seen.eq(1) & data.match_method.eq("unum")]
            data = data.dropna(subset=["dist","ls_rpos_x","ls_rpos_y"])
            estimate = np.hypot(data.ls_rpos_x.to_numpy(),data.ls_rpos_y.to_numpy())
            lookup = data.dist.to_numpy() - .001
            def matches(value):
                index = np.searchsorted(entries[:,0],value,side="left")
                inside = index < len(entries)
                midpoint = entries[np.minimum(index,len(entries)-1),1]
                return inside & (np.abs(estimate-midpoint) <= tolerance)
            primary = matches(lookup)
            # Range is logged to 4 decimals. Allow adjacent bins only if the
            # distance could round across the lookup threshold.
            rounded = primary | matches(lookup-.00005) | matches(lookup+.00005)
            records.append(dict(game=game.name,observer=observer,n=len(data),
                                matches=int(primary.sum()),matches_allow_csv_rounding=int(rounded.sum())))
    totals = {field:sum(record[field] for record in records)
              for field in ("n","matches","matches_allow_csv_rounding")}
    return dict(table_entries=len(entries), tolerance_m=tolerance, records=records, **totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs",type=Path)
    parser.add_argument("object_table_cpp",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    result = replay(args.logs,args.object_table_cpp)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"Lookup matched {result['matches_allow_csv_rounding']:,} of {result['n']:,} rows")
