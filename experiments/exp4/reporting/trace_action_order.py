#!/usr/bin/env python3
"""Trace illustrative simultaneous kick/foul sends on the shared host clock."""
import argparse,json
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

def read_sends(job):
    path,cycles=job
    f=pd.read_csv(path/"raw_stream.csv.gz",usecols=["cycle","stopped","dir","local_ms","msg"])
    f=f[f.stopped.eq(0)&f.dir.eq("send")&f.cycle.isin(cycles)]
    return str(path),{int(c):g[["local_ms","msg"]].to_dict("records") for c,g in f.groupby("cycle")}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("logs",type=Path);p.add_argument("selection",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--per-category",type=int,default=2);p.add_argument("--events",type=Path,help="restrict to verified new-charge events");args=p.parse_args()
    selected=json.loads(args.selection.read_text());counts=Counter();samples=[]
    if args.events:
        used=set()
        for e in json.loads(args.events.read_text())["cases"]:
            if len(e["tacklers"])!=1:continue
            category=e["victim"][0]+("_ack" if e["acknowledged"] else "_unack")
            if counts[category]>=args.per_category or (category,e["game"])in used:continue
            samples.append(dict(game=e["game"],cycle=e["cycle"],victim=e["victim"],tackler=e["tacklers"][0],
                                acknowledged=e["acknowledged"],category=category))
            counts[category]+=1;used.add((category,e["game"]))
        selected=[]
    for game,_ in selected:
        frames={};fouls={}
        for mp in sorted((args.logs/game).glob("*/meta.json")):
            name=mp.parent.name
            a=pd.read_csv(mp.with_name("agent_cycle.csv.gz"),usecols=["cycle","stopped","game_mode","decision_made","body_cmd","tackle_foul","sb_kick_count"],dtype={"tackle_foul":str})
            a=a[a.stopped.eq(0)&a.decision_made.eq(1)].set_index("cycle");frames[name]=a
            fouls[name]=set(a.index[a.game_mode.eq("play_on")&a.body_cmd.eq("tackle")&a.tackle_foul.fillna("").str.lower().isin(["on","true","1","1.0"])])
        used=set()
        for victim,a in frames.items():
            side=victim[0]
            kicks=a[a.game_mode.eq("play_on")&a.body_cmd.eq("kick")]
            for cycle,row in kicks.iterrows():
                if cycle+1 not in a.index or a.loc[cycle+1,"game_mode"]!="play_on":continue
                ack=bool(a.loc[cycle+1,"sb_kick_count"]>row.sb_kick_count)
                category=side+("_ack" if ack else "_unack")
                if counts[category]>=args.per_category or category in used:continue
                tacklers=[name for name,cycles in fouls.items() if name[0]!=side and cycle in cycles]
                if not tacklers:continue
                samples.append(dict(game=game,cycle=int(cycle),victim=victim,tackler=tacklers[0],acknowledged=ack,category=category))
                counts[category]+=1;used.add(category)
        if len(counts)==4 and min(counts.values())>=args.per_category:break
    requests=defaultdict(set)
    for e in samples:
        for name in [e["victim"],e["tackler"]]:requests[args.logs/e["game"]/name].add(e["cycle"])
    with ProcessPoolExecutor(6)as pool:result=dict(pool.map(read_sends,requests.items()))
    for e in samples:
        victim=result[str(args.logs/e["game"]/e["victim"])][e["cycle"]]
        tackler=result[str(args.logs/e["game"]/e["tackler"])][e["cycle"]]
        v=next(x for x in victim if "(kick "in x["msg"]);t=next(x for x in tackler if "(tackle "in x["msg"])
        e.update(kick_send=v,foul_send=t,kick_minus_foul_ms=int(v["local_ms"]-t["local_ms"]))
    args.output.write_text(json.dumps(dict(samples=samples,categories=counts),indent=2)+"\n")
    for e in samples:print(e["category"],e["game"],e["cycle"],e["kick_minus_foul_ms"])
