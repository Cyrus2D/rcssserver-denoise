#!/usr/bin/env python3
"""Compare kick execution when a new same-cycle foul charge is recorded."""
import argparse,json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd

def game(path):
    frames={};fouls={};out={s:Counter()for s in "lr"};cases=[]
    for p in sorted(path.glob("*/meta.json")):
        name=p.parent.name
        a=pd.read_csv(p.with_name("agent_cycle.csv.gz"),usecols=["cycle","stopped","game_mode","decision_made","body_cmd","tackle_foul","sb_kick_count","sb_charged_expires"],dtype={"tackle_foul":str})
        a=a[a.stopped.eq(0)&a.decision_made.eq(1)].set_index("cycle")
        frames[name]=a
        fouls[name]=a.index[a.game_mode.eq("play_on")&a.body_cmd.eq("tackle")&a.tackle_foul.fillna("").str.lower().isin(["on","true","1","1.0"])].to_numpy()
    for name,a in frames.items():
        kick=a[a.game_mode.eq("play_on")&a.body_cmd.eq("kick")];nxt=a.reindex(kick.index.to_numpy()+1)
        eligible=nxt.game_mode.eq("play_on").to_numpy()
        newcharge=(kick.sb_charged_expires.eq(0).to_numpy()&nxt.sb_charged_expires.gt(0).to_numpy()&eligible)
        success=nxt.sb_kick_count.to_numpy()>kick.sb_kick_count.to_numpy()
        opponent=np.isin(kick.index,np.concatenate([v for n,v in fouls.items() if n[0]!=name[0]]))
        c=out[name[0]]
        c["paired_kicks"]+=int(eligible.sum())
        c["new_charge"]+=int(newcharge.sum())
        c["new_charge_kick_acked"]+=int((newcharge&success).sum())
        c["new_charge_kick_blocked"]+=int((newcharge&~success).sum())
        c["new_charge_same_cycle_opponent_foul"]+=int((newcharge&opponent).sum())
        for j in np.flatnonzero(newcharge):
            cycle=int(kick.index[j]);tacklers=[n for n,v in fouls.items()if n[0]!=name[0]and cycle in v]
            cases.append(dict(game=path.name,cycle=cycle,victim=name,acknowledged=bool(success[j]),tacklers=tacklers))
    return path.name,out,cases
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("logs",type=Path);p.add_argument("selection",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args();selected=json.loads(args.selection.read_text());total={s:Counter()for s in "lr"};per_game={};cases=[]
    with ProcessPoolExecutor(12)as pool:
        for name,c,cc in pool.map(game,[args.logs/g for g,_ in selected]):
            per_game[name]=c;cases.extend(cc)
            for side in "lr":total[side].update(c[side])
    args.output.write_text(json.dumps(dict(counts=total,per_game=per_game,cases=cases),indent=2)+"\n");print(json.dumps(total))
