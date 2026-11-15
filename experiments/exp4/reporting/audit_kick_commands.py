#!/usr/bin/env python3
"""Check successful-kick acknowledgements and physical range on paired PlayOn commands."""
import argparse,json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd

AGENT="""cycle stopped game_mode decision_made body_cmd sb_kick_count est_kickable see_used sb_charged_expires tackle_foul msec_sb_to_decision
tru_pos_x tru_pos_y tru_player_type action_label kick_power kick_dir next_ball_pos_x next_ball_pos_y next_ball_vel_x next_ball_vel_y""".split()
def run_game(path):
    meta_paths=sorted(path.glob("*/meta.json"));meta0=json.loads(meta_paths[0].read_text());s=meta0["server_param"]
    canonical=pd.read_csv(meta_paths[0].with_name("objects.csv.gz"),usecols=["cycle","stopped","object_id","truth_valid","tru_pos_x","tru_pos_y","tru_vel_x","tru_vel_y"])
    canonical=canonical[canonical.object_id.eq(0)&canonical.stopped.eq(0)&canonical.truth_valid.eq(1)].set_index("cycle")
    counts={side:Counter()for side in "lr"};cases=[];labels={side:Counter()for side in "lr"};by_unum={side:Counter()for side in "lr"}
    cache={};fouls={}
    for mp in meta_paths:
        m=json.loads(mp.read_text());identity=m["unum"]+(11 if m["side"]=="r" else 0)
        a=pd.read_csv(mp.with_name("agent_cycle.csv.gz"),usecols=AGENT,dtype={"tackle_foul":str})
        a=a[a.stopped.eq(0)&a.decision_made.eq(1)].set_index("cycle").sort_index()
        cache[identity]=(m,a)
        fouls[identity]=a.index[a.game_mode.eq("play_on")&a.body_cmd.eq("tackle")&a.tackle_foul.fillna("").str.lower().isin(["on","true","1","1.0"])].to_numpy()
    for identity,(m,a) in cache.items():
        side=m["side"];unum=m["unum"];types={p["id"]:p for p in m["player_types"]}
        radius={k:t["player_size"]+t["kickable_margin"]+float(s["ball_size"])for k,t in types.items()}
        kick=a[a.game_mode.eq("play_on")&a.body_cmd.eq("kick")]
        nxt=a.reindex(kick.index.to_numpy()+1)
        paired=nxt.game_mode.eq("play_on").to_numpy();kick=kick.iloc[np.flatnonzero(paired)];nxt=nxt.iloc[np.flatnonzero(paired)]
        success=nxt.sb_kick_count.to_numpy()>kick.sb_kick_count.to_numpy()
        b=canonical.reindex(kick.index)
        distance=np.hypot(b.tru_pos_x-kick.tru_pos_x,b.tru_pos_y-kick.tru_pos_y).to_numpy()
        area=kick.tru_player_type.map(radius).to_numpy();margin=distance-area
        good=np.isfinite(margin);outside=margin>.0002
        c=counts[side];c["commands"]+=len(kick);c["acknowledged"]+=int(success.sum());c["unacknowledged"]+=int((~success).sum())
        c["intentional_foul_commands"]+=len(fouls[identity])
        coincident=np.isin(kick.index,np.concatenate([v for i,v in fouls.items() if (i<=11)!=(identity<=11)]))
        c["coincident_foul_kick_inside"]+=int((coincident&good&~outside).sum())
        c["coincident_foul_kick_inside_unack"]+=int((coincident&good&~outside&~success).sum())
        c["coincident_foul_kick_inside_ack"]+=int((coincident&good&~outside&success).sum())
        charged=nxt.sb_charged_expires.gt(0).to_numpy()
        c["unack_next_charged"]+=int((~success&charged).sum())
        c["unack_not_charged"]+=int((~success&~charged).sum())
        c["true_outside_commands"]+=int((good&outside).sum())
        c["unack_outside"]+=int((~success&good&outside).sum());c["unack_inside"]+=int((~success&good&~outside).sum())
        c["unack_missing_truth"]+=int((~success&~good).sum());c["ack_outside"]+=int((success&good&outside).sum())
        c["unack_estimated_kickable"]+=int((~success&kick.est_kickable.eq(1).to_numpy()).sum())
        for j in np.flatnonzero(~success):
            row=kick.iloc[j];label=str(row.action_label);labels[side][label]+=1;by_unum[side][unum]+=1
            cases.append(dict(game=path.name,side=side,unum=unum,cycle=int(kick.index[j]),true_distance=float(distance[j]),kickable_area=float(area[j]),
                              outside_by=float(margin[j]),estimated_kickable=int(row.est_kickable),see_used=int(row.see_used),
                              kick_power=float(row.kick_power),label=label,
                              charged_before=int(row.sb_charged_expires),charged_next=int(nxt.sb_charged_expires.iloc[j]),
                              decision_ms=float(row.msec_sb_to_decision),opponent_foul_same_cycle=bool(coincident[j]),
                              opposing_foul_unums=[i if i<=11 else i-11 for i,v in fouls.items() if (i<=11)!=(identity<=11) and int(kick.index[j]) in v]))
    return path.name,counts,cases,labels,by_unum

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs",type=Path);p.add_argument("output",type=Path);p.add_argument("--selection",type=Path);p.add_argument("--workers",type=int,default=12)
    args=p.parse_args()
    if args.selection:games=[args.logs/g for g,_ in json.loads(args.selection.read_text())]
    else:
        games=[]
        for g in sorted(args.logs.iterdir()):
            metas=list(g.glob("*/meta.json"))
            if len(metas)==22 and all(json.loads(m.read_text()).get("final")for m in metas):games.append(g)
    total={s:Counter()for s in "lr"};labels={s:Counter()for s in "lr"};by_unum={s:Counter()for s in "lr"};cases=[];pergame={}
    with ProcessPoolExecutor(args.workers) as pool:
        for name,c,cc,ll,uu in pool.map(run_game,games):
            pergame[name]=c;cases.extend(cc)
            for side in "lr":total[side].update(c[side]);labels[side].update(ll[side]);by_unum[side].update(uu[side])
    result=dict(games=len(games),counts=total,labels=labels,by_unum=by_unum,cases=cases,per_game=pergame)
    args.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(total),flush=True)
