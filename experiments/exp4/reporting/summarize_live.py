#!/usr/bin/env python3
"""Score schema-3 live candidates and actual gated estimates separately."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd

def xy(frame,prefix):
    return frame[[prefix+"_x",prefix+"_y"]].to_numpy()
def angle(a,b):
    return np.abs((a-b+180)%360-180)
def observer(path):
    meta=json.loads((path/"meta.json").read_text())
    a=pd.read_csv(path/"agent_cycle.csv.gz",usecols=["cycle","stopped","game_mode","decision_made","see_used","est_pos_x","est_pos_y","est_face"])
    a=a[(a.stopped==0)&a.decision_made.eq(1)&a.game_mode.eq("play_on")]
    fields="""cycle stopped object_id truth_valid tru_on_field dist body_rel true_dist true_dir
ls_pos_x ls_pos_y denoise_pos_x denoise_pos_y denoise_applied
tru_pos_x tru_pos_y tru_vel_x tru_vel_y tru_body trk_pos_x trk_pos_y trk_vel_x trk_vel_y trk_body
exp4_pos_x exp4_pos_y exp4_vel_x exp4_vel_y exp4_body exp4_next_pos_x exp4_next_pos_y
exp4_next_vel_x exp4_next_vel_y exp4_stock_pos_x exp4_stock_pos_y exp4_stock_vel_x exp4_stock_vel_y
exp4_stock_body exp4_applied exp4_pos_applied exp4_vel_applied exp4_body_applied exp4_forecast_applied""".split()
    o=pd.read_csv(path/"objects.csv.gz",usecols=fields).merge(a,on=["cycle","stopped"])
    counters=Counter(decisions=len(a),see_updates=int(a.see_used.sum()))
    stats={}
    def add(key,old,candidate,applied,target,body=False):
        f=angle if body else lambda v,t:np.linalg.norm(v-t,axis=1)
        old,can,new=(f(v,target) for v in (old,candidate,applied))
        valid=np.isfinite(old)&np.isfinite(can)&np.isfinite(new)
        old,can,new=old[valid],can[valid],new[valid]
        stats[key]=np.array([len(old),old.sum(),can.sum(),new.sum(),(new>old+.0002).sum(),(can>old+.0002).sum()])
    pos=o[o.object_id.ne(0)&o.denoise_applied.eq(1)&o.truth_valid.eq(1)&o.tru_on_field.eq(1)]
    counters["step26_position_applied"]=len(pos)
    if len(pos):
        theta=np.deg2rad(pos.est_face+pos.true_dir)
        truth_relative=np.column_stack((pos.est_pos_x+pos.true_dist*np.cos(theta),
                                        pos.est_pos_y+pos.true_dist*np.sin(theta)))
        add("player_position_relative/all",xy(pos,"ls_pos"),xy(pos,"denoise_pos"),xy(pos,"trk_pos"),truth_relative)
        add("player_position_global/all",xy(pos,"ls_pos"),xy(pos,"denoise_pos"),xy(pos,"trk_pos"),xy(pos,"tru_pos"))
    ball=o[o.object_id.eq(0)].set_index("cycle")
    data=o[o.exp4_vel_x.notna()&o.truth_valid.eq(1)&o.tru_on_field.eq(1)]
    options=meta["config"]
    for isball,frame in [(False,data[data.object_id.ne(0)]),(True,data[data.object_id.eq(0)])]:
        kind="ball" if isball else "player"
        if not len(frame):continue
        ov=np.nan_to_num(xy(frame,"exp4_stock_vel"));cv=xy(frame,"exp4_vel")
        av=np.nan_to_num(xy(frame,"trk_vel"));tv=xy(frame,"tru_vel")
        groups={"all":np.ones(len(frame),bool)}
        if isball:
            groups.update(under_5m=frame.dist.lt(5).to_numpy(),at_least_5m=frame.dist.ge(5).to_numpy())
        else:
            groups.update(under_10m=frame.dist.lt(10).to_numpy(),at_least_10m=frame.dist.ge(10).to_numpy())
        for name,mask in groups.items():add(kind+"_velocity/"+name,ov[mask],cv[mask],av[mask],tv[mask])
        expected=np.where(frame.exp4_vel_applied.eq(1).to_numpy()[:,None],cv,ov)
        counters[kind+"_velocity_write_mismatch"]+=int((np.linalg.norm(expected-av,axis=1)>.0002).sum())
        if isball:
            op=xy(frame,"exp4_stock_pos");cp=xy(frame,"exp4_pos");ap=xy(frame,"trk_pos");tp=xy(frame,"tru_pos")
            for name,mask in groups.items():add("ball_position/"+name,op[mask],cp[mask],ap[mask],tp[mask])
            expect=np.where(frame.exp4_pos_applied.eq(1).to_numpy()[:,None],cp,op)
            counters["ball_position_write_mismatch"]+=int((np.linalg.norm(expect-ap,axis=1)>.0002).sum())
            counters["ball_position_guard_violations"]+=int((frame.dist.lt(options["exp4_ball_position_min_dist"])&frame.exp4_pos_applied.eq(1)).sum())
            counters["ball_velocity_guard_violations"]+=int((frame.dist.lt(options["exp4_ball_velocity_min_dist"])&frame.exp4_vel_applied.eq(1)).sum())
            nxt=ball.reindex(frame.cycle.to_numpy()+1)
            valid=nxt.truth_valid.eq(1).to_numpy()&nxt.tru_on_field.eq(1).to_numpy()
            for label,old,candidate,actual,target in [
                ("next_position",op+ov,xy(frame,"exp4_next_pos"),ap+av,xy(nxt,"tru_pos")),
                ("next_velocity",float(meta["server_param"]["ball_decay"])*ov,xy(frame,"exp4_next_vel"),float(meta["server_param"]["ball_decay"])*av,xy(nxt,"tru_vel"))]:
                add("ball_"+label+"/all",old[valid],candidate[valid],actual[valid],target[valid])
        else:
            old=frame.exp4_stock_body.to_numpy()
            old=np.where(np.isfinite(old),old,np.rad2deg(np.arctan2(ov[:,1],ov[:,0])))
            candidate=np.where(frame.body_rel.notna(),old,frame.exp4_body)
            actual=np.where(frame.exp4_body_applied.eq(1),frame.exp4_body,old)
            add("player_body/all",old,candidate,actual,frame.tru_body.to_numpy(),True)
            counters["player_velocity_guard_violations"]+=int((frame.dist.lt(options["exp4_player_velocity_min_dist"])&frame.exp4_vel_applied.eq(1)).sum())
            counters["reported_body_overwritten"]+=int((frame.body_rel.notna()&frame.exp4_body_applied.eq(1)).sum())
        counters[kind+"_candidates"]+=len(frame)
        if isball:counters["ball_position_applied"]+=int(frame.exp4_pos_applied.eq(1).sum())
        counters[kind+"_velocity_applied"]+=int(frame.exp4_vel_applied.eq(1).sum())
        counters[kind+"_body_applied"]+=int(frame.exp4_body_applied.eq(1).sum())
    counters["forecast_applied"]+=int(data.exp4_forecast_applied.eq(1).sum())
    return path.parent.name,meta["side"],counters,stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--workers",type=int,default=12)
    args=p.parse_args()
    paths=sorted(args.logs.glob("*/*/meta.json"))
    if not paths:paths=sorted(args.logs.glob("*/meta.json"))
    sources=[]
    for path in paths:
        meta=json.loads(path.read_text())
        if not meta["final"] or meta["status"]!="ok":raise ValueError("Unfinalized log: "+str(path))
        if meta["schema_version"]<3:raise ValueError("Use audit_live.py for old schemas")
        sources.append(path.parent)
    if not sources:raise ValueError("No completed observer logs")
    sums={};counts={s:Counter() for s in "lr"};games=set()
    with ProcessPoolExecutor(args.workers) as pool:
        for game,side,c,stats in pool.map(observer,sources):
            games.add(game);counts[side].update(c)
            for k,v in stats.items():
                key=side+"/"+k;sums[key]=sums.get(key,np.zeros(6))+v
    metrics={k:dict(n=int(v[0]),mean_stock=v[1]/v[0],mean_candidate=v[2]/v[0],mean_applied=v[3]/v[0],
                    worse_applied_share=v[4]/v[0],worse_candidate_share=v[5]/v[0]) for k,v in sums.items() if v[0]}
    result=dict(games=len(games),observers=len(sources),counts=counts,metrics=metrics,
                future_note="Candidate is learned forecast; applied is ordinary coasting from the actually gated current state.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
