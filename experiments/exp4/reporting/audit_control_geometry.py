#!/usr/bin/env python3
"""Complement the all-observer audit with observer-2 control/validity diagnostics."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd

def norm(v):return np.linalg.norm(v,axis=1)
def xy(f,p):return f[[p+"_x",p+"_y"]].to_numpy()
def angle(a,b):return abs((a-b+180)%360-180)
def job(item):
    path,observer=item;p=path/f"l_{observer}"
    meta=json.loads((p/"meta.json").read_text());s=meta["server_param"];types={t["id"]:t for t in meta["player_types"]}
    a=pd.read_csv(p/"agent_cycle.csv.gz",usecols=["cycle","stopped","game_mode","decision_made","est_kickable","est_body","tru_body","tru_pos_x","tru_pos_y","tru_player_type"])
    a=a[a.stopped.eq(0)&a.decision_made.eq(1)&a.game_mode.eq("play_on")].set_index("cycle")
    cols="""cycle stopped object_id player_type truth_valid tru_on_field true_dist
tru_pos_x tru_pos_y tru_vel_x tru_vel_y tru_body trk_pos_x trk_pos_y trk_vel_x trk_vel_y
trk_dist_from_self trk_angle_from_self exp4_stock_pos_x exp4_stock_pos_y exp4_stock_vel_x exp4_stock_vel_y
exp4_stock_body exp4_stock_body_count exp4_stock_vel_count exp4_vel_x exp4_body exp4_vel_applied exp4_body_applied denoise_applied""".split()
    o=pd.read_csv(p/"objects.csv.gz",usecols=cols)
    o=o[o.stopped.eq(0)&o.cycle.isin(a.index)&o.truth_valid.eq(1)&o.tru_on_field.eq(1)]
    b=o[o.object_id.eq(0)&o.exp4_vel_x.notna()].set_index("cycle")
    targets=o[o.object_id.ne(0)&o.exp4_vel_x.notna()&o.denoise_applied.eq(1)&o.cycle.isin(b.index)].copy()
    default_area=sum(float(s[k])for k in ["player_size","ball_size","kickable_margin"])
    radii={k:v["player_size"]+v["kickable_margin"]+float(s["ball_size"])for k,v in types.items()}
    area=targets.player_type.map(radii).fillna(default_area).to_numpy()
    def rel(prefix,bprefix):
        return xy(targets,prefix)-np.column_stack((targets.cycle.map(b[bprefix+"_x"]),targets.cycle.map(b[bprefix+"_y"])))
    old=norm(rel("exp4_stock_pos","exp4_stock_pos"));new=norm(rel("trk_pos","trk_pos"));truth=norm(rel("tru_pos","tru_pos"))
    truth_control=truth<=area
    groups={"all":np.ones(len(targets),bool),"observer_under10":targets.true_dist.lt(10).to_numpy(),
            "observer_10_to40":targets.true_dist.between(10,40,inclusive="left").to_numpy(),"observer_atleast40":targets.true_dist.ge(40).to_numpy(),
            "opponents_under10":targets.object_id.gt(11).to_numpy()&targets.true_dist.lt(10).to_numpy(),
            "true_in_range":truth_control,"near_boundary":abs(truth-area)<.3}
    counts={};metrics={}
    for name,m in groups.items():
        truthc=truth_control[m];before=old[m]<=area[m];after=new[m]<=area[m]
        counts[name]=[int(m.sum()),int(truthc.sum()),int((before&~truthc).sum()),int((after&~truthc).sum()),
                      int((~before&truthc).sum()),int((~after&truthc).sum())]
        metrics["distance_bias/"+name]=[int(m.sum()),float((old[m]-truth[m]).sum()),float((new[m]-truth[m]).sum()),
                                       float(abs(old[m]-truth[m]).sum()),float(abs(new[m]-truth[m]).sum())]
    promoted_body=targets.exp4_body_applied.eq(1)&targets.exp4_stock_body.isna()
    pb=angle(targets.exp4_body.to_numpy(),targets.tru_body.to_numpy())
    promoted_vel=targets.exp4_vel_applied.eq(1)&targets.exp4_stock_vel_x.isna()
    pv=norm(xy(targets,"trk_vel")-xy(targets,"tru_vel"))
    for key,mask,errors,threshold in [("promoted_body",promoted_body,pb,45),("promoted_velocity",promoted_vel,pv,.2)]:
        m=mask.to_numpy();metrics[key]=[int(m.sum()),float(errors[m].sum()),int((errors[m]>threshold).sum()),len(targets)]
    # Kick efficiency compares actual body-relative geometry with truth geometry.
    selfa=a.reindex(b.index);selftypes=selfa.tru_player_type.map(types)
    size=np.array([v["player_size"] for v in selftypes]);margin=np.array([v["kickable_margin"]for v in selftypes])
    diff=xy(b,"tru_pos")-selfa[["tru_pos_x","tru_pos_y"]].to_numpy();distance=norm(diff)
    bearing=np.rad2deg(np.arctan2(diff[:,1],diff[:,0]))-selfa.tru_body.to_numpy()
    observed=angle(b.trk_angle_from_self.to_numpy(),selfa.est_body.to_numpy())
    actual=angle(bearing,0)
    oldfactor=1-.25*observed/180-.25*(b.trk_dist_from_self.to_numpy()-size-float(s["ball_size"]))/margin
    truefactor=1-.25*actual/180-.25*(distance-size-float(s["ball_size"]))/margin
    m=selfa.est_kickable.eq(1).to_numpy()&np.isfinite(truefactor)&(truefactor>0)
    e=abs(oldfactor-truefactor)/truefactor
    metrics["kick_efficiency_relative_error"]=[int(m.sum()),float(e[m].sum()),int((e[m]>.05).sum()),int((e[m]>.1).sum())]
    return counts,metrics

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs",type=Path);p.add_argument("selection",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--observer",type=int,default=2);p.add_argument("--workers",type=int,default=8)
    args=p.parse_args();selected=json.loads(args.selection.read_text());counts={};metrics={}
    with ProcessPoolExecutor(args.workers) as pool:
        for c,m in pool.map(job,[(args.logs/g,args.observer)for g,_ in selected]):
            for k,v in c.items():counts[k]=np.array(counts.get(k,np.zeros(len(v))))+v
            for k,v in m.items():metrics[k]=np.array(metrics.get(k,np.zeros(len(v))))+v
    args.output.write_text(json.dumps(dict(observer=args.observer,games=len(selected),counts={k:v.tolist()for k,v in counts.items()},metrics={k:v.tolist()for k,v in metrics.items()}),indent=2)+"\n")
    print("Wrote",args.output)
