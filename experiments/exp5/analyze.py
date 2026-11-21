#!/usr/bin/env python3
"""Baseline visual-gap audit. Inputs are observations only; fullstate scores errors."""
import argparse,json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

WIDTH=.01
BINS=10001
METHODS=["wm","hold","constant_velocity","decay"]
AGE_NAMES=["0","1","2","3-5","6-10","11-20","21+"]
def xy(f,p):return f[[p+"_x",p+"_y"]].to_numpy(float)
def add(stats,key,error):
    error=error[np.isfinite(error)]
    if not len(error):return
    hist=np.bincount(np.minimum((error/WIDTH).astype(int),BINS-1),minlength=BINS)
    if key not in stats:stats[key]=[0,0.,np.zeros(BINS,np.int64),0.]
    value=stats[key];value[0]+=len(error);value[1]+=float(error.sum());value[2]+=hist;value[3]=max(value[3],float(error.max()))

def observer(job):
    path,allow_quantized=job
    meta=json.loads((path/"meta.json").read_text());cfg=meta["config"];sp=meta["server_param"]
    assert not cfg["use_fullstate"],"Fullstate must be reference-only"
    assert sp["full_view"]=="false" and sp["say_msg_size"]=="0","Use finite views with hearing disabled"
    if not allow_quantized:
        assert sp.get("see_no_quantization")=="true" and sp["see_no_fade_all"]=="true","Wrong sensor configuration"
    agent=pd.read_csv(path/"agent_cycle.csv.gz",usecols=["cycle","stopped","decision_made","game_mode","see_used","view_width","sb_view_width",
       "n_est_pos_d","n_est_vel_d","n_est_face"])
    agent=agent[agent.decision_made.eq(1)].copy()
    # Find discontinuities before dropping stopped-time rows: a setplay can
    # occur entirely in stopped time between consecutive PlayOn cycle numbers.
    continuous=(agent.stopped.eq(0)&agent.stopped.shift().eq(0)&agent.cycle.diff().eq(1)
                &agent.game_mode.eq("play_on")&agent.game_mode.shift().eq("play_on"))
    agent["block"]=(~continuous).cumsum()
    agent=agent[agent.stopped.eq(0)&agent.game_mode.eq("play_on")].copy()
    # Use the received view width, not a potentially queued next-view command.
    agent["view_width"]=agent.sb_view_width.fillna(agent.view_width)
    columns="""cycle stopped object_id truth_valid tru_on_field seen unum_visible match_ambiguous
true_dist trk_pos_x trk_pos_y trk_vel_x trk_vel_y ls_pos_x ls_pos_y ls_vel_x ls_vel_y
tru_pos_x tru_pos_y n_dist n_dir n_dist_chg n_dir_chg n_body_rel n_ls_pos_d n_ls_rpos_d""".split()
    objects=pd.read_csv(path/"objects.csv.gz",usecols=columns,low_memory=False)
    objects=objects[objects.stopped.eq(0)&objects.object_id.ne(0)&objects.truth_valid.eq(1)&objects.tru_on_field.eq(1)]
    objects=objects.merge(agent,on=["cycle","stopped"],validate="many_to_one")
    counts=Counter(observers=1,decisions=len(agent),see_updates=int(agent.see_used.sum()),player_rows=len(objects))
    counts.update({"view_"+str(k):int(v)for k,v in agent.view_width.value_counts().items()})
    stats={}
    for name in ["n_est_pos_d","n_est_vel_d","n_est_face"]:
        add(stats,"sensor/"+name,agent[name].abs().to_numpy())
    known=objects.seen.eq(1)&objects.unum_visible.eq(1)&objects.match_ambiguous.fillna(0).eq(0)
    for name in ["n_dist","n_dir","n_dist_chg","n_dir_chg","n_body_rel","n_ls_pos_d","n_ls_rpos_d"]:
        add(stats,"sensor/"+name,objects.loc[known,name].abs().to_numpy())
    # Each target is processed within an uninterrupted PlayOn decision block.
    # No truth position, velocity, body or fullstate-only identity enters a prediction.
    for _,f in objects.groupby(["block","object_id"],sort=False):
        f=f.sort_values("cycle")
        pos=xy(f,"ls_pos");wm=xy(f,"trk_pos");truth=xy(f,"tru_pos")
        sight=(f.seen.eq(1)&f.unum_visible.eq(1)&f.match_ambiguous.fillna(0).eq(0)).to_numpy()&np.isfinite(pos).all(axis=1)
        index=np.arange(len(f));last=np.maximum.accumulate(np.where(sight,index,-1))
        history=last>=0
        if not history.any():continue
        f=f.iloc[np.flatnonzero(history)]
        wm=wm[history];truth=truth[history];sight=sight[history];last=last[history]
        fullpos=pos;lastpos=fullpos[last]
        # Last available localized velocity; otherwise the valid WM value at that sighting.
        # The group remains in original sorted order for velocity indexing.
        # Reconstruct source velocities from the original group without accessing truth.
        # f now contains the retained tail; last indices are offset by the first sighting.
        first=int(np.flatnonzero(history)[0]);relative_last=last-first
        localized=xy(f,"ls_vel");tracked=xy(f,"trk_vel")
        velocity=np.where(np.isfinite(localized).all(axis=1)[:,None],localized,tracked)
        velocity=np.nan_to_num(velocity)[relative_last]
        ages=f.cycle.to_numpy()-f.cycle.to_numpy()[relative_last]
        hold=lastpos
        cv=lastpos+velocity*ages[:,None]
        decay=float(sp["player_decay"])
        distance=(1-np.power(decay,ages))/(1-decay) if decay!=1 else ages
        coast=lastpos+velocity*distance[:,None]
        estimates=[wm,hold,cv,coast]
        valid=np.logical_and.reduce([np.isfinite(v).all(axis=1)for v in estimates])&np.isfinite(truth).all(axis=1)
        counts["history_rows"]+=len(f);counts["wm_missing_with_history"]+=int((~np.isfinite(wm).all(axis=1)).sum())
        anonymous=f.seen.eq(1).to_numpy()&~sight
        counts["anonymous_sighting_rows_omitted"]+=int((valid&anonymous).sum())
        valid &= ~anonymous
        counts["paired_rows"]+=int(valid.sum());counts["unseen_paired_rows"]+=int((valid&(ages>0)).sum())
        if not valid.any():continue
        agebin=np.select([ages==0,ages==1,ages==2,ages<=5,ages<=10,ages<=20],AGE_NAMES[:-1],default="21+")
        groups={"all":valid,"unseen":valid&(ages>0)}
        groups.update({"age/"+k:valid&(agebin==k)for k in AGE_NAMES})
        for width in ["narrow","normal","wide"]:
            groups["view/"+width]=valid&f.view_width.eq(width).to_numpy()&(ages>0)
        for label,low,high in [("0-10",0,10),("10-20",10,20),("20-40",20,40),("40+",40,np.inf)]:
            groups["distance/"+label]=valid&f.true_dist.ge(low).to_numpy()&f.true_dist.lt(high).to_numpy()&(ages>0)
        for method,estimate in zip(METHODS,estimates):
            error=np.linalg.norm(estimate-truth,axis=1)
            for label,mask in groups.items():add(stats,method+"/"+label,error[mask])
    return path.parent.name,counts,stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--workers",type=int,default=2);p.add_argument("--limit",type=int,default=0)
    p.add_argument("--allow-quantized",action="store_true",help="negative-control audit only")
    a=p.parse_args()
    games=[a.logs]if list(a.logs.glob("*/meta.json"))else sorted(x for x in a.logs.iterdir()if x.is_dir())
    selected=[];skipped=[]
    for g in games:
        files=sorted(g.glob("*/meta.json"))
        if len(files)!=22:skipped.append(g.name);continue
        meta=[json.loads(m.read_text())for m in files]
        if not all(m.get("final")and m.get("status")=="ok"for m in meta):skipped.append(g.name);continue
        selected.append((g,files))
    if a.limit:selected=selected[:a.limit]
    if not selected:raise ValueError("No fully finalized 22-agent games; wait for clean game shutdown.")
    counts=Counter();total={}
    jobs=[(m.parent,a.allow_quantized)for _,files in selected for m in files]
    with ProcessPoolExecutor(max_workers=a.workers)as pool:
        for game,c,stats in pool.map(observer,jobs):
            counts.update(c)
            for k,v in stats.items():
                if k not in total:total[k]=[0,0.,np.zeros(BINS,np.int64),0.]
                for i in range(3):total[k][i]+=v[i]
                total[k][3]=max(total[k][3],v[3])
    metrics={k:dict(n=int(v[0]),mean=float(v[1]/v[0]),max=float(v[3]),p90_histogram=float((np.searchsorted(v[2].cumsum(),.9*v[0])+.5)*WIDTH))
             for k,v in total.items()if v[0]}
    if not a.allow_quantized:
        for key in ["n_dist","n_dir","n_dist_chg","n_dir_chg","n_body_rel"]:
            value=metrics.get("sensor/"+key)
            if value and value["max"]>.000005:
                raise ValueError("Quantization-free observation check failed: "+key+" max="+str(value["max"]))
    result=dict(games=[g.name for g,_ in selected],skipped=skipped,counts=counts,metrics=metrics,
        method="Equal weight per paired target-cycle; all four methods use the same rows. Baseline velocity falls back to zero when unavailable at the last identified sighting. Decay uses the server default, not fullstate player type.",
        gap_age="Cycles since last unambiguous identified visual localization inside the same uninterrupted PlayOn block. Anonymous close sightings do not refresh labeled history and are omitted from scored gap rows.",
        p90_note="Histogram approximation with 1 cm position bins; sensor means are exact aggregates. No future sightings are used.")
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Scored",len(selected),"games,",len(jobs),"observers,",counts["paired_rows"],"paired player rows")
    render(a.output,result)

def render(out,data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m=data["metrics"];fig,ax=plt.subplots(figsize=(9,4.5))
    labels=[x for x in AGE_NAMES if "wm/age/"+x in m]
    for method,label in zip(METHODS,["librcsc WM","Hold last sighting","Constant velocity","Decayed velocity"]):
        ax.plot(labels,[100*m[method+"/age/"+x]["mean"]for x in labels],marker="o",label=label)
    ax.set(xlabel="Cycles since identified visual sighting",ylabel="Mean position error (cm)",
           title="Unseen-player baseline errors by observation age")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/"gap-errors.svg");plt.close(fig)
    rows=[]
    for age in labels:
        rows.append("<tr><td>"+age+"</td>"+"".join(f"<td>{100*m[k+'/age/'+age]['mean']:.2f}</td>"for k in METHODS)+"</tr>")
    sensor="".join(f"<li>{key.removeprefix('sensor/')}: mean absolute error {v['mean']:.8f} ({v['n']:,} rows)</li>"for key,v in m.items()if key.startswith("sensor/"))
    html="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Exp5 baseline: finite visual gaps</title><style>body{font:16px/1.5 system-ui;color:#163047;max-width:1050px;margin:35px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}svg{width:100%;height:auto}</style>
<h1>Exp5 · Finite-view baseline</h1>"""
    c=data["counts"]
    html+=f"<p>{len(data['games'])} finalized games, {c['observers']:,} observers; {c['paired_rows']:,} paired rows, {c['unseen_paired_rows']:,} unseen rows. Every paired sample has equal weight.</p>"
    html+=(out/"gap-errors.svg").read_text()
    html+="<table><tr><th>Age</th><th>WM</th><th>Hold</th><th>Constant v</th><th>Decayed v</th></tr>"+"".join(rows)+"</table><p>Mean position distance error in cm. These are baseline predictions, not learned corrections.</p>"
    html+=f"<p>{data['gap_age']}</p><p>Tracked positions missing despite available history: {c['wm_missing_with_history']:,} of {c['history_rows']:,}. Missing WM rows are excluded from paired means and retained in coverage counts.</p>"
    html+="<h2>Observation checks</h2><ul>"+sensor+"</ul><p>View and true-distance subgroups, exact sample counts and the frozen game list are in metrics.json. Sensor p90 histogram resolution is unsuitable for tiny exact-mode errors; use the reported sensor means.</p><p>"+data["method"]+"</p></html>"
    (out/"report.html").write_text(html)

if __name__=="__main__":main()
