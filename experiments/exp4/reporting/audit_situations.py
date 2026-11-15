#!/usr/bin/env python3
"""Freeze completed F4 games and score action-critical ball/player regimes.

Each estimate gets one weight. Successful kick events come from changes in
the kicker's sense-body kick counter, not fullstate's attempt/display flag.
Reference geometry supplies evaluation groups only. Never changes live code.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

KEY=["cycle","stopped"]
AGENT="""cycle stopped game_mode decision_made see_used msec_sb_to_decision body_cmd
sb_kick_count sb_dash_count sb_turn_count est_pos_x est_pos_y est_vel_x est_vel_y
est_body est_face est_kickable tru_pos_x tru_pos_y tru_vel_x tru_vel_y tru_body tru_neck tru_player_type
next_ball_pos_x next_ball_pos_y next_ball_vel_x next_ball_vel_y kick_power kick_dir""".split()
OBJECT="""cycle stopped object_id player_type truth_valid tru_on_field true_dist true_dir dist
dist_chg body_rel seen trk_pos_count trk_pos_x trk_pos_y trk_vel_x trk_vel_y trk_body
trk_dist_from_self trk_angle_from_self ls_pos_x ls_pos_y ls_rpos_x ls_rpos_y denoise_pos_x denoise_pos_y denoise_applied
tru_pos_x tru_pos_y tru_vel_x tru_vel_y tru_body
exp4_stock_pos_x exp4_stock_pos_y exp4_stock_vel_x exp4_stock_vel_y exp4_stock_body
exp4_pos_x exp4_pos_y exp4_vel_x exp4_vel_y exp4_body exp4_body_applied
exp4_next_pos_x exp4_next_pos_y exp4_next_vel_x exp4_next_vel_y
exp4_pos_applied exp4_vel_applied exp4_forecast_applied exp4_applied exp4_controlled exp4_owner""".split()

def xy(f,p):return f[[p+"_x",p+"_y"]].to_numpy()
def norm(a):return np.linalg.norm(a,axis=-1)
def angular(a,b):return np.abs((a-b+180)%360-180)
def lag_age(events,cycles):
    index=np.arange(len(cycles))
    last=np.maximum.accumulate(np.where(events,cycles,-100000))
    start=np.maximum.accumulate(np.where(np.r_[True,np.diff(cycles)!=1],cycles,0))
    age=cycles-last
    return np.where(last>=start,age,100000)
def add(stats,hists,key,old,new,candidate=None,body=False):
    old,new=np.asarray(old),np.asarray(new)
    candidate=new if candidate is None else np.asarray(candidate)
    mask=np.isfinite(old)&np.isfinite(new)&np.isfinite(candidate)
    old,new,candidate=old[mask],new[mask],candidate[mask]
    if not len(old):return
    values=np.array([len(old),old.sum(),new.sum(),candidate.sum(),(new>old+.0002).sum(),
                     (new>old+.1).sum(),(old>.5).sum(),(new>.5).sum(),(new>old+.5).sum()],np.float64)
    stats[key]=stats.get(key,np.zeros(9))+values
    if key.rsplit("/",1)[-1] in ("all","observer_can_kick","after_other_kick_0","opponent_in_range"):
        width=.1 if body else .005
        h=np.stack([np.bincount(np.minimum((v/width).astype(int),2000),minlength=2001) for v in (old,new)])
        if key not in hists:hists[key]=[np.zeros((2,2001),np.int64),width]
        hists[key][0]+=h
def grouped(stats,hists,prefix,old,new,truth,groups,candidate=None,body=False):
    fun=angular if body else lambda v,t:norm(v-t)
    e0,e1=fun(old,truth),fun(new,truth)
    ec=e1 if candidate is None else fun(candidate,truth)
    for name,mask in groups.items():add(stats,hists,prefix+"/"+name,e0[mask],e1[mask],ec[mask],body)
def q(hist,p,width):return float((np.searchsorted(hist.cumsum(),p*hist.sum())+.5)*width)

def game_job(job):
    path,score=job
    frames={};metadata={};events={};commands={s:Counter() for s in "lr"}
    latency={s:np.zeros(3001,np.int64) for s in "lr"};checks=Counter();stats={};hists={}
    for p in sorted(path.glob("*/meta.json")):
        meta=json.loads(p.read_text());side=meta["side"];unum=meta["unum"];identity=unum+(11 if side=="r" else 0)
        metadata[identity]=meta
        a=pd.read_csv(p.with_name("agent_cycle.csv.gz"),usecols=AGENT)
        a=a[a.stopped.eq(0)&a.decision_made.eq(1)].set_index("cycle",verify_integrity=True).sort_index()
        frames[identity]=a
        continuous=a.index.to_series().diff().eq(1).to_numpy()
        events[identity]=a.index[(a.sb_kick_count.diff().gt(0).to_numpy())&continuous].to_numpy()
        play=a.game_mode.eq("play_on");aa=a[play]
        commands[side]["decisions"]+=len(aa);commands[side]["see_updates"]+=int(aa.see_used.sum())
        latency[side]+=np.bincount(np.minimum(aa.msec_sb_to_decision.fillna(0).clip(lower=0).astype(int),3000),minlength=3001)
        nxt=a.reindex(aa.index.to_numpy()+1)
        valid=nxt.game_mode.eq("play_on").to_numpy()
        for cmd,col in [("kick","sb_kick_count"),("dash","sb_dash_count"),("turn","sb_turn_count")]:
            mask=aa.body_cmd.eq(cmd).to_numpy()&valid
            delta=nxt[col].to_numpy()-aa[col].to_numpy()
            commands[side][cmd+"_commands_paired"]+=int(mask.sum())
            commands[side][cmd+"_acknowledged"]+=int((mask&(delta>0)).sum())
            commands[side][cmd+"_unacknowledged"]+=int((mask&(delta<=0)).sum())
            commands[side][cmd+"_multiple_counter_steps"]+=int((mask&(delta>1)).sum())
        checks["body_rows_seen"]+=len(a)
    # All other-object rows are read from treatment observers only.
    for identity in range(1,12):
        meta=metadata[identity];cfg=meta["config"];server=meta["server_param"]
        a=frames[identity];a=a[a.game_mode.eq("play_on")].copy()
        o=pd.read_csv(path/f"l_{identity}"/"objects.csv.gz",usecols=OBJECT)
        observer_frame=a.reset_index().rename(columns={k:"self_"+k for k in ("tru_pos_x","tru_pos_y","tru_vel_x","tru_vel_y","tru_body","tru_neck","tru_player_type")})
        o=o.merge(observer_frame,on=KEY,validate="many_to_one")
        ball=o[o.object_id.eq(0)].set_index("cycle")
        p=o[o.object_id.ne(0)&o.truth_valid.eq(1)&o.tru_on_field.eq(1)].copy()
        b=ball[ball.exp4_vel_x.notna()&ball.truth_valid.eq(1)&ball.tru_on_field.eq(1)].copy()
        if not len(b):continue
        types={t["id"]:t for t in meta["player_types"]}
        ball_size=float(server["ball_size"]);decay=float(server["ball_decay"])
        default_area=sum(float(server[k])for k in ("player_size","ball_size","kickable_margin"))
        areas={k:t["player_size"]+t["kickable_margin"]+ball_size for k,t in types.items()}
        p["area"]=p.player_type.map(areas).fillna(default_area)
        p["ball_dist"]=np.hypot(p.tru_pos_x-p.cycle.map(ball.tru_pos_x),p.tru_pos_y-p.cycle.map(ball.tru_pos_y))
        near=p.ball_dist.le(p.area)
        carrier=p[near]
        own_control=carrier[carrier.object_id.le(11)].groupby("cycle").size().reindex(b.index,fill_value=0).gt(0).to_numpy()
        opp_control=carrier[carrier.object_id.gt(11)].groupby("cycle").size().reindex(b.index,fill_value=0).gt(0).to_numpy()
        self_area=a.tru_player_type.map(areas).reindex(b.index).fillna(default_area).to_numpy()
        truth_ball=b[["tru_pos_x_x","tru_pos_y_x"]].to_numpy() if "tru_pos_x_x" in b else xy(b,"tru_pos")
        truth_vel=b[["tru_vel_x_x","tru_vel_y_x"]].to_numpy() if "tru_vel_x_x" in b else xy(b,"tru_vel")
        self_truth=a.reindex(b.index)[["tru_pos_x","tru_pos_y"]].to_numpy()
        self_dist=norm(truth_ball-self_truth)
        self_control=self_dist<=self_area
        allcycles=ball.index.to_numpy()
        kickother=np.isin(allcycles,np.concatenate([v for k,v in events.items() if k!=identity]))
        kickopp=np.isin(allcycles,np.concatenate([v for k,v in events.items() if k>11]))
        kickmate=np.isin(allcycles,np.concatenate([v for k,v in events.items() if k<=11 and k!=identity]))
        age_other=pd.Series(lag_age(kickother,allcycles),index=allcycles).reindex(b.index).to_numpy()
        age_opp=pd.Series(lag_age(kickopp,allcycles),index=allcycles).reindex(b.index).to_numpy()
        age_mate=pd.Series(lag_age(kickmate,allcycles),index=allcycles).reindex(b.index).to_numpy()
        future_other=np.isin(b.index.to_numpy()+1,np.concatenate([v for k,v in events.items() if k!=identity]))
        prev=ball.reindex(b.index.to_numpy()-1)
        prev_vel=prev[["tru_vel_x_x","tru_vel_y_x"]].to_numpy() if "tru_vel_x_x" in prev else xy(prev,"tru_vel")
        impulse=norm(truth_vel-decay*prev_vel)>.2
        groups={"all":np.ones(len(b),bool),"under_2m":self_dist<2,"2_to_5m":(self_dist>=2)&(self_dist<5),
                "5_to_10m":(self_dist>=5)&(self_dist<10),"10_to_20m":(self_dist>=10)&(self_dist<20),
                "20_to_40m":(self_dist>=20)&(self_dist<40),"at_least_40m":self_dist>=40,
                "observer_can_kick":b.est_kickable.eq(1).to_numpy(),"observer_true_in_range":self_control,
                "opponent_in_range":opp_control,"teammate_in_range":own_control,
                "nobody_in_range":~(self_control|opp_control|own_control),
                "after_other_kick_0":age_other==0,"after_other_kick_1":age_other==1,
                "after_other_kick_2_3":(age_other>=2)&(age_other<=3),
                "after_opponent_kick_0":age_opp==0,"after_teammate_kick_0":age_mate==0,
                "after_other_kick_0_under10":(age_other==0)&(self_dist<10),
                "after_other_kick_0_atleast10":(age_other==0)&(self_dist>=10),
                "next_other_kick":future_other,"no_next_other_kick":~future_other,
                "current_impulse":impulse,"no_current_impulse":~impulse,
                "raw_motion":b.dist_chg.notna().to_numpy(),"no_raw_motion":b.dist_chg.isna().to_numpy(),
                "slow_ball":norm(truth_vel)<.1,"fast_ball":norm(truth_vel)>=1}
        op,ap,cp=xy(b,"exp4_stock_pos"),xy(b,"trk_pos"),xy(b,"exp4_pos")
        ov,av,cv=np.nan_to_num(xy(b,"exp4_stock_vel")),np.nan_to_num(xy(b,"trk_vel")),xy(b,"exp4_vel")
        grouped(stats,hists,"ball_position",op,ap,truth_ball,groups,cp)
        grouped(stats,hists,"ball_velocity",ov,av,truth_vel,groups,cv)
        checks["ball_candidates"]+=len(b)
        checks["ball_position_write_mismatch"]+=int((norm(np.where(b.exp4_pos_applied.eq(1).to_numpy()[:,None],cp,op)-ap)>.0002).sum())
        checks["ball_velocity_write_mismatch"]+=int((norm(np.where(b.exp4_vel_applied.eq(1).to_numpy()[:,None],cv,ov)-av)>.0002).sum())
        checks["ball_position_guard_violations"]+=int((b.dist.lt(cfg["exp4_ball_position_min_dist"])&b.exp4_pos_applied.eq(1)).sum())
        checks["ball_velocity_guard_violations"]+=int((b.dist.lt(cfg["exp4_ball_velocity_min_dist"])&b.exp4_vel_applied.eq(1)).sum())
        checks["forecast_applied"]+=int(b.exp4_forecast_applied.eq(1).sum())
        checks["ball_applied"]+=int(b.exp4_vel_applied.eq(1).sum())
        nextball=ball.reindex(b.index.to_numpy()+1)
        valid=nextball.truth_valid.eq(1).to_numpy()&nextball.tru_on_field.eq(1).to_numpy()
        ntp=nextball[["tru_pos_x_x","tru_pos_y_x"]].to_numpy() if "tru_pos_x_x" in nextball else xy(nextball,"tru_pos")
        ntv=nextball[["tru_vel_x_x","tru_vel_y_x"]].to_numpy() if "tru_vel_x_x" in nextball else xy(nextball,"tru_vel")
        gnext={k:m&valid for k,m in groups.items()}
        grouped(stats,hists,"next_ball_position_coast",op+ov,ap+av,ntp,gnext,xy(b,"exp4_next_pos"))
        grouped(stats,hists,"next_ball_velocity_coast",decay*ov,decay*av,ntv,gnext,xy(b,"exp4_next_vel"))
        # The queued command prediction includes this observer's own intended kick.
        ownkick=b.body_cmd.eq("kick").to_numpy()&valid
        grouped(stats,hists,"queued_after_own_kick_position",xy(b,"next_ball_pos"),xy(b,"next_ball_pos"),ntp,{"all":ownkick})
        grouped(stats,hists,"queued_after_own_kick_velocity",xy(b,"next_ball_vel"),xy(b,"next_ball_vel"),ntv,{"all":ownkick})
        # Actual relative geometry is more relevant to kick efficiency than global position.
        selfbody=a.est_body.reindex(b.index).to_numpy();truebody=a.tru_body.reindex(b.index).to_numpy()
        true_angle=np.rad2deg(np.arctan2((truth_ball-self_truth)[:,1],(truth_ball-self_truth)[:,0]))
        range_error=np.abs(b.trk_dist_from_self.to_numpy()-self_dist)
        bearing_error=angular(b.trk_angle_from_self.to_numpy()-selfbody,true_angle-truebody)
        for key,err,body in [("ball_self_range",range_error,False),("ball_self_body_bearing",bearing_error,True)]:
            for name in ("observer_can_kick","observer_true_in_range","under_2m"):
                add(stats,hists,key+"/"+name,err[groups[name]],err[groups[name]],body=body)
        # Player arrays were renamed on merge below; every target uses its own radius.
        player_groups={"all":np.ones(len(p),bool),"under_10m":p.true_dist.lt(10).to_numpy(),
                       "10_to_40m":p.true_dist.between(10,40,inclusive="left").to_numpy(),"at_least_40m":p.true_dist.ge(40).to_numpy(),
                       "near_ball":p.ball_dist.lt(2).to_numpy(),"in_kicking_range":near.to_numpy(),
                       "raw_motion":p.dist_chg.notna().to_numpy(),"no_raw_motion":p.dist_chg.isna().to_numpy(),
                       "no_raw_body":p.body_rel.isna().to_numpy(),"raw_body":p.body_rel.notna().to_numpy()}
        kicked=np.isin(p.cycle.to_numpy()*32+p.object_id.to_numpy(),np.concatenate([v*32+k for k,v in events.items()]))
        player_groups["target_just_kicked"]=kicked
        haspos=p.denoise_applied.eq(1).to_numpy()
        truep=p[["tru_pos_x_x","tru_pos_y_x"]].to_numpy() if "tru_pos_x_x" in p else xy(p,"tru_pos")
        truev=p[["tru_vel_x_x","tru_vel_y_x"]].to_numpy() if "tru_vel_x_x" in p else xy(p,"tru_vel")
        truebodyp=p.tru_body_x.to_numpy() if "tru_body_x" in p else p.tru_body.to_numpy()
        pg={k:m&haspos for k,m in player_groups.items()}
        grouped(stats,hists,"player_position_global",xy(p,"ls_pos"),xy(p,"trk_pos"),truep,pg,xy(p,"denoise_pos"))
        theta=np.deg2rad(p.est_face+p.true_dir)
        reltruth=np.column_stack((p.est_pos_x+p.true_dist*np.cos(theta),p.est_pos_y+p.true_dist*np.sin(theta)))
        grouped(stats,hists,"player_position_relative",xy(p,"ls_pos"),xy(p,"trk_pos"),reltruth,pg,xy(p,"denoise_pos"))
        ready=p.exp4_vel_x.notna().to_numpy()
        pv0=np.nan_to_num(xy(p,"exp4_stock_vel"));pv1=np.nan_to_num(xy(p,"trk_vel"));pvc=xy(p,"exp4_vel")
        grouped(stats,hists,"player_velocity",pv0,pv1,truev,{k:m&ready for k,m in player_groups.items()},pvc)
        pb0=p.exp4_stock_body.to_numpy()
        pb0=np.where(np.isfinite(pb0),pb0,np.rad2deg(np.arctan2(pv0[:,1],pv0[:,0])))
        pb1=np.where(p.exp4_body_applied.eq(1),p.exp4_body,pb0)
        grouped(stats,hists,"player_body",pb0,pb1,truebodyp,{k:m&ready for k,m in player_groups.items()},body=True)
        checks["player_candidates"]+=int(ready.sum())
        checks["reported_body_overwritten"]+=int((p.body_rel.notna()&p.exp4_body_applied.eq(1)).sum())
        checks["player_velocity_guard_violations"]+=int((p.dist.lt(cfg["exp4_player_velocity_min_dist"])&p.exp4_vel_applied.eq(1)).sum())
        checks["player_velocity_write_mismatch"]+=int((ready&(norm(np.where(p.exp4_vel_applied.eq(1).to_numpy()[:,None],pvc,pv0)-pv1)>.0002)).sum())
        checks["player_position_corrections"]+=int(haspos.sum())
        checks["player_tracks_nonfresh"]+=int(p.trk_pos_count.gt(0).sum())
        checks["player_tracks"]+=len(p)
        # Does independent denoising improve the player-to-ball geometry?
        match=ready&haspos&p.cycle.isin(b.index).to_numpy()
        oldb=p.cycle.map(b.exp4_stock_pos_x).to_numpy();oldby=p.cycle.map(b.exp4_stock_pos_y).to_numpy()
        newb=p.cycle.map(b.trk_pos_x).to_numpy();newby=p.cycle.map(b.trk_pos_y).to_numpy()
        oldrel=xy(p,"exp4_stock_pos")-np.column_stack((oldb,oldby))
        newrel=xy(p,"trk_pos")-np.column_stack((newb,newby))
        tbx=p.cycle.map(ball["tru_pos_x_x"] if "tru_pos_x_x" in ball else ball.tru_pos_x)
        tby=p.cycle.map(ball["tru_pos_y_x"] if "tru_pos_y_x" in ball else ball.tru_pos_y)
        targetrel=truep-np.column_stack((tbx,tby))
        geo_groups={"all":match,"near_ball":match&p.ball_dist.lt(2).to_numpy(),
                    "near_boundary":match&np.abs(p.ball_dist-p.area).lt(.3).to_numpy(),
                    "opponents_near_ball":match&p.object_id.gt(11).to_numpy()&p.ball_dist.lt(2).to_numpy()}
        grouped(stats,hists,"player_ball_relative_geometry",oldrel,newrel,targetrel,geo_groups)
        for name,mask in geo_groups.items():
            actual=p.ball_dist.le(p.area).to_numpy()[mask]
            predicted0=(norm(oldrel)<=p.area.to_numpy())[mask];predicted1=(norm(newrel)<=p.area.to_numpy())[mask]
            checks["geometry_"+name+"_rows"]+=int(mask.sum())
            checks["geometry_"+name+"_true_in_range"]+=int(actual.sum())
            for label,pred in [("stock",predicted0),("applied",predicted1)]:
                checks["geometry_"+name+"_"+label+"_false_in_range"]+=int((pred&~actual).sum())
                checks["geometry_"+name+"_"+label+"_missed_in_range"]+=int((~pred&actual).sum())
        # Self localization is a shared error source but is not changed in F4.
        true_self=a[["tru_pos_x","tru_pos_y"]].to_numpy()
        self_error=norm(a[["est_pos_x","est_pos_y"]].to_numpy()-true_self)
        add(stats,hists,"self_position/all",self_error,self_error)
    return path.name,stats,hists,checks,commands,latency

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs",type=Path);parser.add_argument("selection",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--workers",type=int,default=16)
    args=parser.parse_args()
    selected=json.loads(args.selection.read_text());issues=[];modes=Counter()
    for game,score in selected:
        metas=sorted((args.logs/game).glob("*/meta.json"))
        if len(metas)!=22:raise ValueError("Incomplete observer set "+game)
        for p in metas:
            m=json.loads(p.read_text());c=m["config"];s=m["server_param"]
            if not(m["final"] and m["status"]=="ok" and m["schema_version"]==3):issues.append(str(p))
            modes[(m["side"],c["exp4_player_state"],c["exp4_ball_state"],c["player_position_routing"])]+=1
            if not(c["exp4_integration"]=="f4-isolated-observations" and not c["use_fullstate"] and c["debug_fullstate"] and not c["gaussian_see"] and s["full_view"]=="true" and s["see_no_fade_unum"]=="true" and s["see_no_fade_all"]=="false" and s["see_noise_free"]=="false" and s["say_msg_size"]=="0"):issues.append(str(p))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    print("Frozen games",len(selected),"metadata issues",len(issues),"modes",modes,flush=True)
    stats={};hists={};counts=Counter();commands={s:Counter()for s in "lr"};latency={s:np.zeros(3001,np.int64)for s in "lr"};per_game={}
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(game_job,(args.logs/game,score))for game,score in selected]
        for i,f in enumerate(as_completed(futures),1):
            name,ss,hh,cc,cmd,lat=f.result();counts.update(cc)
            for k,v in ss.items():stats[k]=stats.get(k,np.zeros(9))+v
            for k,(h,w) in hh.items():
                if k not in hists:hists[k]=[np.zeros((2,2001),np.int64),w]
                hists[k][0]+=h
            for side in "lr":commands[side].update(cmd[side]);latency[side]+=lat[side]
            per_game[name]={k:v.tolist()for k,v in ss.items() if k.endswith("/all")}
            print("Audited",i,"/",len(selected),name,flush=True)
    metrics={}
    for k,v in stats.items():
        n=v[0];item=dict(n=int(n),mean_stock=v[1]/n,mean_applied=v[2]/n,mean_candidate=v[3]/n,
                        worse_share=v[4]/n,worse_10cm_share=v[5]/n,stock_over_50cm_share=v[6]/n,applied_over_50cm_share=v[7]/n,worse_50cm_share=v[8]/n)
        if k in hists:
            h,w=hists[k];item.update(p90_stock=q(h[0],.9,w),p90_applied=q(h[1],.9,w))
        metrics[k]=item
    score=np.array([s for _,s in selected]);delta=score[:,0]-score[:,1]
    rng=np.random.default_rng(42);boot=rng.integers(len(delta),size=(20000,len(delta)))
    matches=dict(n=len(delta),goals=score.sum(axis=0).tolist(),wdl=[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())],
                 goal_margin_ci95=np.quantile(delta[boot].mean(axis=1),[.025,.975]).tolist(),
                 point_margin_ci95=np.quantile((3*np.sign(delta))[boot].mean(axis=1),[.025,.975]).tolist())
    result=dict(created=datetime.now(timezone.utc).isoformat(),games=[g for g,_ in selected],matches=matches,metadata_issues=issues,
                modes={str(k):v for k,v in modes.items()},checks=counts,commands=commands,
                latency_ms={s:{str(p):q(h,p,1)for p in [.5,.95,.99]}for s,h in latency.items()},
                metrics=metrics,per_game=per_game)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("Wrote",args.output,flush=True)
if __name__=="__main__":main()
