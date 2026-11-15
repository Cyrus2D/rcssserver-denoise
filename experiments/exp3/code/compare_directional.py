#!/usr/bin/env python3
"""Compare directional and librcsc motion priors on fixed normal-view holdouts."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import re

import numpy as np
import torch

from build_sequences import range_bounds
from replay_normal import F, MAX_AGE, quantize_distance, rotate, unit
from train_normal import Store, confidence_gate, errors, gap_prediction, load_models, observer, singles


METHODS = ("step13", "wm_previous", "isotropic", "directional", "transport_2", "transport_5", "transport_full")
AGES = ("fresh", "age_1", "age_2", "age_3_plus", "beyond_12")
FRESH_DISTANCES = ("fresh_under_10m", "fresh_10_to_25m", "fresh_25_to_40m", "fresh_at_least_40m")


def world_position(relative, rows):
    return rows[:, 14:16] + rotate(relative, rows[:, 16])


def step13(store, game, models, calibration, config):
    _, rows, _ = store.games[game]
    reference = rows[..., 4:6]
    estimate = reference.copy()
    active = np.argwhere(rows[..., 17] <= MAX_AGE)
    with torch.inference_mode():
        for start in range(0, len(active), 2048):
            points = active[start:start+2048]
            features, current = store.batch(game, points)
            target = gap_prediction(models[config["gap"]], features, current).numpy()
            if config["fallback"] != "none":
                short = (current[:,18]>0) & (features[...,14].sum(axis=1)<=1)
                target[short] = singles(models["single"],calibration,features,current)[config["fallback"]][short]
            reference_local = observer(current, current[:,4:6])
            chosen = confidence_gate(target, reference_local, current[:,17], config["confidence_fallback"])
            estimate[tuple(points.T)] = world_position(chosen,current)
    return estimate


def bounded_velocity(v, limit):
    norm = np.linalg.norm(v,axis=-1)
    return v*np.minimum(1,limit/np.maximum(norm,1e-6))[:,None]


def candidate_positions(rows, pose):
    # Reproduce the server's observable polar cells around the measured ray.
    bearing = np.deg2rad(rows[:,13])
    origin, face = pose[:2], np.deg2rad(pose[2])
    distance = rows[:,12]
    focus_distance, focus_bearing = pose[5], np.deg2rad(pose[6])
    low,high,_ = range_bounds(distance[None], bearing[None], np.array([focus_distance]),np.array([focus_bearing]))
    radius = low.ravel()[:,None,None] + (high-low).ravel()[:,None,None]*np.linspace(.001,.999,21)[None,None,:]
    angles = bearing[:,None] + np.deg2rad(np.array([-.45,0,.45]))
    focus_delta = angles - focus_bearing
    focus = np.sqrt(np.maximum(0,radius**2+focus_distance**2
                               -2*radius*focus_distance*np.cos(focus_delta)[...,None]))
    valid = np.abs(quantize_distance(radius,focus)-distance[:,None,None])<.02
    valid = np.where(valid.any(axis=(1,2))[:,None,None],valid,True)
    points = origin + radius[...,None]*unit(angles+face)[...,None,:]
    width = (high-low).ravel()
    return points.reshape(len(rows),-1,2),valid.reshape(len(rows),-1),width


def update(prior, points, valid, heading, parallel, transverse):
    delta=points-prior[:,None]
    side=np.stack((-heading[:,1],heading[:,0]),axis=1)
    along=(delta*heading[:,None]).sum(axis=-1)/parallel[:,None]
    across=(delta*side[:,None]).sum(axis=-1)/transverse[:,None]
    log_weight=np.where(valid,-.5*(along**2+across**2),-np.inf)
    weight=np.exp(log_weight-log_weight.max(axis=-1,keepdims=True))
    return (points*weight[...,None]).sum(axis=1)/weight.sum(axis=1,keepdims=True)


def alternatives(rows, pose, cycles, kinematics, reference, speed_limit):
    tracks,steps,_=rows.shape
    wm=rows[...,4:6]
    result=np.repeat(reference[None],len(METHODS)-1,axis=0)
    posterior=np.zeros((2,tracks,2),np.float32)
    correction=np.zeros((tracks,2),np.float32)
    anchor=np.zeros(tracks,bool)
    prev_active=np.zeros(tracks,bool)
    prev_age=np.full(tracks,MAX_AGE+1)
    for t in range(steps):
        gap=t==0 or cycles[t]-cycles[t-1]!=1
        if gap:
            prev_active.fill(False)
            anchor.fill(False)
        current=rows[:,t]
        age=current[:,17]
        fresh=current[:,18]>0
        active=age<=MAX_AGE
        if gap:
            velocity=np.zeros((tracks,2),np.float32)
            wm_reliable=np.zeros(tracks,bool)
        else:
            state=kinematics[:,t-1]
            wm_reliable=np.isfinite(state).all(axis=-1)&(state[:,2]<=6)
            # Replay velocities are stored in the observer frame.
            own_velocity=rotate(rows[:,t-1,F+6:F+8],rows[:,t-1,16])
            limit = speed_limit[:, t - 1] if isinstance(speed_limit, np.ndarray) else speed_limit
            velocity=bounded_velocity(np.where(wm_reliable[:,None],state[:,:2],own_velocity),limit)
            wm_prior=wm[:,t-1]+velocity
            viable=active&~fresh&wm_reliable&np.isfinite(wm_prior).all(axis=-1)
            result[0,viable,t]=wm_prior[viable]

        carried=posterior+velocity[None]
        alive=prev_active&active
        indexes=np.flatnonzero(fresh)
        if len(indexes):
            points,valid,width=candidate_positions(current[indexes],pose[t])
            d=current[indexes,12]
            elapsed=np.minimum(MAX_AGE,prev_age[indexes]+1)
            speed=np.linalg.norm(velocity[indexes],axis=-1)
            heading=velocity[indexes]/np.maximum(speed[:,None],1e-6)
            moving=speed>.08
            heading=np.where(moving[:,None],heading,[1,0])
            # Model the motion uncertainty along and across the measured
            # direction of travel. The isotropic control has equal trace.
            along=.25+.30*elapsed+.01*d+.10*width
            cross=.16+.13*elapsed+.006*d+.06*width
            iso=np.sqrt((along**2+cross**2)/2)
            cross=np.where(moving,cross,iso)
            along=np.where(moving,along,iso)
            for j,sigma in enumerate(((iso,iso),(along,cross))):
                corrected=update(carried[j,indexes],points,valid,heading,*sigma)
                carried[j,indexes]=np.where(alive[indexes,None],corrected,reference[indexes,t])
            paired=np.isfinite(reference[indexes,t]).all(axis=-1)&np.isfinite(wm[indexes,t]).all(axis=-1)
            anchor[indexes]=paired
            correction[indexes]=np.where(paired[:,None],reference[indexes,t]-wm[indexes,t],0)

        output=alive|fresh
        for j in range(2):
            result[1+j,output,t]=carried[j,output]
        posterior=np.where(output[None,:,None],carried,0)
        anchor &= active
        stale=active&~fresh&anchor&np.isfinite(wm[:,t]).all(axis=-1)
        for j,tau in enumerate((2,5,np.inf)):
            result[3+j,stale,t]=wm[stale,t]+correction[stale]*np.exp(-age[stale,None]/tau)
        prev_active=output
        prev_age=age.copy()
    return dict(zip(METHODS[1:],result))


def public_speed(source, game, number, side):
    path=source/game/f"{side}_{number}"/"meta.json"
    meta=json.loads(path.read_text())
    return max([float(meta["server_param"]["player_speed_max"]),
                *[float(p["player_speed_max"]) for p in meta["player_types"]]])


TYPE_CHANGE = re.compile(r"\(change_player_type (\d+) (\d+)\)")


def teammate_limits(path, cycles, ids, side, shared_speed):
    meta = json.loads((path / "meta.json").read_text())
    dash = float(meta["server_param"]["max_dash_power"])
    # Same steady-state running-speed formula as librcsc realSpeedMax().
    # It is a prior regularizer, not a hard physical displacement bound.
    speeds = {
        int(kind["id"]): min(float(kind["player_speed_max"]),
                             dash * kind["dash_power_rate"] * kind["effort_max"] / (1 - kind["player_decay"]))
        for kind in meta["player_types"]
    }
    lookup = {int(value): i for i, value in enumerate(ids)}
    limits = np.full((len(ids), len(cycles)), shared_speed, np.float32)
    offset = 0 if side == "l" else 11
    with gzip.open(path / "raw_stream.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dir"] != "recv":
                continue
            match = TYPE_CHANGE.fullmatch(row["msg"])
            if match:
                target, kind = int(match[1]) + offset, int(match[2])
                if target in lookup:
                    start = np.searchsorted(cycles, int(row["cycle"]))
                    limits[lookup[target], start:] = speeds.get(kind, shared_speed)
    return limits


def run_split(args, split, selected=None):
    store=Store(args.associated,args.events,split,1)
    known=Store(args.known,args.events,split,1)
    models, calibration=load_models(args.runs)
    config=json.loads((args.runs/"audited-validation.json").read_text())["config"]
    games=json.loads((args.events/"manifest.json").read_text())["games"]
    numbers={game["name"]:game["observer_unum"] for game in games}
    methods = (*METHODS, "librcsc")
    groups = ("all", *AGES, *FRESH_DISTANCES)
    if args.individual_types:
        methods += ("directional_types",)
        groups += ("teammates", "teammates_fresh", "teammates_age_3_12")
    stats = {group: {name: [] for name in methods} for group in groups}
    per_game={}
    cache=[]
    for i,(name,rows,cycles) in enumerate(store.games):
        if known.games[i][0]!=name:
            raise ValueError("known and associated manifests differ")
        side="lr"[i%2]
        label=f"{name}-{side}"
        with np.load(args.events/f"{label}.npz",allow_pickle=False) as events:
            pose, ids = events["pose"], events["ids"]
            if not np.array_equal(events["cycle"],cycles):
                raise ValueError(f"{label}: event alignment mismatch")
        kin=np.load(args.kinematics/f"{label}.npy",mmap_mode="r")
        if kin.shape!=(*rows.shape[:2],3):
            raise ValueError(f"{label}: kinematics alignment mismatch")
        reference=step13(store,i,models,calibration,config)
        speed = public_speed(args.source, name, numbers[name], side)
        alternatives_world = alternatives(rows, pose, cycles, kin, reference, speed)
        if args.individual_types:
            path = args.source / name / f"{side}_{numbers[name]}"
            limits = teammate_limits(path, cycles, ids, side, speed)
            typed = alternatives(rows, pose, cycles, kin, reference, limits)
            alternatives_world["directional_types"] = typed["directional"]
        flat=rows.reshape(-1,rows.shape[-1])
        supported=np.isfinite(flat[:,:6]).all(axis=-1)
        flat=flat[supported]
        prior_age=known.games[i][1][...,17].reshape(-1)[supported]
        # Selection uses the age of the track after anonymous association.
        # Reporting uses explicit-observation age for consistency with Step 13.
        age=flat[:,17]
        bucket=np.where(age>MAX_AGE,4,np.minimum(age,3)).astype(int)
        report=np.where(prior_age>MAX_AGE,4,np.minimum(prior_age,3)).astype(int)
        distance = np.searchsorted([10, 25, 40], flat[:, 2], side="right")
        groups = {group: report == j for j, group in enumerate(AGES)}
        groups.update({group: (report == 0) & (distance == j)
                       for j, group in enumerate(FRESH_DISTANCES)})
        if args.individual_types:
            teammates = np.broadcast_to(((ids <= 11) == (side == "l"))[:, None], rows.shape[:2])
            teammates = teammates.ravel()[supported]
            groups.update(teammates=teammates, teammates_fresh=teammates & (report == 0),
                          teammates_age_3_12=teammates & (report == 3))
        positions={"step13":reference,"librcsc":rows[...,4:6],**alternatives_world}
        score={}
        for method,world in positions.items():
            rel, glob=errors(flat,observer(flat,world.reshape(-1,2)[supported]))
            if not np.isfinite(rel).all():
                raise ValueError(f"{label}/{method}: invalid prediction")
            score[method]=np.stack((rel,glob),axis=-1)
            stats["all"][method].append(score[method])
            for group, mask in groups.items():
                stats[group][method].append(score[method][mask])
        cache.append((name, bucket, groups, score))
        print(f"{split}: {i+1}/{len(store.games)} observers",flush=True)

    if selected is None:
        selected={}
        for i,group in enumerate(AGES):
            totals={}
            count=sum(int((bucket==i).sum()) for _,bucket,_,_ in cache)
            for method in methods:
                totals[method]=sum(float(data[method][bucket==i,0].sum(dtype=np.float64))
                                   for _,bucket,_,data in cache)
            selected[group]=min(totals,key=totals.get) if count>=100 else "step13"
    for group in stats:
        stats[group]["selected"]=[]
    for name, bucket, groups, score in cache:
        combined=score["step13"].copy()
        for i,group in enumerate(AGES):
            mask=bucket==i
            combined[mask]=score[selected[group]][mask]
        stats["all"]["selected"].append(combined)
        for group, mask in groups.items():
            stats[group]["selected"].append(combined[mask])
        entry=per_game.setdefault(name,{"n":0,"sum_relative_gain_m":0.0})
        entry["n"]+=len(combined)
        entry["sum_relative_gain_m"]+=float((score["step13"][:,0]-combined[:,0]).sum(dtype=np.float64))
    metrics={}
    for group,methods in stats.items():
        metrics[group]={}
        for method,batches in methods.items():
            data=np.concatenate(batches)
            if len(data):
                m=data.mean(axis=0,dtype=np.float64)
                p=np.quantile(data,.9,axis=0).astype(float)
                metrics[group][method]={"n":len(data),"relative_mean_m":m[0],"relative_p90_m":p[0],
                                        "global_mean_m":m[1],"global_p90_m":p[1]}
    counts=np.array([g["n"] for g in per_game.values()])
    gains=np.array([g["sum_relative_gain_m"] for g in per_game.values()])
    draw=np.random.default_rng(42).integers(len(counts),size=(10000,len(counts)))
    boot=gains[draw].sum(axis=1)/counts[draw].sum(axis=1)
    return {"selection":selected,"metrics":metrics,"per_game":per_game,
            "gain_ci95_m":np.quantile(boot,[.025,.975]).tolist()}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","events","known","associated","kinematics","runs","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--individual-types", action="store_true")
    p.add_argument("--threads",type=int,default=8)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    validation=run_split(args,"validation")
    print("Selected on validation:",validation["selection"],flush=True)
    test=run_split(args,"test",validation["selection"])
    args.output.write_text(json.dumps({"validation":validation,"test":test},indent=2)+"\n")
