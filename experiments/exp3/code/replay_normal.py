#!/usr/bin/env python3
"""Replay known and reachability-associated player tracks through normal-view gaps."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from build_sequences import range_bounds

MAX_AGE = 12
HISTORY = 8
CAP = 1.5
FEATURES = 18
F = 19
RAYS = 21


def unit(angle):
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def length(x):
    return np.sqrt((x*x).sum(axis=-1))


def cap(x):
    return x / np.maximum(1., length(x)[:, None] / CAP)


def rotate(x, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c*x[..., 0]-s*x[..., 1], s*x[..., 0]+c*x[..., 1]), axis=-1)


def quantize_distance(d, f):
    def log_round(x):
        return np.exp(.1 * np.rint(np.log(x + 1e-10) / .1))
    return .1 * np.rint(np.maximum(0, (d - f + log_round(d) + log_round(f)) / 2) / .1)


def measurements(events, pose):
    bearing = np.deg2rad(events[..., 3])
    focus_dir = np.deg2rad(pose[:, 6])
    low, high, _ = range_bounds(events[..., 2], bearing, pose[:, 5], focus_dir)
    radii = low[..., None] + (high-low)[..., None] * np.linspace(.001, .999, RAYS, dtype=np.float32)
    focus = pose[:, 5, None, None]
    direction = np.cos(bearing - focus_dir[:, None])[..., None]
    f = np.sqrt(np.maximum(0, radii**2 + focus**2 - 2*radii*focus*direction))
    consistent = np.abs(quantize_distance(radii, f) - events[..., 2, None]) < .02
    consistent = np.where(consistent.any(axis=-1)[..., None], consistent, True)
    rays = unit(bearing + np.deg2rad(pose[:, 2, None]))
    midpoint = (low+high)/2
    return rays, radii, consistent, midpoint, high-low


def associate(events, ids, reserved, active, predicted, age, origin, rays, mid, width):
    unknown = np.flatnonzero(events[:, 0] == 0)
    candidates = np.flatnonzero(active & ~reserved & (age <= 8))
    if not len(unknown) or not len(candidates):
        return []
    displacement = predicted[candidates]-origin
    ray = rays[unknown]
    radial = displacement @ ray.T
    angular = displacement @ np.stack((-ray[:, 1], ray[:, 0]), axis=0)
    radial_error = np.maximum(0, np.abs(radial-mid[unknown]) - width[unknown]/2)
    tangential = np.maximum(0, np.abs(angular)-np.deg2rad(.5)*mid[unknown])
    cost = np.hypot(radial_error,tangential)
    side = np.where(ids[candidates] <= 11, 1, -1)
    ok = ((side[:, None]==events[unknown, 1]) | (events[unknown, 1]==0))
    cost = np.where(ok & (cost < .65 + .32*age[candidates,None]), cost, np.inf)
    pairs = []
    for j in range(cost.shape[1]):
        order = np.argsort(cost[:,j])
        i = order[0]
        value = cost[i,j]
        if not np.isfinite(value) or (len(order)>1 and cost[order[1],j]-value < .25):
            continue
        rivals = np.delete(cost[i],j)
        if len(rivals) and rivals.min()-value < .25:
            continue
        pairs.append((candidates[i],unknown[j]))
    return pairs


def identity_scores(events, ids, truth, known, pairs):
    """Truth-only audit for uniquely re-quantizable anonymous reports."""
    audit = Counter(accepted=len(pairs))
    if not pairs:
        return audit
    measured_range = quantize_distance(truth[:,2], truth[:,4])
    measured_dir = np.rint(truth[:,3])
    side = np.where(ids <= 11, 1, -1)
    for track, row in pairs:
        event = events[row]
        candidate = ((np.abs(measured_range-event[2]) < .02)
                     & (np.abs((measured_dir-event[3]+180)%360-180)<.1) & ~known
                     & ((event[1]==0) | (event[1]==side)))
        if candidate.sum()==1:
            audit["scored"] += 1
            audit["correct"] += int(candidate[track])
    return audit


def replay(data, allow_association):
    cycles, pose, events, ids, truth, wm = (data[key] for key in ("cycle","pose","events","ids","truth","wm"))
    rays, radii, consistent, mid, width = measurements(events,pose)
    size = len(ids)
    lookup = {i:j for j,i in enumerate(ids)}
    rows = np.full((len(cycles),size,F+FEATURES),np.nan,np.float32)
    rows[..., :4] = truth[...,:4]
    rows[..., 4:6] = wm
    rows[..., 14:16] = pose[:,None,:2]
    rows[..., 16] = np.deg2rad(pose[:,None,2])
    age = np.full(size, MAX_AGE+1, dtype=int)
    position = np.zeros((size,2),np.float32)
    velocity = np.zeros_like(position)
    qposition, qvelocity = position.copy(), velocity.copy()
    history = np.zeros(size,np.int32)
    audit = Counter(accepted=0,scored=0,correct=0)
    for t in range(len(cycles)):
        if t == 0 or cycles[t]-cycles[t-1]!=1:
            age.fill(MAX_AGE+1)
            history.fill(0)
        old_age = age.copy()
        age = np.minimum(age+1, MAX_AGE+1)
        previous = age <= MAX_AGE
        prior = position+velocity
        qprior = qposition+qvelocity
        origin, facing = pose[t,:2], np.deg2rad(pose[t,2])
        explicit = [(lookup[int(row[0])],i) for i,row in enumerate(events[t])
                    if np.isfinite(row[0]) and row[0]>0 and int(row[0]) in lookup]
        known = np.zeros(size,bool)
        for track,_ in explicit:
            known[track]=True
        recovered = (associate(events[t],ids,known,previous,qprior,age,origin,rays[t],mid[t],width[t])
                     if allow_association else [])
        if recovered:
            audit.update(identity_scores(events[t],ids,truth[t],known,recovered))
        fresh = np.zeros(size,np.float32)
        sample = np.zeros((size,5),np.float32)
        simple = prior.copy()
        quantized = qprior.copy()
        v = velocity*.83
        qv = qvelocity*.83
        pairs = explicit+recovered
        if pairs:
            target, event = np.array(pairs).T
            raw = events[t,event]
            direction = rays[t,event]
            dist = mid[t,event]
            observation = origin + dist[:,None]*direction
            continuing = previous[target]
            measurement = np.where(continuing[:,None],.65*observation+.35*prior[target],observation)
            simple[target] = measurement
            radial = ((qprior[target]-origin)*direction).sum(axis=-1)
            spread = .18+.24*age[target]+.009*dist
            log_likelihood = -.5*((radii[t,event]-radial[:,None])/spread[:,None])**2
            log_likelihood = np.where(consistent[t,event],log_likelihood,-np.inf)
            weights = np.exp(log_likelihood - log_likelihood.max(axis=-1,keepdims=True))
            posterior = (radii[t,event]*weights).sum(axis=-1)/weights.sum(axis=-1)
            radius = np.where(continuing,posterior,dist)
            quantized[target] = origin+radius[:,None]*direction
            speed_valid = np.isfinite(raw[:,4:6]).all(axis=-1)
            lateral = np.stack((-direction[:,1],direction[:,0]),axis=-1)
            raw_speed = np.nan_to_num(raw[:,4:5])*direction + (
                np.deg2rad(np.nan_to_num(raw[:,5:6]))*dist[:,None])*lateral + pose[t,3:5]
            elapsed = age[target,None]
            simple_speed = (measurement-position[target])/elapsed
            quant_speed = (quantized[target]-qposition[target])/elapsed
            v[target] = np.where(speed_valid[:,None],.55*raw_speed+.45*v[target],
                                 np.where(continuing[:,None],.35*simple_speed+.65*v[target],0))
            qv[target] = np.where(speed_valid[:,None],.55*raw_speed+.45*qv[target],
                                  np.where(continuing[:,None],.35*quant_speed+.65*qv[target],0))
            rows[t,target,10:12] = observation
            rows[t,target,12:14] = raw[:,2:4]
            sample[target] = np.column_stack((dist/60, np.sin(np.deg2rad(raw[:,3])),np.cos(np.deg2rad(raw[:,3])),
                                              width[t,event]/4,speed_valid))
            fresh[target] = np.where(known[target],1,2)
            age[target] = 0
        active = age <= MAX_AGE
        history = np.where(active,history+1,0)
        position = np.where(active[:,None],simple,0)
        qposition = np.where(active[:,None],quantized,0)
        velocity = np.where(active[:,None],cap(v),0)
        qvelocity = np.where(active[:,None],cap(qv),0)
        rows[t,active,6:8] = position[active]
        rows[t,active,8:10] = qposition[active]
        rows[t,:,17] = age
        rows[t,:,18] = fresh
        own_movement = pose[t,:2]-pose[t-1,:2] if t and cycles[t]-cycles[t-1]==1 else np.zeros(2)
        features = np.column_stack((
            rotate(position-origin,-facing)/60, rotate(qposition-origin,-facing)/60,
            rotate(velocity,-facing), rotate(qvelocity,-facing), sample,
            np.minimum(age,MAX_AGE)/MAX_AGE, (fresh>0), np.minimum(old_age,MAX_AGE)/MAX_AGE,
            np.broadcast_to(rotate(own_movement,-facing),(size,2)),
        ))
        rows[t,:,F:] = np.where(active[:,None],features,0)
    return np.ascontiguousarray(rows.transpose(1,0,2)),dict(audit)


def build_game(job):
    root, output, game, associate = job
    counts = Counter()
    for side in "lr":
        with np.load(root/f"{game}-{side}.npz") as data:
            rows,audit = replay(data,associate)
        np.save(output/f"{game}-{side}.npy",rows,allow_pickle=False)
        counts.update(audit)
    return game,dict(counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("events",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--associate",action="store_true")
    p.add_argument("--workers",type=int,default=24)
    a=p.parse_args()
    manifest=json.loads((a.events/"manifest.json").read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    jobs=[(a.events,a.output,g["name"],a.associate) for g in manifest["games"]]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,(game,audit) in enumerate(pool.map(build_game,jobs),1):
            manifest["games"][i-1]["identity_audit"]=audit
            if i==1 or i%10==0 or i==len(jobs):
                print(f"Replayed {i}/{len(jobs)} games",flush=True)
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
