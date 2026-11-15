#!/usr/bin/env python3
"""Train gap-aware and single-observation baselines and select a normal-view stack."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from replay_normal import F, FEATURES, HISTORY, MAX_AGE, rotate, unit

GAP_MODES = ("gap_blend", "gap_quantized")
FALLBACKS = ("midpoint", "median", "mlp")


class Store:
    def __init__(self, data, events, split, stride=1):
        manifest=json.loads((data/"manifest.json").read_text())
        self.games=[]
        for game in manifest["games"]:
            if game["split"]!=split:
                continue
            for side in "lr":
                name=f'{game["name"]}-{side}'
                rows=np.load(data/f"{name}.npy",mmap_mode="r",allow_pickle=False)
                with np.load(events/f"{name}.npz",allow_pickle=False) as raw:
                    cycles=raw["cycle"]
                if rows.shape[1]!=len(cycles):
                    raise ValueError(f"{name}: replay cycle mismatch")
                self.games.append((game["name"],rows,cycles))
        if not self.games:
            raise ValueError(f"no {split} games")
        self.stride=stride

    def endpoints(self, active=False, fresh=False):
        result=[]
        for _,rows,_ in self.games:
            data=rows[:,::self.stride]
            mask=np.isfinite(data[...,:4]).all(axis=-1)
            if active:
                mask &= data[...,17]<=MAX_AGE
            if fresh:
                mask &= data[...,18]>0
            points=np.argwhere(mask).astype(np.int32)
            points[:,1]*=self.stride
            result.append(points)
        return result

    def batch(self, game, points):
        _, rows, cycles=self.games[game]
        track,end=points.T
        indexes=np.maximum(0,end[:,None]+np.arange(1-HISTORY,1))
        feature=rows[track[:,None],indexes,F:].copy()
        contiguous=cycles[end,None]-cycles[indexes]==np.arange(HISTORY-1,-1,-1)
        feature[~contiguous]=0
        return feature,rows[track,end].copy()


class GapGRU(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode=mode
        self.gru=nn.GRU(FEATURES,48,batch_first=True)
        self.output=nn.Linear(48,2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature, anchor, age):
        if self.mode=="gap_blend":
            # Remove the quantized state and velocity channels from this ablation.
            feature=feature.clone()
            feature[...,2:4]=0
            feature[...,6:8]=0
        states,_=self.gru(feature)
        radius=.4+.035*torch.linalg.vector_norm(anchor,dim=-1)+.22*age
        return anchor+radius[:,None]*self.output(states[:,-1]).tanh()


class SingleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, sample, base):
        return base+.5*(sample[:,3]*4)*self.layers(sample).squeeze(-1).tanh()


def observer(rows, position):
    return rotate(position-rows[:,14:16],-rows[:,16])


def truth_relative(rows):
    return rows[:,2,None]*unit(np.deg2rad(rows[:,3]))


def gap_prediction(model, features, rows):
    start=6 if model.mode=="gap_blend" else 8
    anchor=observer(rows,np.nan_to_num(rows[:,start:start+2],nan=0.0))
    estimate=model(torch.from_numpy(features),torch.from_numpy(anchor),
                   torch.from_numpy(np.minimum(MAX_AGE,rows[:,17])))
    return estimate


def medians(store, points):
    bins, residuals=[],[]
    for (_,rows,_), endpoints in zip(store.games,points):
        data=rows[tuple(endpoints.T)]
        bins.append(np.rint(data[:,12]*10).astype(np.int32))
        residuals.append(data[:,2]-np.linalg.norm(data[:,10:12]-data[:,14:16],axis=-1))
    keys=np.concatenate(bins)
    values=np.concatenate(residuals)
    order=np.argsort(keys,kind="stable")
    keys,values=keys[order],values[order]
    unique,start,n=np.unique(keys,return_index=True,return_counts=True)
    corrections=np.zeros(keys.max()+1,np.float32)
    for bin_,i,count in zip(unique,start,n):
        if count>=32:
            corrections[bin_]=np.median(values[i:i+count])
    return corrections


def singles(model, calibration, features, rows):
    r=np.linalg.norm(rows[:,10:12]-rows[:,14:16],axis=-1)
    angle=np.deg2rad(rows[:,13])
    width=features[:,-1,11]*4
    valid=np.isfinite(r)
    idx=np.rint(np.nan_to_num(rows[:,12])*10).astype(int)
    correction=np.where(idx<len(calibration),calibration[np.minimum(idx,len(calibration)-1)],0)
    corrected=r+np.clip(correction,-width/2,width/2)
    learned=model(torch.from_numpy(features[:,-1,8:13].copy()),torch.from_numpy(np.nan_to_num(r))).numpy()
    learned[~valid]=np.nan
    ray=unit(angle)
    return dict(zip(FALLBACKS,[r[:,None]*ray,corrected[:,None]*ray,learned[:,None]*ray]))


def fit(args, mode, gap_model=GapGRU):
    torch.manual_seed(42)
    rng=np.random.default_rng(42)
    train=Store(args.known,args.events,"train")
    valid=Store(args.known,args.events,"validation",24)
    single=mode=="single"
    model=SingleMLP() if single else gap_model(mode)
    points=train.endpoints(active=True,fresh=single)
    validation=valid.endpoints(active=True,fresh=single)
    n=np.array([len(rows) for rows in points])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=float("inf")
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        for game in rng.choice(len(n),args.batches,p=n/n.sum()):
            f,r=train.batch(game,points[game][rng.integers(n[game],size=args.batch_size)])
            if single:
                base=np.linalg.norm(r[:,10:12]-r[:,14:16],axis=-1)
                pred=model(torch.from_numpy(f[:,-1,8:13].copy()),torch.from_numpy(base))
                loss=nn.functional.smooth_l1_loss(pred,torch.from_numpy(r[:,2]),beta=.3)
            else:
                pred=gap_prediction(model,f,r)
                error=torch.linalg.vector_norm(pred-torch.from_numpy(truth_relative(r)),dim=-1)
                loss=nn.functional.smooth_l1_loss(error,torch.zeros_like(error),beta=.25)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
        model.eval()
        total=count=0
        with torch.inference_mode():
            for game,ends in enumerate(validation):
                for start in range(0,len(ends),768):
                    f,r=valid.batch(game,ends[start:start+768])
                    if single:
                        base=np.linalg.norm(r[:,10:12]-r[:,14:16],axis=-1)
                        pred=model(torch.from_numpy(f[:,-1,8:13].copy()),torch.from_numpy(base)).numpy()
                        est=pred[:,None]*unit(np.deg2rad(r[:,13]))
                    else:
                        est=gap_prediction(model,f,r).numpy()
                    error=np.linalg.norm(est-truth_relative(r),axis=-1)
                    total+=float(error.sum(dtype=np.float64))
                    count+=len(error)
        score=total/count
        history.append({"epoch":epoch,"validation_mean_m":score})
        print(f"{mode} epoch {epoch}: {score:.4f} m relative on {count:,} endpoints",flush=True)
        if score<best:
            best=score
            torch.save(model.state_dict(),args.output/f"{mode}.pt")
        (args.output/f"{mode}-training.json").write_text(json.dumps(history,indent=2)+"\n")
    if single:
        np.save(args.output/"bin-medians.npy",medians(train,points),allow_pickle=False)


def load_models(root):
    models={mode:GapGRU(mode) for mode in GAP_MODES}
    models["single"]=SingleMLP()
    for key,model in models.items():
        model.load_state_dict(torch.load(root/f"{key}.pt",weights_only=True))
        model.eval()
    return models,np.load(root/"bin-medians.npy",allow_pickle=False)


def predictions(store, models, calibration, game, points):
    f,r=store.batch(game,points)
    results={}
    for name,start in (("librcsc",4),("blend",6),("quantized",8),("midpoint",10)):
        results[name]=observer(r,r[:,start:start+2])
    for mode in GAP_MODES:
        estimate=gap_prediction(models[mode],f,r).numpy()
        estimate[r[:,17]>MAX_AGE]=np.nan
        results[mode]=estimate
    results.update(singles(models["single"],calibration,f,r))
    short=(r[:,18]>0)&(f[...,14].sum(axis=1)<=1)
    return r,results,short


def stack(pred, short, config):
    estimate=pred[config["gap"]].copy()
    if config["fallback"]!="none":
        estimate[short]=pred[config["fallback"]][short]
    return np.where(np.isfinite(estimate).all(axis=1)[:,None],estimate,pred["librcsc"])


def cohorts(known, associated, models, calibration, config=None):
    values=[]
    points=known.endpoints()
    with torch.inference_mode():
        for game,ends in enumerate(points):
            for start in range(0,len(ends),2048):
                part=ends[start:start+2048]
                r,p,short=predictions(known,models,calibration,game,part)
                a,pa,short_a=predictions(associated,models,calibration,game,part)
                if not np.array_equal(a[:,:4],r[:,:4]):
                    raise ValueError("known and associated labels differ")
                values.append((r,a,p,pa,short,short_a))
    r,a=[np.concatenate([x[i] for x in values]) for i in (0,1)]
    p,pa=[{k:np.concatenate([x[i][k] for x in values]) for k in values[0][i]} for i in (2,3)]
    short,short_a=[np.concatenate([x[i] for x in values]) for i in (4,5)]
    return r,a,p,pa,short,short_a,[len(endpoints) for endpoints in points]


def errors(rows, positions):
    relative=np.linalg.norm(positions-truth_relative(rows),axis=-1)
    global_pos=rotate(positions,rows[:,16])+rows[:,14:16]
    world=np.linalg.norm(global_pos-rows[:,:2],axis=-1)
    return relative,world


def gate(estimate, baseline, ages, groups):
    age=np.minimum(ages.astype(int),3)
    keep=(np.linalg.norm(estimate,axis=-1)<10)&np.isin(age,groups)
    return np.where((keep&np.isfinite(baseline).all(axis=1))[:,None],baseline,estimate)


def confidence_gate(estimate, baseline, ages, groups):
    age = np.minimum(ages.astype(int), 3)
    distance = np.searchsorted([10, 25], np.linalg.norm(estimate, axis=-1), side="right")
    keep = np.zeros(len(age), dtype=bool)
    for range_bin, age_group in groups:
        keep |= (distance == range_bin) & (age == age_group)
    return np.where((keep & np.isfinite(baseline).all(axis=1))[:, None], baseline, estimate)


def analyze(args, split, config=None):
    stride=args.stride if split=="validation" else 1
    known=Store(args.known,args.events,split,stride)
    associated=Store(args.associated,args.events,split,stride)
    if [g[0] for g in known.games]!=[g[0] for g in associated.games]:
        raise ValueError("replay manifests use different games")
    models,calibration=load_models(args.output)
    r,a,p,pa,short,short_a,game_sizes=cohorts(known,associated,models,calibration)
    common=np.isfinite(p["librcsc"]).all(axis=1)
    selected=common&(r[:,17]<=MAX_AGE)
    fresh=common&(r[:,18]>0)
    if config is None:
        best=lambda methods,cohort: min(methods,key=lambda key:errors(r,p[key])[0][cohort].mean())
        gap=best(GAP_MODES,selected)
        fallback_mask=selected&short
        fallback=best((*FALLBACKS,gap),fallback_mask)
        config={"gap":gap,"fallback":fallback if fallback!=gap else "none"}
        known_stack=stack(p,short,config)
        assoc_stack=stack(pa,short_a,config)
        config["associate"]=bool(errors(r,assoc_stack)[0][common].mean()<errors(r,known_stack)[0][common].mean())
        pre=assoc_stack if config["associate"] else known_stack
        ages=a[:,17] if config["associate"] else r[:,17]
        e=errors(r,pre)[0]
        b=errors(r,p["librcsc"])[0]
        close=common&(np.linalg.norm(pre,axis=-1)<10)
        group=np.minimum(ages.astype(int),3)
        config["gate_ages"]=[age for age in range(4)
                             if (mask:=close&(group==age)).sum()>=100 and b[mask].mean()<e[mask].mean()]
        range_bins = np.searchsorted([10, 25], np.linalg.norm(pre, axis=-1), side="right")
        config["confidence_fallback"] = [
            [distance, age] for distance in range(3) for age in range(4)
            if (mask := common & (range_bins == distance) & (group == age)).sum() >= 100
            and b[mask].mean() < e[mask].mean()
        ]
    known_stack,assoc_stack=stack(p,short,config),stack(pa,short_a,config)
    pre=assoc_stack if config["associate"] else known_stack
    ages=a[:,17] if config["associate"] else r[:,17]
    protected=gate(pre,p["librcsc"],ages,config["gate_ages"])
    confident=confidence_gate(pre,p["librcsc"],ages,config["confidence_fallback"])
    all_methods={**p,"known_stack":known_stack,"associated_stack":assoc_stack,
                 "protected_stack":protected,"confidence_stack":confident}
    groups={"all_librcsc":common,"active_common":selected,"fresh":fresh,"short_history":selected&short,
            "under_10m":common&(r[:,2]<10),"10_to_20m":common&(r[:,2]>=10)&(r[:,2]<20),
            "20_to_40m":common&(r[:,2]>=20)&(r[:,2]<40),"40m_plus":common&(r[:,2]>=40)}
    groups.update({f"age_{i}":common&(r[:,17]==i) for i in range(3)})
    groups["age_3_plus"]=common&(r[:,17]>=3)&(r[:,17]<=MAX_AGE)
    result={}
    for name,estimate in all_methods.items():
        e,g=errors(r,estimate)
        for key,mask in groups.items():
            ok=mask&np.isfinite(e)&np.isfinite(g)
            if ok.any():
                result.setdefault(key,{})[name]={"n":int(ok.sum()),"relative_mean_m":float(e[ok].mean(dtype=np.float64)),
                                                "relative_p90_m":float(np.quantile(e[ok],.9)),
                                                "global_mean_m":float(g[ok].mean(dtype=np.float64))}
    audit=Counter()
    manifest=json.loads((args.associated/"manifest.json").read_text())
    for game in manifest["games"]:
        if game["split"]==split:
            audit.update(game["identity_audit"])
    coverage={"truth_slots":len(r),"librcsc":int(common.sum()),"known_active":int((r[:,17]<=MAX_AGE).sum()),
              "associated_active":int((a[:,17]<=MAX_AGE).sum()),"short_history":int(short.sum()),
              "accepted_anonymous":int((a[:,18]==2).sum()),
              "final":int(np.isfinite(confident).all(axis=1).sum()),
              "guard_replaced":int((np.isfinite(pre).all(axis=1) & np.any(confident != pre, axis=1)).sum())}
    game_stats={}
    improvement=errors(r,p["librcsc"])[0]-errors(r,confident)[0]
    offset=0
    for (name,_,_),size in zip(known.games,game_sizes):
        end=offset+size
        mask=common[offset:end]
        entry=game_stats.setdefault(name,{"n":0,"sum_gain_m":0.0})
        entry["n"]+=int(mask.sum())
        entry["sum_gain_m"]+=float(improvement[offset:end][mask].sum(dtype=np.float64))
        offset=end
    counts=np.array([game["n"] for game in game_stats.values()])
    gains=np.array([game["sum_gain_m"] for game in game_stats.values()])
    draw=np.random.default_rng(42).integers(len(counts),size=(10000,len(counts)))
    bootstrap=gains[draw].sum(axis=1)/counts[draw].sum(axis=1)
    return {"split":split,"stride":stride,"config":config,"scores":result,"coverage":coverage,
            "identity_audit":dict(audit),"games":game_stats,
            "gain_ci95_m":np.quantile(bootstrap,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("events","known","associated","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--train",choices=(*GAP_MODES,"single"))
    p.add_argument("--evaluate",action="store_true")
    p.add_argument("--result-prefix",default="")
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batches",type=int,default=256)
    p.add_argument("--batch-size",type=int,default=192)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--stride",type=int,default=10)
    a=p.parse_args()
    if bool(a.train)==a.evaluate:
        p.error("use one of --train or --evaluate")
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True,exist_ok=True)
    if a.train:
        fit(a,a.train)
    else:
        validation=analyze(a,"validation")
        (a.output/f"{a.result_prefix}validation.json").write_text(json.dumps(validation,indent=2)+"\n")
        print(validation["config"],flush=True)
        test=analyze(a,"test",validation["config"])
        (a.output/f"{a.result_prefix}test.json").write_text(json.dumps(test,indent=2)+"\n")
        print(test["coverage"],test["identity_audit"],flush=True)


if __name__=="__main__":
    main()
