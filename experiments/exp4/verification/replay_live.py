#!/usr/bin/env python3
"""Rebuild logged pre-apply inputs and compare Python with live C++ candidates.

Schema 3 retains stock ages, ownership and component flags. Truth supplies labels
only. Omit each segment's initial windows whose pre-apply history is incomplete.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp3/code"))
from build_dataset import observer
from build_ball_bounds import build
from train_state import Model, Stream


def run(source, runs, output):
    meta=json.loads((source/"meta.json").read_text())
    assert meta["schema_version"] >= 3
    original=pd.read_csv
    def reader(p,*args,**kw):
        frame=original(p,*args,**kw)
        if str(p).endswith("objects.csv.gz"):
            fields=("pos_x","pos_y","vel_x","vel_y","body","pos_count","vel_count","body_count")
            extra=original(p,usecols=["exp4_vel_x","object_id","ls_pos_x","ls_pos_y",*["exp4_stock_"+f for f in fields]])
            ready=extra.exp4_vel_x.notna()
            # Step26 changes only decision position. Recover the fresh localization
            # for incomplete/disabled Exp4 player windows as well.
            fresh=(extra.object_id!=0)&extra.ls_pos_x.notna()
            frame.loc[fresh,["trk_pos_x","trk_pos_y"]]=extra.loc[fresh,["ls_pos_x","ls_pos_y"]].values
            for f in fields:frame.loc[ready,"trk_"+f]=extra.loc[ready,"exp4_stock_"+f]
            frame.loc[ready & extra.exp4_stock_vel_x.isna(),"trk_vel_count"]=np.nan
            frame.loc[ready & extra.exp4_stock_body.isna(),"trk_body_count"]=np.nan
        return frame
    pd.read_csv=reader
    try:observer(source,output)
    finally:pd.read_csv=original
    build((source,output))
    live=original(source/"objects.csv.gz")
    live=live[(live.stopped==0)&live.object_id.eq(0)].set_index("cycle")
    data=Stream(output,True,1)
    with np.load(output/"timing.npz") as t:cycles=t["cycles"]
    live=live.reindex(cycles)
    for name,attr in [("exp4_controlled","controlled"),("exp4_owner","owner")]:
        a=live[name].to_numpy()
        old=getattr(data,attr)
        setattr(data,attr,np.where(np.isfinite(a),np.nan_to_num(a),old).astype(old.dtype))
    models=[]
    for variant in ["ball_quantized","ball_experts"]:
        model=Model(variant).eval()
        model.load_state_dict(torch.load(runs/(variant+".pt"),weights_only=True,map_location="cpu"))
        models.append(model)
    comparisons=[]
    errors=[]
    with torch.inference_mode():
        for start in range(0,len(data.points),1024):
            pts=data.points[start:start+1024]
            # At least eight prior candidate records prove all retained stock
            # history is available for reconstruction.
            complete=live.exp4_vel_x.notna().to_numpy()
            mask=np.array([complete[t-7:t+1].all() for t in pts[:,0]])
            pts=pts[mask]
            if not len(pts):continue
            x,c,o,b,target,g,tags=data.batch(pts,True)
            now=models[0](*map(torch.from_numpy,(x,c,o,b,data.controlled[pts[:,0]].copy(),g)))
            pred=models[1](*map(torch.from_numpy,(x,c,o,b,data.controlled[pts[:,0]].copy(),g))).numpy()
            pred[:,:2]=now.numpy()[:,:2]
            logged=live.iloc[pts[:,0]][["exp4_pos_x","exp4_pos_y","exp4_vel_x","exp4_vel_y","exp4_next_pos_x","exp4_next_pos_y","exp4_next_vel_x","exp4_next_vel_y"]].values
            if meta["side"] == "r": logged *= -1
            comparisons.append(np.linalg.norm((pred-logged).reshape(-1,4,2),axis=2))
            errors.append(np.concatenate((np.linalg.norm((b-target).reshape(-1,4,2),axis=2),
                                          np.linalg.norm((pred-target).reshape(-1,4,2),axis=2)),axis=1))
    a=np.concatenate(comparisons);e=np.concatenate(errors)
    return dict(source=str(source),n=len(a),mean_cpp_difference=a.mean(axis=0).tolist(),
                max_cpp_difference=a.max(axis=0).tolist(),p99_cpp_difference=np.quantile(a,.99,axis=0).tolist(),
                mean_error_stock_then_candidate=e.mean(axis=0).tolist())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("runs",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args();torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="exp4-parity-live-") as tmp:
        result=run(args.source,args.runs,Path(tmp))
    args.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
