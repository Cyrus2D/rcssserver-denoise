#!/usr/bin/env python3
"""Test frozen models offline on new games' untreated observers."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"exp3/code"))
from build_dataset import observer
from build_ball_bounds import build
from train_state import Model, Stream, predict, error


def create(job):
    source, out = job
    out.mkdir(parents=True)
    observer(source,out)
    build((source,out))
    return out

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs",type=Path)
    parser.add_argument("game_logs",type=Path)
    parser.add_argument("runs",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    games=[g for g in sorted(args.logs.iterdir()) if (args.game_logs/"result.d"/g.name).exists()
           and "Score:" in (args.game_logs/"result.d"/g.name).read_text()]
    chosen=[games[i] for i in np.linspace(0,len(games)-1,8,dtype=int)]
    root=Path(tempfile.mkdtemp(prefix="exp4-f3-control-"))
    with ProcessPoolExecutor(8) as pool:
        paths=list(pool.map(create,[(g/"r_2",root/g.name) for g in chosen]))
    scores={"player":{},"ball":{}}
    for ball,names in [(False,["player_context"]),(True,["ball_quantized","ball_experts"])]:
        collections={name:[] for name in ["stock",*names]}
        models={}
        for name in names:
            m=Model(name).eval();m.load_state_dict(torch.load(args.runs/f"{name}.pt",weights_only=True));models[name]=m
        with torch.inference_mode():
            for path in paths:
                data=Stream(path,ball,1 if ball else 8)
                for start in range(0,len(data.points),1024):
                    batch=data.batch(data.points[start:start+1024],ball)
                    t=torch.from_numpy(batch[4])
                    collections["stock"].append(error(torch.from_numpy(batch[3]),t,ball).numpy())
                    for name,m in models.items():
                        v,_=predict(m,batch)
                        if not ball:
                            # Preserve a reported body direction, as in live apply.
                            raw=batch[-1][:,3]
                            v[raw,2:]=torch.from_numpy(batch[3][raw,2:])
                        collections[name].append(error(v,t,ball).numpy())
        for name,values in collections.items():
            a=np.concatenate(values)
            scores["ball" if ball else "player"][name]={"n":len(a),"mean":a.mean(axis=0,dtype=np.float64).tolist(),"p90":np.quantile(a,.9,axis=0).tolist()}
    args.output.write_text(json.dumps({"games":[g.name for g in chosen],"metrics":scores},indent=2)+"\n")
    print(json.dumps(scores))
