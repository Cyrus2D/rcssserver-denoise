#!/usr/bin/env python3
"""Three matched position-only experiments; select epochs without test data."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from data import Data,batches
from model import PositionModel,loss,point_output

def validate(model,data,indices,history,device,batch_size=1024):
    model.eval();total={'mean':0.,'map':0.};n=0
    with torch.inference_mode():
        for idx in batches(indices,batch_size):
            x,u,y=data.batch(idx,history,device);out=model(x,u)
            for rule in total:
                total[rule]+=float((point_output(out,rule)-y).norm(dim=-1).sum())
            n+=len(idx)
    return {k:v/n for k,v in total.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--steps',type=int,default=800)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--history',type=int,default=32)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--seed',type=int,default=50501)
    p.add_argument('--device',default='auto');p.add_argument('--threads',type=int,default=8)
    p.add_argument('--kinds',nargs='+',default=['point','gaussian','mixture']);p.add_argument('--validation-samples',type=int,default=200000)
    args=p.parse_args();torch.set_num_threads(args.threads)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    args.output.mkdir(parents=True,exist_ok=True)
    data=Data(args.dataset,['train','validation'])
    rng=np.random.default_rng(args.seed)
    candidates=data.split_indices['selection'];valid=np.sort(rng.choice(candidates,min(args.validation_samples,len(candidates)),replace=False))
    np.save(args.output/'selection_indices.npy',valid)
    from torch.utils.tensorboard import SummaryWriter
    records=[]
    for kind in args.kinds:
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if device=='cuda':torch.cuda.manual_seed_all(args.seed)
        generator=np.random.default_rng(args.seed+1)
        model=PositionModel(len(data.feature_names),kind,args.hidden).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-4)
        run=args.output/kind;run.mkdir(exist_ok=True);writer=SummaryWriter(str(run/'tensorboard'))
        config=dict(vars(args));config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
        config.update(kind=kind,device=device,parameters=sum(p.numel() for p in model.parameters()),features=data.feature_names,
                      selection_games=data.selection_games,calibration_games=data.calibration_games,
                      train_games=sorted(g for g,s in data.manifest['split'].items() if s=='train'),
                      train_eligible=len(data.split_indices['train']),selection_samples=len(valid),
                      loss='mean Euclidean distance (smoothed 0.01m)' if kind=='point' else 'mixture/Gaussian NLL + 0.2 * mean Euclidean distance',
                      max_gap=30,output='displacement from last confirmed named sighting; CV prior plus learned residual')
        (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        best=float('inf');logs=[];start=time.monotonic()
        for epoch in range(1,args.epochs+1):
            model.train();running=0.
            for step in range(args.steps):
                ids=data.split_indices['train'][generator.integers(len(data.split_indices['train']),size=args.batch_size)]
                x,u,y=data.batch(ids,args.history,device);optimizer.zero_grad(set_to_none=True)
                out=model(x,u);value=loss(out,y)
                if not torch.isfinite(value):raise ValueError('Nonfinite training objective')
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();running+=float(value.detach())
            scores=validate(model,data,valid,args.history,device)
            rule=min(scores,key=scores.get) if kind=='mixture' else 'mean';score=scores[rule]
            record=dict(epoch=epoch,loss=running/args.steps,validation=scores,readout=rule,elapsed_seconds=time.monotonic()-start)
            logs.append(record);(run/'training.json').write_text(json.dumps(logs,indent=2)+'\n')
            writer.add_scalar('train/loss',record['loss'],epoch);writer.add_scalar('validation/position_error_cm',score*100,epoch)
            writer.add_scalar('validation/wm_error_cm',float(np.linalg.norm(data.wm[valid]-data.y[valid],axis=1).mean()*100),epoch)
            if score<best:
                best=score;torch.save(dict(state=model.state_dict(),config=config,epoch=epoch,readout=rule,validation=score),run/'best.pt')
            schedule.step()
            print(f'{kind} epoch {epoch}/{args.epochs}: position {score*100:.3f} cm, mean/map {scores}, loss {record["loss"]:.4f}, elapsed {record["elapsed_seconds"]:.1f}s',flush=True)
        writer.close();records.append(dict(kind=kind,best_validation_cm=best*100,parameters=config['parameters']))
    (args.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
