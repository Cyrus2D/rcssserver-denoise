#!/usr/bin/env python3
"""Exploratory fixed-budget width/depth/data/training-duration study."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data import Data,batches
from model import PositionModel,loss

VARIANTS=[dict(name='continued64',hidden=64,layers=1,games=160,continue_baseline=True,lr=3e-4),
          dict(name='wide128',hidden=128,layers=1,games=160,continue_baseline=False,lr=1e-3),
          dict(name='deep64x2',hidden=64,layers=2,games=160,continue_baseline=False,lr=1e-3),
          dict(name='data40',hidden=64,layers=1,games=40,continue_baseline=False,lr=1e-3)]

def create(features,hidden=64,layers=1):
    model=PositionModel(features,'point',hidden)
    if layers>1:model.gru=nn.GRU(hidden,hidden,num_layers=layers,batch_first=True)
    return model

def infer(model,data,indices,device,batch=4096):
    result=np.empty(len(indices),np.float32);offset=0;model.eval()
    with torch.inference_mode():
        for idx in batches(indices,batch):
            x,u,y=data.batch(idx,32,device);e=(model(x,u)['point']-y).norm(dim=-1).cpu().numpy()
            result[offset:offset+len(idx)]=e;offset+=len(idx)
    return result

def summary(error,baseline):
    e=np.asarray(error,float);b=np.asarray(baseline,float);change=e-b
    result=dict(n=len(e),mean_m=float(e.mean()),baseline_mean_m=float(b.mean()),gain_pct=float(100*(b-e).mean()/b.mean()),
                p95_m=float(np.quantile(e,.95)),p99_m=float(np.quantile(e,.99)),max_error_m=float(e.max()),
                max_added_error_m=float(max(0,change.max())))
    for cm in [0,1,5,10,25,50,100]:result['worse_over_'+str(cm)+'cm_pct']=float(100*(change>(1e-5 if cm==0 else cm/100)).mean())
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path);parser.add_argument('baseline',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--steps',type=int,default=800);parser.add_argument('--test-samples',type=int,default=500000)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8);device='cuda' if torch.cuda.is_available() else 'cpu'
    protocol=dict(seed=50501,history=32,epochs=args.epochs,steps=args.steps,batch_size=256,variants=VARIANTS,
                  endpoint_budget=args.epochs*args.steps*256,selection_samples=200000,train_probe_samples=50000,test_samples=args.test_samples,
                  caveat='Exploratory reuse of the previously inspected 20 test games. Single seed. Hyperparameters fixed before this study test scoring; checkpoints selected only on selection10.',
                  dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest())
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    data=Data(args.dataset,['train','validation']);rng=np.random.default_rng(50501)
    val=np.sort(rng.choice(data.split_indices['selection'],min(200000,len(data.split_indices['selection'])),replace=False))
    trainpool=data.split_indices['train'];probe=np.sort(np.random.default_rng(50512).choice(trainpool,50000,replace=False))
    np.save(args.output/'selection_indices.npy',val);np.save(args.output/'train_probe_indices.npy',probe)
    checkpoint=torch.load(args.baseline,map_location='cpu',weights_only=False)
    baseline=create(len(data.feature_names)).to(device);baseline.load_state_dict(checkpoint['state'])
    records={'baseline64':dict(name='baseline64',hidden=64,layers=1,games=160,parameters=sum(p.numel() for p in baseline.parameters()),
                              epoch=checkpoint['epoch'],train_probe_mean_m=float(infer(baseline,data,probe,device).mean()),
                              selection_mean_m=float(infer(baseline,data,val,device).mean()),checkpoint=str(args.baseline.resolve()),
                              endpoint_budget=2048000,training_seconds=97.7718)}
    print('Baseline diagnostic',records['baseline64'],flush=True);del baseline
    train_names=sorted(g for g,s in data.manifest['split'].items() if s=='train');small=np.random.default_rng(50511).choice(train_names,40,replace=False).tolist()
    small_game_ids=np.array([data.games.index(g) for g in small]);smallpool=trainpool[np.isin(data.game[trainpool],small_game_ids)]
    (args.output/'small_training_games.json').write_text(json.dumps(sorted(small),indent=2)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    for variant in VARIANTS:
        run=args.output/variant['name'];run.mkdir(exist_ok=True)
        if (run/'training.json').exists() and len(json.loads((run/'training.json').read_text()))==args.epochs:
            print('Reusing finished training',variant['name'],flush=True);saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False);records[variant['name']]=saved['record'];continue
        torch.manual_seed(50501);torch.cuda.manual_seed_all(50501)
        model=create(len(data.feature_names),variant['hidden'],variant['layers']).to(device)
        if variant['continue_baseline']:model.load_state_dict(checkpoint['state'])
        pool=smallpool if variant['games']==40 else trainpool;generator=np.random.default_rng(50502 if not variant['continue_baseline'] else 50503)
        optimizer=torch.optim.AdamW(model.parameters(),lr=variant['lr'],weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=5e-5 if variant['continue_baseline'] else 1e-4)
        writer=SummaryWriter(str(run/'tensorboard'));best=float('inf');logs=[];start=time.monotonic()
        record=dict(variant,parameters=sum(p.numel() for p in model.parameters()),checkpoint=str((run/'best.pt').resolve()),
                    endpoint_budget=args.epochs*args.steps*256+(2048000 if variant['continue_baseline'] else 0),
                    training_games=sorted(small) if variant['games']==40 else train_names)
        for epoch in range(1,args.epochs+1):
            model.train();total=0.
            for step in range(args.steps):
                ids=pool[generator.integers(len(pool),size=256)];x,u,y=data.batch(ids,32,device)
                optimizer.zero_grad(set_to_none=True);value=loss(model(x,u),y)
                if not torch.isfinite(value):raise ValueError('Nonfinite training')
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(value.detach())
            train_error=float(infer(model,data,probe,device).mean());val_error=float(infer(model,data,val,device).mean())
            row=dict(epoch=epoch,loss=total/args.steps,train_probe_mean_m=train_error,selection_mean_m=val_error,elapsed_seconds=time.monotonic()-start)
            logs.append(row);(run/'training.json').write_text(json.dumps(logs,indent=2)+'\n')
            writer.add_scalar('train/probe_error_cm',train_error*100,epoch);writer.add_scalar('selection/position_error_cm',val_error*100,epoch)
            if val_error<best:
                best=val_error;record.update(epoch=epoch,train_probe_mean_m=train_error,selection_mean_m=val_error,
                                            training_seconds=row['elapsed_seconds'])
                torch.save(dict(state=model.state_dict(),record=record,features=len(data.feature_names)),run/'best.pt')
            scheduler.step();print(variant['name'],epoch,row,flush=True)
        writer.close();record['total_training_seconds']=time.monotonic()-start
        # Persist the final resource duration separately from selected-epoch duration.
        (run/'record.json').write_text(json.dumps(record,indent=2)+'\n');records[variant['name']]=record
        del model;torch.cuda.empty_cache()
        (args.output/'training_summary.json').write_text(json.dumps(records,indent=2)+'\n')
    del data;gc.collect();torch.cuda.empty_cache()
    print('Loading held-out data after all training and checkpoint choices are frozen.',flush=True)
    data=Data(args.dataset,['validation','test']);pool=data.split_indices['test']
    positions=np.sort(np.random.default_rng(50513).choice(len(pool),min(args.test_samples,len(pool)),replace=False));ids=pool[positions]
    np.save(args.output/'test_positions.npy',positions);np.save(args.output/'test_indices.npy',ids)
    baseline_error=np.linalg.norm(data.wm[ids]-data.y[ids],axis=1)
    np.savez(args.output/'test_labels.npz',baseline_error=baseline_error,age=data.age[ids],flags=data.flags[ids],count=data.count[ids],game=data.game[ids])
    results={}
    for name,record in records.items():
        if name=='baseline64':saved=checkpoint;model=create(len(data.feature_names)).to(device)
        else:
            saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False);model=create(len(data.feature_names),record['hidden'],record['layers']).to(device)
        model.load_state_dict(saved['state']);start=time.monotonic();error=infer(model,data,ids,device)
        np.save(args.output/(name+'-test-error.npy'),error)
        result=dict(record,test=summary(error,baseline_error),test_inference_seconds=time.monotonic()-start)
        result['already_accurate_stock']=summary(error[baseline_error<.1],baseline_error[baseline_error<.1])
        result['by_age']={str(low)+'-'+str(high):summary(error[(data.age[ids]>=low)&(data.age[ids]<=high)],baseline_error[(data.age[ids]>=low)&(data.age[ids]<=high)]) for low,high in [(1,1),(2,2),(3,5),(6,10),(11,20),(21,30)]}
        results[name]=result;print('TEST',name,result['test'],flush=True);del model
        (args.output/'results.json').write_text(json.dumps(dict(protocol=protocol,models=results,test_games=sorted(g for g,s in data.manifest['split'].items() if s=='test')),indent=2)+'\n')
    best_name=min(records,key=lambda k:records[k]['selection_mean_m'])
    print('Strongest validation candidate',best_name,flush=True)

if __name__=='__main__':main()
