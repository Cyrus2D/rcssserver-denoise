"""Exp5-S1 refinement: frozen caches, harm-aware training and seed repeats."""
from pathlib import Path
import argparse,json,time,hashlib,gc
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from data import Data
from scaling_study import create
from model import PositionModel

SEEDS=[50511,50521]
VARIANTS=[dict(name='continue128',seed=50501,penalty=0.,hard_weight=0.,fresh=False),
          dict(name='harm128_l1',seed=50501,penalty=1.,hard_weight=0.,fresh=False),
          dict(name='harm128_l3',seed=50501,penalty=3.,hard_weight=0.,fresh=False),
          dict(name='hard128_l1',seed=50501,penalty=1.,hard_weight=3.,fresh=False)]+[
          dict(name=f'seed128_{s}',seed=s,penalty=0.,hard_weight=0.,fresh=True) for s in SEEDS]
VARIANTS += [dict(name='wmresidual128_l1',seed=50501,penalty=1.,hard_weight=0.,fresh=False,wm_residual=True)]

class WMResidualModel(PositionModel):
    """Learn a correction around WM; zero head exactly preserves current WM."""
    def __init__(self):super().__init__(51,'point',128)
    def forward(self,x,unum):
        result=super().forward(x,unum)
        result['point']=result['point']+x[:,-1,16:18]*10
        return result

def atomic_json(path,d):
    p=Path(path);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)

def inference(model,data,ids,device,batch=4096):
    result=np.empty((len(ids),2),np.float32);model.eval()
    with torch.inference_mode():
        for start in range(0,len(ids),batch):
            idx=ids[start:start+batch];x,u,_=data.batch(idx,32,device)
            result[start:start+len(idx)]=model(x,u)['point'].cpu().numpy()
    return result

def get_model(file,device):
    saved=torch.load(file,map_location='cpu',weights_only=False)
    model=(WMResidualModel() if saved.get('record',{}).get('wm_residual') else create(51,128)).to(device)
    model.load_state_dict(saved['state']);model.eval();return model

def prepare(args):
    data=Data(args.dataset,['validation','test']);rng=np.random.default_rng(50502)
    ids={k:np.sort(rng.choice(data.split_indices[k],500000,replace=False)) for k in ['selection','calibration']}
    old=np.load(args.logs/'selector-study/indices.npz')
    ids['test']=old['test_dataset_indices']
    np.save(args.out/'test_positions.npy',old['test_cache_positions'])
    manifest=dict(split={k:sorted(data.games[i] for i in np.unique(data.game[v])) for k,v in ids.items()},
                  dataset_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
                  note='Old inspected test20 reused exploratorily. True distance is a report filter only; runtime gates use WM-estimated distance.',
                  variants=VARIANTS,fit_samples=250000,history=32,epochs=10,steps=800,batch=256)
    manifest['confirmation_protocol']=dict(designated_practical_objective='near50_risk',fresh_games=10,
       rule='Freeze all policies before new-game prediction scoring; no new-game tuning.',
       fixed_safety_sensitivity=dict(policy='near50_risk',extra_cap_m=[1,2,4]))
    old=args.out/'protocol.json'
    if old.exists() and json.loads(old.read_text())['dataset_sha256']!=manifest['dataset_sha256']:
        raise ValueError('Dataset changed: use a new refinement output directory')
    atomic_json(args.out/'study-requirements.json',dict(models=['wide128']+[v['name'] for v in VARIANTS]))
    atomic_json(args.out/'protocol.json',manifest)
    model=get_model(args.logs/'scaling-study/wide128/best.pt',args.device)
    for split,idx in ids.items():
        np.savez(args.out/(split+'-data.npz'),indices=idx,x=data.x[idx],y=data.y[idx],wm=data.wm[idx],
                 age=data.age[idx],count=data.count[idx],unum=data.unum[idx],distance=data.distance[idx],
                 game=data.game[idx],flags=data.flags[idx])
        pred=inference(model,data,idx,args.device);np.save(args.out/('wide128-'+split+'-pred.npy'),pred)
        print('Prepared',split,len(idx),'wide128 mean',np.linalg.norm(pred-data.y[idx],axis=1).mean(),flush=True)

def train(args):
    data=Data(args.dataset,['train','validation']);pool=data.split_indices['train']
    val=np.sort(np.random.default_rng(50531).choice(data.split_indices['selection'],100000,replace=False))
    vy=data.y[val];vb=np.linalg.norm(data.wm[val]-vy,axis=1)
    base=torch.load(args.logs/'scaling-study/wide128/best.pt',map_location='cpu',weights_only=False)
    records={};startall=time.monotonic()
    for variant in VARIANTS:
        run=args.out/variant['name'];run.mkdir(exist_ok=True)
        if (run/'record.json').exists():
            records[variant['name']]=json.loads((run/'record.json').read_text());print('Reuse',variant['name'],flush=True);continue
        torch.manual_seed(variant['seed']);torch.cuda.manual_seed_all(variant['seed'])
        model=(WMResidualModel() if variant.get('wm_residual') else create(51,128)).to(args.device)
        if not variant['fresh']:model.load_state_dict(base['state'])
        if variant.get('wm_residual'):
            # Reuse the encoder but initialize the new WM-centered readout to no correction.
            torch.nn.init.zeros_(model.head.weight);torch.nn.init.zeros_(model.head.bias)
        lr=1e-3 if variant['fresh'] else 3e-4
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,10,eta_min=1e-4 if variant['fresh'] else 5e-5)
        # All fine-tuning ablations share identical endpoints; new seeds vary initialization and samples.
        rng=np.random.default_rng(variant['seed']+100 if variant['fresh'] else 50532)
        writer=SummaryWriter(str(run/'tensorboard'));rows=[];best=float('inf');started=time.monotonic()
        for epoch in range(1,11):
            model.train();total=0.
            for step in range(800):
                ix=pool[rng.integers(len(pool),size=256)];x,u,y=data.batch(ix,32,args.device)
                wm=torch.as_tensor(data.wm[ix],device=args.device)
                error=(model(x,u)['point']-y).square().sum(-1).add(1e-4).sqrt()
                stock=(wm-y).square().sum(-1).add(1e-4).sqrt()
                value=error+variant['penalty']*torch.relu(error-stock)
                weight=1+variant['hard_weight']*(stock<.25).float()
                objective=(value*weight).sum()/weight.sum()
                if not torch.isfinite(objective):raise ValueError('Nonfinite objective')
                optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(objective.detach())
            p=inference(model,data,val,args.device);e=np.linalg.norm(p-vy,axis=1)
            criterion=float((e+variant['penalty']*np.maximum(e-vb,0)).mean())
            row=dict(epoch=epoch,loss=total/800,selection_mean=float(e.mean()),selection_objective=criterion,
                     worse25_pct=float(100*(e-vb>.25).mean()),seconds=time.monotonic()-started)
            rows.append(row);atomic_json(run/'training.json',rows)
            for key in ['selection_mean','selection_objective','worse25_pct','loss']:writer.add_scalar(key,row[key],epoch)
            if criterion<best:
                best=criterion;record=dict(variant,epoch=epoch,selection_objective=criterion,selection_mean=float(e.mean()),
                    checkpoint=str((run/'best.pt').resolve()),parameters=sum(p.numel() for p in model.parameters()),
                    sampled_endpoints=2048000,base_endpoints=0 if variant['fresh'] else 2048000,
                    base_checkpoint=None if variant['fresh'] else str(args.logs/'scaling-study/wide128/best.pt'))
                torch.save(dict(state=model.state_dict(),record=record,features=51),run/'best.pt')
            scheduler.step();print(variant['name'],row,flush=True)
        writer.close();record['training_seconds']=time.monotonic()-started;records[variant['name']]=record
        atomic_json(run/'record.json',record);atomic_json(args.out/'training-results.json',records)
        del model;torch.cuda.empty_cache()
    print('Training complete',time.monotonic()-startall,flush=True)

def cache(args):
    data=Data(args.dataset,['validation','test'])
    records=json.loads((args.out/'training-results.json').read_text())
    for name,r in records.items():
        model=get_model(r['checkpoint'],args.device)
        for split in ['selection','calibration','test']:
            dest=args.out/(name+'-'+split+'-pred.npy')
            if dest.exists():continue
            idx=np.load(args.out/(split+'-data.npz'))['indices'];p=inference(model,data,idx,args.device)
            np.save(dest,p);print('Cached',name,split,len(p),flush=True)
        atomic_json(args.out/('cache-'+name+'.json'),dict(complete=True,checkpoint=r['checkpoint']))
        del model;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','cache']);p.add_argument('dataset',type=Path);p.add_argument('logs',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);a.device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(8)
    globals()[a.action](a)
