#!/usr/bin/env python3
"""Paired benefit/harm, uncertainty calibration, and fixed conservative policy."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from data import Data,batches
from model import PositionModel,point_output,covariance,mahalanobis,log_prob

AGES=['1','2','3','4','5','6-10','11-20','21-30']
def age_groups(age):
    return np.select([age==1,age==2,age==3,age==4,age==5,age<=10,age<=20],np.arange(7),default=7)

def predict(model,data,ids,history,device,rule,batch_size):
    n=len(ids);prob=model.kind!='point'
    result=dict(pred=np.empty((n,2),np.float32),error=np.empty(n,np.float32))
    if prob:
        result.update(cov=np.empty((n,3),np.float32),center=np.empty((n,2),np.float32),md=np.empty(n,np.float32),nll=np.empty(n,np.float32))
        if model.kind=='mixture':
            for k,d in [('means',6),('scale',6),('rho',3),('weight',3)]:result[k]=np.empty((n,d),np.float32)
    offset=0;start=time.monotonic();model.eval()
    with torch.inference_mode():
        for batchnum,idx in enumerate(batches(ids,batch_size),1):
            x,u,y=data.batch(idx,history,device);out=model(x,u);pred=point_output(out,rule);sl=slice(offset,offset+len(idx))
            result['pred'][sl]=pred.cpu().numpy();result['error'][sl]=(pred-y).norm(dim=-1).cpu().numpy()
            if prob:
                result['cov'][sl]=covariance(out).cpu().numpy();result['center'][sl]=out['point'].cpu().numpy()
                result['md'][sl]=mahalanobis(out,y).cpu().numpy();result['nll'][sl]=(-log_prob(out,y)).cpu().numpy()
                if model.kind=='mixture':
                    for key in ['means','scale','rho','weight']:result[key][sl]=out[key].reshape(len(idx),-1).cpu().numpy()
            offset+=len(idx)
            if batchnum%500==0:print(f'{model.kind}: inferred {offset:,}/{n:,} rows in {time.monotonic()-start:.0f}s',flush=True)
    return result

def metrics(error,baseline):
    n=len(error)
    if not n:return None
    delta=baseline-error;helped=delta>.01;hurt=delta<-.01
    return dict(n=n,mean=float(error.mean()),baseline_mean=float(baseline.mean()),gain=float(delta.mean()),
                gain_pct=float(100*delta.mean()/baseline.mean()),rmse=float(np.sqrt(np.mean(error.astype(float)**2))),
                median=float(np.median(error)),p90=float(np.quantile(error,.9)),p95=float(np.quantile(error,.95)),
                p99=float(np.quantile(error,.99)),max=float(error.max()),help_pct=float(100*helped.mean()),hurt_pct=float(100*hurt.mean()),
                tie_pct=float(100*(~(helped|hurt)).mean()),mean_help=float(delta[helped].mean()) if helped.any() else 0.,
                mean_harm=float(-delta[hurt].mean()) if hurt.any() else 0.,p95_harm=float(np.quantile(-delta[hurt],.95)) if hurt.any() else 0.,
                max_harm=float(max(0,-delta.min())),harm_over_25cm_pct=float(100*(delta<-.25).mean()),
                harm_over_50cm_pct=float(100*(delta<-.5).mean()),harm_over_1m_pct=float(100*(delta<-1).mean()),
                error_over_1m_pct=float(100*(error>1).mean()),error_over_2m_pct=float(100*(error>2).mean()))

def group_masks(data,ids,base):
    age=data.age[ids];groups=age_groups(age);x=data.x[ids];wmcount=data.count[ids];dist=data.distance[ids]
    masks={'all':np.ones(len(ids),bool),'strict_unseen':(data.flags[ids]&8)==0,'anonymous_current':(data.flags[ids]&8)!=0,
           'teammate':x[:,37]>.5,'opponent':x[:,37]<=.5,'unnamed_wm_refresh':x[:,15]>.5}
    for i,label in enumerate(AGES):masks['gap/'+label]=groups==i
    for i in range(6):masks['position_count/'+str(i)]=wmcount==i
    masks['position_count/>5']=wmcount>5
    for label,width in [('60',60),('120',120),('180',180)]:masks['view/'+label]=np.isclose(x[:,28]*180,width,atol=1)
    for label,low,high in [('0-10',0,10),('10-20',10,20),('20-40',20,40),('40+',40,np.inf)]:masks['distance/'+label]=(dist>=low)&(dist<high)
    for label,lo,hi in [('0-10cm',0,.1),('10-25cm',.1,.25),('25-50cm',.25,.5),('50-100cm',.5,1),('100cm+',1,np.inf)]:masks['baseline_error/'+label]=(base>=lo)&(base<hi)
    ball=x[:,30:32]*30;center=x[:,2:4]*1.5*age[:,None]
    masks['estimated_near_ball_2m']=(x[:,34]>.5)&(np.linalg.norm(ball-center,axis=1)<2)
    return masks

def calibrate(data,ids,pred):
    group=age_groups(data.age[ids]);levels=[.5,.8,.9,.95];result={}
    for i,label in enumerate(AGES):
        mask=group==i;v=pred['md'][mask] if mask.sum()>=2000 else pred['md']
        result[label]={str(level):float(np.quantile(v,level)) for level in levels}
        result[label]['n']=int(mask.sum());result[label]['pooled_fallback']=bool(mask.sum()<2000)
    return result

def radii(data,ids,pred,calibration,level=.9):
    group=age_groups(data.age[ids]);q=np.array([calibration[a][str(level)] for a in AGES])[group]
    cov=pred['cov'];det=np.maximum(cov[:,0]*cov[:,1]-cov[:,2]**2,1e-12)
    area=np.pi*q*np.sqrt(det)
    return q,area,np.sqrt(area/np.pi)

def select_gate(data,ids,pred,baseline,calibration):
    difference=pred['pred']-data.wm[ids];norm=np.linalg.norm(difference,axis=1)
    capped=data.wm[ids]+difference*np.minimum(1,.25/np.maximum(norm,1e-8))[:,None]
    candidate=np.linalg.norm(capped-data.y[ids],axis=1)
    group=age_groups(data.age[ids]);r=radii(data,ids,pred,calibration)[2] if calibration else np.zeros(len(ids))
    thresholds={}
    for i,label in enumerate(AGES):
        mask=group==i
        if not mask.any():thresholds[label]=-1.;continue
        choices=[-1.,.1,.25,.5,1.,2.,4.,1e9] if calibration else [-1.,1e9]
        scores=[float(np.where(r[mask]<=threshold,candidate[mask],baseline[mask]).mean()) for threshold in choices]
        thresholds[label]=choices[int(np.argmin(scores))]
    return thresholds

def apply_gate(data,ids,pred,gate,calibration):
    difference=pred['pred']-data.wm[ids];norm=np.linalg.norm(difference,axis=1)
    capped=data.wm[ids]+difference*np.minimum(1,.25/np.maximum(norm,1e-8))[:,None]
    group=age_groups(data.age[ids]);r=radii(data,ids,pred,calibration)[2] if calibration else np.zeros(len(ids))
    threshold=np.array([gate[a] for a in AGES])[group];accept=r<=threshold
    return np.where(accept[:,None],capped,data.wm[ids]),accept

def probability_metrics(data,ids,pred,calibration):
    result={};group=age_groups(data.age[ids])
    for level in [.5,.8,.9,.95]:
        q,area,r=radii(data,ids,pred,calibration,level)
        result[str(level)]=dict(coverage=float((pred['md']<=q).mean()),mean_area_m2=float(area.mean()),mean_equivalent_radius_m=float(r.mean()))
    q,area,r=radii(data,ids,pred,calibration)
    result['by_age']={a:dict(n=int((group==i).sum()),coverage=float((pred['md'][group==i]<=q[group==i]).mean()),
                                   mean_radius=float(r[group==i].mean())) for i,a in enumerate(AGES) if (group==i).any()}
    result['nll_mean']=float(pred['nll'].mean())
    ranks=np.argsort(r);result['uncertainty_ranking']={}
    for fraction in [.1,.25,.5,1.]:
        idx=ranks[:max(1,int(len(ranks)*fraction))]
        result['uncertainty_ranking'][str(fraction)]=dict(mean_error=float(pred['error'][idx].mean()),mean_radius=float(r[idx].mean()))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('runs',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=4096);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--calibration-samples',type=int,default=500000);p.add_argument('--selection-samples',type=int,default=500000)
    args=p.parse_args();torch.set_num_threads(args.threads);args.output.mkdir(parents=True,exist_ok=True)
    signature=dict(dataset=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
                   checkpoints={k:hashlib.sha256((args.runs/k/'best.pt').read_bytes()).hexdigest() for k in ['point','gaussian','mixture']},
                   code={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['evaluate.py','model.py','data.py']},
                   calibration_samples=args.calibration_samples,selection_samples=args.selection_samples,seed=50502)
    signature_file=args.output/'signature.json'
    if signature_file.exists() and json.loads(signature_file.read_text())!=signature:
        raise ValueError('Weights, data, code or evaluation configuration changed: use a new output directory.')
    signature_file.write_text(json.dumps(signature,indent=2)+'\n')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    data=Data(args.dataset,['validation','test']);rng=np.random.default_rng(50502)
    ids={key:np.sort(rng.choice(data.split_indices[key],min(n,len(data.split_indices[key])),replace=False)) for key,n in [('selection',args.selection_samples),('calibration',args.calibration_samples)]}
    ids['test']=data.split_indices['test']
    baseline={key:np.linalg.norm(data.wm[v]-data.y[v],axis=1) for key,v in ids.items()}
    test=ids['test'];masks=group_masks(data,test,baseline['test']);summary={}
    decay=float(data.manifest.get('server_default_player_decay',.4))
    age=data.age[test].astype(float);velocity=data.x[test,2:4]*1.5
    physics=dict(wm=baseline['test'],hold=np.linalg.norm(data.y[test],axis=1),
                 constant_velocity=np.linalg.norm(velocity*age[:,None]-data.y[test],axis=1),
                 decay=np.linalg.norm(velocity*((1-decay**age)/(1-decay))[:,None]-data.y[test],axis=1))
    baseline_tables={name:{k:metrics(error[mask],baseline['test'][mask]) for k,mask in masks.items() if mask.any()} for name,error in physics.items()}
    # Persist shared labels once; probabilities never select a hypothesis using truth.
    np.savez(args.output/'cohort.npz',indices=test,age=data.age[test],count=data.count[test],flags=data.flags[test],
             baseline=baseline['test'],truth=data.y[test],wm=data.wm[test],game=data.game[test],distance=data.distance[test],
             constant_velocity_error=physics['constant_velocity'],hold_error=physics['hold'],decay_error=physics['decay'])
    for kind in ['point','gaussian','mixture']:
        checkpoint=torch.load(args.runs/kind/'best.pt',map_location='cpu',weights_only=False);config=checkpoint['config']
        model=PositionModel(len(data.feature_names),kind,config['hidden']).to(device);model.load_state_dict(checkpoint['state'])
        preds={}
        for split in ['calibration','selection','test']:
            cache=args.output/(kind+'-'+split+'.npz')
            if cache.exists():
                loaded=np.load(cache);preds[split]={k:loaded[k] for k in loaded.files}
                if len(preds[split]['error'])!=len(ids[split]):raise ValueError('Cached cohort mismatch')
            else:
                preds[split]=predict(model,data,ids[split],config['history'],device,checkpoint['readout'],args.batch_size)
                np.savez(cache,**preds[split])
            print(kind,split,'mean cm',float(preds[split]['error'].mean()*100),flush=True)
        cal=calibrate(data,ids['calibration'],preds['calibration']) if kind!='point' else None
        gate=select_gate(data,ids['selection'],preds['selection'],baseline['selection'],cal)
        guarded,accept=apply_gate(data,test,preds['test'],gate,cal)
        errors=dict(raw=preds['test']['error'],guarded=np.linalg.norm(guarded-data.y[test],axis=1))
        np.save(args.output/(kind+'-guarded-error.npy'),errors['guarded'])
        record=dict(parameters=config['parameters'],epoch=checkpoint['epoch'],readout=checkpoint['readout'],calibration=cal,gate=gate,
                    versus_constant_velocity=metrics(errors['raw'],physics['constant_velocity']),
                    guard_accept_pct=float(100*accept.mean()),groups={},probability=probability_metrics(data,test,preds['test'],cal) if cal else None)
        for policy,error in errors.items():record['groups'][policy]={k:metrics(error[mask],baseline['test'][mask]) for k,mask in masks.items() if mask.any()}
        record['per_game']={data.games[int(g)]:metrics(errors['raw'][data.game[test]==g],baseline['test'][data.game[test]==g]) for g in np.unique(data.game[test])}
        summary[kind]=record
        (args.output/'metrics.json').write_text(json.dumps(dict(models=summary,baselines=baseline_tables,test_samples=len(test),
            train_games=sorted(g for g,s in data.manifest['split'].items() if s=='train'),selection_games=data.selection_games,
            calibration_games=data.calibration_games,test_games=sorted(g for g,s in data.manifest['split'].items() if s=='test'),
            cohort='Named history within uninterrupted PlayOn; no current identified sighting; last identified position 1-30 cycles old; valid known-number WM position and same-cycle truth. Unknown proximity-matched tracks excluded. Equal weight per paired target-cycle.',
            caveat='Offline position-error benefit/harm, not measured action quality or game outcomes. One training seed. Covariance ellipses are calibrated total-mixture moment ellipses, not separate confidence guarantees for individual components.',
            device=device),indent=2)+'\n')
        del preds;print('Completed',kind,flush=True)

if __name__=='__main__':main()
