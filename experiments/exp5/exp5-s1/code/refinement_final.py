"""Freeze validation-selected policies, then evaluate old/fresh full cohorts."""
from pathlib import Path
import argparse,json,pickle,hashlib,time
import numpy as np
import torch
from data import Data
from refinement_models import atomic_json,get_model
from refinement_select import features,corrected,groups,summary,score_model
from scaling_study import create
from concurrent.futures import ThreadPoolExecutor

def freeze(out):
    routing=json.loads((out/'routing-results.json').read_text());blend=json.loads((out/'blend-results.json').read_text())
    policies={}
    for mode,v in blend['winners'].items():
        r=routing['winners'].get(mode)
        if r and r['tuning']['utility']>=v['tuning']['utility']:policies[mode]=dict(r,alpha=1.,cap_m=1e9,cap_per_cycle=None,source='routing')
        else:policies[mode]=dict(v,source='blend')
    clean={k:{key:v[key] for key in ['model','candidate','alpha','cap_m','cap_per_cycle','tuning','source']} for k,v in policies.items()}
    clean['provenance']=dict(choice='Original calibration10 only. No test metrics choose policy.',
        checkpoint_sha256={name:hashlib.sha256(checkpoint(out,name).read_bytes()).hexdigest() for name in set(v['model'] for v in policies.values())},
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [out/'routing-results.json',out/'blend-results.json']})
    dest=out/'frozen-policies.json'
    if dest.exists() and json.loads(dest.read_text())!=clean:raise ValueError('Frozen policies changed. Use new output directory.')
    atomic_json(dest,clean);print('Policies frozen',[(k,v['model'],v['candidate'],v['alpha'],v['cap_m']) for k,v in policies.items()],flush=True)

def checkpoint(out,name):
    return out.parent/'scaling-study/wide128/best.pt' if name=='wide128' else out/name/'best.pt'

def accepted(config,bank,d,p):
    basic,strong=features(d,p);key=config['candidate']
    if key=='always':a=np.ones(len(p),bool)
    elif key in ['count','count_speed','count_speed_correction']:
        r=config['tuning'];a=(basic[:,0]>r['count'])&(basic[:,2]>=r['speed'])&(basic[:,7]<=r['correction'])
    else:
        m=bank[key];z=strong if m['strong'] else basic;score=score_model(m['model'],z,m['classification'])
        a=score>config['tuning']['threshold']
    return a&(d['count']>0)

def metadata(data,ids):
    return {k:getattr(data,k)[ids] for k in ['x','wm','y','age','count','unum','distance','game','flags']}

def bootstrap(gain,bins,seed=50650):
    codes,group=np.unique(bins,return_inverse=True);counts=np.bincount(group);sums=np.bincount(group,weights=gain)
    pick=np.random.default_rng(seed).integers(len(codes),size=(3000,len(codes)))
    means=sums[pick].sum(1)/counts[pick].sum(1)
    return dict(games=len(codes),gain_mean=float(gain.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),method='3000 paired whole-game bootstrap draws; ratio of summed gains to summed sample counts')

def evaluate(a,fresh=False):
    frozen=json.loads((a.out/'frozen-policies.json').read_text());configs={k:v for k,v in frozen.items() if k!='provenance'}
    data=Data(a.fresh_dataset if fresh else a.dataset,['test'] if fresh else ['validation','test'])
    ids=data.split_indices['test'];label='fresh' if fresh else 'test';output=a.out/('fresh-evaluation' if fresh else 'full-evaluation');output.mkdir(exist_ok=True)
    if not fresh:
        original_ids=np.load(a.out.parent/'evaluation/cohort.npz')['indices']
        if not np.array_equal(ids,original_ids):raise ValueError('Original full-cohort endpoint order changed')
        del original_ids
    else:
        original_games=json.loads((a.dataset/'manifest.json').read_text())['split']
        if set(data.games)&set(original_games):raise ValueError('Fresh games overlap original game IDs')
    for name,expected in frozen['provenance']['checkpoint_sha256'].items():
        if hashlib.sha256(checkpoint(a.out,name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen checkpoint changed')
    names=sorted(set(v['model'] for v in configs.values())|({'wide128','original64'} if fresh else set()))
    start=time.monotonic()
    signature=dict(n=len(ids),names=names,frozen_sha256=hashlib.sha256((a.out/'frozen-policies.json').read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256(((a.fresh_dataset if fresh else a.dataset)/'manifest.json').read_bytes()).hexdigest())
    marker=output/'prediction-cache.json'
    if marker.exists() and json.loads(marker.read_text())==signature:
        arrays={n:np.load(output/(n+'-pred.npy'),mmap_mode='r') for n in names}
        print('Reusing verified complete predictions',label,flush=True)
    else:
        models={n:get_model(checkpoint(a.out,n),a.device) for n in names if n!='original64'}
        if fresh:
            baseline=create(51,64).to(a.device)
            baseline.load_state_dict(torch.load(a.out.parent/'runs/point/best.pt',map_location='cpu',weights_only=False)['state'])
            baseline.eval();models['original64']=baseline
        progress=output/'prediction-progress.json';done=0
        if progress.exists():
            saved_progress=json.loads(progress.read_text())
            if saved_progress['signature']==signature:done=saved_progress['done']
        arrays={n:np.lib.format.open_memmap(output/(n+'-pred.npy'),mode='r+' if done else 'w+',dtype=np.float32,shape=(len(ids),2)) for n in names}
        for begin in range(done,len(ids),8192):
            ix=ids[begin:begin+8192];x,u,_=data.batch(ix,32,a.device)
            with torch.inference_mode():
                for name,m in models.items():arrays[name][begin:begin+len(ix)]=m(x,u)['point'].cpu().numpy()
            if begin%(8192*250)==0:
                for arr in arrays.values():arr.flush()
                atomic_json(progress,dict(signature=signature,done=begin+len(ix)))
                print(label,'full prediction',begin+len(ix),'/',len(ids),'models',names,'seconds',round(time.monotonic()-start),flush=True)
        for arr in arrays.values():arr.flush()
        atomic_json(marker,signature);models.clear();torch.cuda.empty_cache()
    d=metadata(data,ids);b=np.linalg.norm(d['wm']-d['y'],axis=1)
    banks={n:pickle.load((a.out/(n+'-selectors.pkl')).open('rb')) for n in set(v['model'] for v in configs.values())}
    results=dict(cohort=label,samples=len(ids),games=len(np.unique(d['game'])),policies={},frozen_policy_sha256=hashlib.sha256((a.out/'frozen-policies.json').read_bytes()).hexdigest())
    raw128=np.linalg.norm(arrays['wide128']-d['y'],axis=1) if fresh else np.load(a.out.parent/'scaling-study/wide128-full-test-error.npy')
    if len(raw128)!=len(b):raise ValueError('Reference cohort size mismatch')
    results['raw128']=groups(raw128,d);results['wm']=groups(b,d,np.zeros(len(b),bool))
    bymodel={n:groups(np.linalg.norm(p-d['y'],axis=1),d) for n,p in arrays.items()}
    results['raw_models']=bymodel;intervals={}
    speed=np.linalg.norm(d['x'][:,2:4]*1.5,axis=1);ball_distance=np.linalg.norm(d['wm']-d['x'][:,30:32]*30,axis=1)
    masks={'WM error<10cm':b<.1,'WM error10–25cm':(b>=.1)&(b<.25),'WM error>1m':b>1.,
           'posCount0':d['count']==0,'posCount1':d['count']==1,'posCount2–3':(d['count']>=2)&(d['count']<=3),
           'posCount4–5':(d['count']>=4)&(d['count']<=5),'posCount>5':d['count']>5,
           'named age1–2':d['age']<=2,'named age3–5':(d['age']>=3)&(d['age']<=5),'named age6–10':(d['age']>=6)&(d['age']<=10),'named age>10':d['age']>10,
           'observed speed<0.15m/c':speed<.15,'observed speed0.15–0.6m/c':(speed>=.15)&(speed<.6),'observed speed>0.6m/c':speed>=.6,
           'velocity unknown':d['x'][:,4]<.5,'velocity age>3':d['x'][:,5]*30>3,
           'estimated target-ball<2m':(d['x'][:,34]>.5)&(ball_distance<2),
           'teammate':d['x'][:,37]>.5,'opponent':d['x'][:,37]<.5,
           'view60':np.isclose(d['x'][:,28]*180,60),'view120':np.isclose(d['x'][:,28]*180,120),'view180':np.isclose(d['x'][:,28]*180,180),
           'true distance<10m':(d['distance']>=0)&(d['distance']<10),'true distance10–25m':(d['distance']>=10)&(d['distance']<25),'true distance25–50m':(d['distance']>=25)&(d['distance']<50),'true distance>=50m':d['distance']>=50}
    # A selector score depends on the raw proposal, not the later cap/threshold.
    # Compute each distinct model/selector once, concurrently on CPU, and reuse.
    def cache_score(key):
        name,kind=key;path=output/(name+'-'+kind+'-score.npy');stamp=path.with_suffix('.json')
        if stamp.exists() and json.loads(stamp.read_text())==signature:return key,np.load(path,mmap_mode='r')
        scores=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(len(ids),));model=banks[name][kind]
        for begin in range(0,len(ids),524288):
            end=min(begin+524288,len(ids));dd={k:v[begin:end] for k,v in d.items()};z=features(dd,arrays[name][begin:end])[int(model['strong'])]
            scores[begin:end]=score_model(model['model'],z,model['classification'])
        scores.flush();atomic_json(stamp,signature);print('Cached full selector score',label,key,flush=True);return key,scores
    keys=sorted({(c['model'],c['candidate']) for c in configs.values() if c['candidate'] in banks[c['model']]})
    with ThreadPoolExecutor(max_workers=3) as pool:score_cache=dict(pool.map(cache_score,keys))
    # Save only a few diagnostic labels; never use these cases for policy selection.
    cases={};near=(d['distance']>=0)&(d['distance']<50)
    for mode,config in configs.items():
        pred=arrays[config['model']];error=np.empty(len(ids),np.float32);accept=np.empty(len(ids),bool)
        for begin in range(0,len(ids),131072):
            end=min(begin+131072,len(ids));dd={k:v[begin:end] for k,v in d.items()};pp=pred[begin:end]
            cap=config['cap_m'] if config.get('cap_per_cycle') is None else np.minimum(config['cap_m'],dd['age']*config['cap_per_cycle'])
            key=(config['model'],config['candidate'])
            yes=((score_cache[key][begin:end]>config['tuning']['threshold'])&(dd['count']>0)) if key in score_cache else accepted(config,banks[config['model']],dd,pp)
            position=corrected(pp,dd['wm'],config['alpha'],cap)
            error[begin:end]=np.where(yes,np.linalg.norm(position-dd['y'],axis=1),b[begin:end]);accept[begin:end]=yes
        m=groups(error,d,accept);subs={};near_subs={}
        for k,mask in masks.items():
            if mask.any():subs[k]=dict(summary(error[mask],b[mask],accept[mask],quantiles=False),raw128_mean=float(raw128[mask].mean()))
            nm=mask&near
            if nm.any():near_subs[k]=dict(summary(error[nm],b[nm],accept[nm],quantiles=False),raw128_mean=float(raw128[nm].mean()))
        results['policies'][mode]=dict(config=config,metrics=m,subgroups=subs,under50_subgroups=near_subs,per_game={str(g):summary(error[gg],b[gg],accept[gg]) for g in np.unique(d['game']) if (gg:=d['game']==g).any()})
        intervals[mode]=bootstrap(raw128-error,d['game']);intervals[mode+'/under50']=bootstrap((raw128-error)[near],d['game'][near])
        np.save(output/(mode+'-error.npy'),error);np.save(output/(mode+'-accept.npy'),accept)
        delta=error-b;worst=np.argpartition(delta,-min(10,len(delta)))[-10:]
        cases[mode]=[dict(dataset_index=int(ids[i]),game=data.games[int(d['game'][i])],pos_count=int(d['count'][i]),named_age=int(d['age'][i]),
                          true_distance=float(d['distance'][i]),wm_error=float(b[i]),policy_error=float(error[i]),added_error=float(delta[i]),wm=d['wm'][i].tolist(),truth=d['y'][i].tolist()) for i in sorted(worst,key=lambda i:-delta[i])]
        print(label,mode,'all',m['all']['mean'],'near50',m['under50']['mean'],'hurt25',m['all']['worse']['25'],flush=True)
    seeds={}
    if not fresh:
        test=load_small(a.out);seednames=['wide128','seed128_50511','seed128_50521']
        for n in seednames:
            p=np.load(a.out/(n+'-test-pred.npy'));seeds[n]=groups(np.linalg.norm(p-test['y'],axis=1),test)
    results['stability']=dict(seeds=seeds,bootstrap=intervals)
    results['diagnostic_cases']=cases;results['seconds']=time.monotonic()-start
    atomic_json(a.out/('fresh-results.json' if fresh else 'final-results.json'),results)
    print('Final evaluation completed',label,flush=True)

def load_small(out):return {k:v for k,v in np.load(out/'test-data.npz').items()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate','fresh']);p.add_argument('dataset',type=Path);p.add_argument('out',type=Path);p.add_argument('--fresh-dataset',type=Path);a=p.parse_args();a.device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(8)
    if a.action=='freeze':freeze(a.out)
    elif a.action=='fresh':
        # Multiple report coordinators can safely wait on the same independent check.
        import fcntl
        with (a.out/'fresh-evaluation.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            result=a.out/'fresh-results.json'
            signature=hashlib.sha256((a.out/'frozen-policies.json').read_bytes()).hexdigest()
            if result.exists() and json.loads(result.read_text())['frozen_policy_sha256']==signature:
                print('Reusing completed frozen-policy new-game evaluation',flush=True)
            else:evaluate(a,True)
    else:evaluate(a,False)
