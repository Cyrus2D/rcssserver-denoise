"""Fixed safety sensitivity of the frozen practical policy; no retuning."""
from pathlib import Path
import argparse,json
import numpy as np
from refinement_select import corrected,groups,summary
from refinement_models import atomic_json

def fresh_labels(root):
    manifest=json.loads((root/'manifest.json').read_text());fields={k:[] for k in ['wm','y','distance','age']}
    for r in manifest['observers']:
        path=Path(r['path']);flags=np.load(path/'flags.npy').ravel();age=np.load(path/'age.npy').ravel()
        yes=((flags&3)==3)&((flags&4)==0)&(age>0)&(age<=30)
        for k in fields:
            arr=np.load(path/(k+'.npy'));fields[k].append(arr.reshape((-1,2) if k in ['wm','y'] else (-1,))[yes])
    return {k:np.concatenate(v) for k,v in fields.items()}

def run(a,cohort):
    fresh=cohort=='fresh';resultfile=a.out/('fresh-results.json' if fresh else 'final-results.json')
    r=json.loads(resultfile.read_text());base=r['policies']['near50_risk'];config=base['config']
    folder=a.out/('fresh-evaluation' if fresh else 'full-evaluation')
    if fresh:d=fresh_labels(a.fresh_dataset)
    else:
        saved=np.load(a.out.parent/'evaluation/cohort.npz');d={k:saved[v] for k,v in [('wm','wm'),('y','truth'),('distance','distance'),('age','age')]}
    p=np.load(folder/(config['model']+'-pred.npy'),mmap_mode='r');accept=np.load(folder/'near50_risk-accept.npy',mmap_mode='r')
    assert len(p)==len(d['y'])==len(accept)
    b=np.linalg.norm(d['wm']-d['y'],axis=1);policies={'uncapped_practical':base['metrics']}
    for cap in [1.,2.,4.]:
        q=corrected(p,d['wm'],config['alpha'],cap);e=np.where(accept,np.linalg.norm(q-d['y'],axis=1),b)
        policies['practical_cap'+str(int(100*cap))]=groups(e,d,accept)
        np.save(folder/f'practical-cap{int(cap*100)}-error.npy',e)
        print(cohort,'practical cap',cap,'mean',policies['practical_cap'+str(int(100*cap))]['all']['mean'],flush=True)
    dest=a.out/'safety-results.json';allresults=json.loads(dest.read_text()) if dest.exists() else {}
    allresults[cohort]=dict(base_policy='near50_risk',config=config,policies=policies,
        note='Additional100/200/400cm ceilings on exactly the frozen practical selector, same acceptance and weights, no threshold tuning. Declared before fresh-game prediction scoring.')
    atomic_json(dest,allresults)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=['test','fresh']);p.add_argument('out',type=Path);p.add_argument('--fresh-dataset',type=Path);a=p.parse_args();run(a,a.cohort)
