"""Additional exploratory new-game scoring of already validation-tuned trees."""
from pathlib import Path
import argparse,json,pickle
import numpy as np
from data import Data
from refinement_select import features,groups
from refinement_models import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('fresh_dataset',type=Path);a=p.parse_args()
    controls=json.loads((a.out/'tree-controls.json').read_text());models=pickle.load((a.out/'tree-controls.pkl').open('rb'))
    data=Data(a.fresh_dataset,['test']);ids=data.split_indices['test']
    labels={k:getattr(data,k)[ids] for k in ['wm','y','distance','count']};baseline=np.linalg.norm(labels['wm']-labels['y'],axis=1)
    result=dict(note='Additional exploratory reuse of new games after primary frozen-policy confirmation. Tree family and thresholds chosen only by original validation utility. No new-game fitting or threshold tuning.',models={})
    for name,mode in [('continue128','mean'),('wmresidual128_l1','near50_risk')]:
        raw=np.load(a.out/'fresh-evaluation'/(name+'-pred.npy'),mmap_mode='r');error=np.linalg.norm(raw-labels['y'],axis=1)
        for depth in [3,5]:
            candidates=['tree74_cost'+str(depth),'tree74_gain'+str(depth)]
            key=max(candidates,key=lambda k:controls['models'][name][k]['policies'][mode]['tuning']['utility'])
            config=controls['models'][name][key];cut=config['policies'][mode]['tuning']['threshold'];model=models[name+'/'+key]
            accept=np.empty(len(ids),bool)
            for begin in range(0,len(ids),524288):
                end=min(begin+524288,len(ids));ix=ids[begin:end];d={k:getattr(data,k)[ix] for k in ['x','wm','age','count','unum']}
                z=features(d,raw[begin:end])[1];score=model.predict_proba(z)[:,1] if 'cost'in key else model.predict(z)
                accept[begin:end]=(score>cut)&(d['count']>0)
            used=np.where(accept,error,baseline);r=dict(model=name,selector=key,nodes=config['nodes'],threshold=cut,objective=mode,metrics=groups(used,labels,accept))
            result['models'][name+'/'+key]=r
            atomic_json(a.out/'tree-controls-fresh.json',result);print('Additional fresh-tree check',name,key,r['metrics']['under50']['mean'],r['metrics']['under50']['worse']['25'],flush=True)
