"""Rescore frozen original predictions for true observer distance below50m."""
from pathlib import Path
import argparse,json,gc
import numpy as np
import joblib
from refinement_select import summary
from refinement_models import atomic_json

def main(a):
    cohort=np.load(a.logs/'evaluation/cohort.npz');b=cohort['baseline'];distance=cohort['distance']
    mask=np.isfinite(distance)&(distance>=0)&(distance<50);age=cohort['age'];truth=cohort['truth']
    result=dict(n=len(b),under50_n=int(mask.sum()),missing_distance=int((~np.isfinite(distance)|(distance<0)).sum()),
                semantics='Server true observer-target distance at the current cycle, strictly <50m. Evaluation labels only. Every retained target-cycle has equal weight.',models={},confidence=[])
    def record(e):
        return dict(all=summary(e,b),under50=summary(e[mask],b[mask]),
                    age_under50={label:summary(e[m],b[m]) for label,lo,hi in [('1',1,1),('2',2,2),('3',3,3),('4',4,4),('5',5,5),('6–10',6,10),('11–20',11,20),('21–30',21,30)] if (m:=mask&(age>=lo)&(age<=hi)).any()})
    result['models']['wm']=record(b)
    for name,path in [('point','evaluation/point-test.npz'),('gaussian','evaluation/gaussian-test.npz'),('mixture','evaluation/mixture-test.npz'),
                      ('wide128','scaling-study/wide128-full-test-error.npy'),('old_selector_mean','selector-study/full-mean-error.npy'),('old_selector_risk','selector-study/full-risk-error.npy')]:
        print('Rescore distance',name,flush=True)
        p=a.logs/path;e=np.load(p)['error'] if p.suffix=='.npz' else np.load(p,mmap_mode='r')
        result['models'][name]=record(e)
    policies=json.loads((a.logs/'confidence-study/results.json').read_text())['policies']
    for kind in ['gaussian','mixture']:
        saved=np.load(a.logs/('evaluation/'+kind+'-test.npz'))
        err=saved['error'][mask]
        if kind=='mixture':
            weight=saved['weight'][mask];centers=saved['means'][mask].reshape(-1,3,2)
            best=centers[np.arange(len(centers)),weight.argmax(1)];maperr=np.linalg.norm(best-truth[mask],axis=1)
        for r in policies:
            if r['kind']!=kind:continue
            candidate=maperr if r['readout']=='highest-weight center' else err
            if r['gate']=='always':accept=np.ones(mask.sum(),bool)
            elif r['gate']=='max_component_weight':accept=weight.max(1)>=r['threshold']
            else:
                radius=r['radius_m'];raw=np.load(a.logs/f'confidence-study/{kind}-R{radius:g}-test.npy',mmap_mode='r')[mask]
                calibrator=joblib.load(a.logs/f'confidence-study/{kind}-R{radius:g}-isotonic.joblib');accept=calibrator.predict(raw)>=r['threshold']
            m=summary(np.where(accept,candidate,b[mask]),b[mask],accept)
            result['confidence'].append(dict(kind=kind,gate=r['gate'],readout=r['readout'],radius_m=r['radius_m'],threshold=r['threshold'],metrics=m))
        del saved;gc.collect()
    atomic_json(a.out,result)
    print('Distance rescoring complete',result['under50_n'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('logs',type=Path);p.add_argument('out',type=Path);main(p.parse_args())
