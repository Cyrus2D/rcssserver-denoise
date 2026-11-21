#!/usr/bin/env python3
"""Observation-only expected-gain selection among WM and three frozen experts.

Uses endpoint gathers, rather than loading the entire 25GB validation/test
feature history. Selector fitting/tuning are separated by whole game.
"""
import argparse,json,pickle,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from selector_study import selector_features,summarize

KINDS=['point','gaussian','mixture']

class EndpointStore:
    def __init__(self,root):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.obs=[r for r in self.manifest['observers'] if r['split'] in ['validation','test']]
        self.ends=np.cumsum([r['rows']*21 for r in self.obs]);self.starts=np.r_[0,self.ends[:-1]]

    def get(self,indices):
        ids=np.asarray(indices,np.int64);n=len(ids)
        x=np.empty((n,51),np.float32);wm=np.empty((n,2),np.float32)
        age=np.empty(n,np.uint16);unum=np.empty(n,np.uint8);count=np.empty(n,np.uint8)
        which=np.searchsorted(self.ends,ids,side='right')
        for ob in np.unique(which):
            mask=which==ob;local=ids[mask]-self.starts[ob];t,slot=np.divmod(local,21);p=Path(self.obs[int(ob)]['path'])
            x[mask]=np.load(p/'x.npy',mmap_mode='r')[t,slot]
            wm[mask]=np.load(p/'wm.npy',mmap_mode='r')[t,slot]
            age[mask]=np.load(p/'age.npy',mmap_mode='r')[t,slot]
            count[mask]=np.load(p/'count.npy',mmap_mode='r')[t,slot]
            unum[mask]=(np.load(p/'ids.npy')[slot]-1)%11+1
        return SimpleNamespace(x=x,wm=wm,age=age,unum=unum,count=count)

def features(endpoint,preds):
    n=len(endpoint.x);x=selector_features(endpoint,np.arange(n),preds['point']['pred'])
    extras=[]
    for kind in ['gaussian','mixture']:
        p=preds[kind];correction=p['pred']-endpoint.wm
        extras.extend([correction[:,0],correction[:,1],np.linalg.norm(correction,axis=1),np.sqrt(np.maximum(0,p['cov'][:,:2].sum(1)))])
    weight=preds['mixture']['weight']
    extras.extend([weight.max(1),-(weight*np.log(np.maximum(weight,1e-12))).sum(1)])
    return np.column_stack([x,*extras]).astype(np.float32)

def read_predictions(evaluation,split,positions=None):
    result={}
    for kind in KINDS:
        p=np.load(evaluation/(kind+'-'+split+'.npz'));r={}
        for name in ['pred','error']+(['cov'] if kind!='point' else [])+(['weight'] if kind=='mixture' else []):
            a=p[name];r[name]=a if positions is None else a[positions]
        result[kind]=r
    return result

def choose(scores,threshold,available):
    expert=scores.argmax(1);value=scores[np.arange(len(scores)),expert]
    return np.where((value>threshold)&available,expert+1,0).astype(np.uint8)

def used_error(choice,base,preds):
    errors=np.column_stack([base]+[preds[k]['error'] for k in KINDS])
    return errors[np.arange(len(choice)),choice]

def tune(scores,gains,available):
    records={}
    for name,penalty in [('mean',0.),('risk',3.)]:
        threshold_grid=[-1e9,-.1,0,.02,.05,.1,.2,.5,1.,1e9]
        values=[]
        for threshold in threshold_grid:
            choice=choose(scores,threshold,available)
            gain=np.column_stack([np.zeros(len(gains)),gains])[np.arange(len(gains)),choice]
            utility=gain-penalty*np.maximum(-gain,0);values.append(float(utility.mean()))
        i=int(np.argmax(values));records[name]=dict(threshold=threshold_grid[i],validation_utility=values[i],penalty=penalty)
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('evaluation',type=Path);p.add_argument('selector_study',type=Path)
    p.add_argument('--batch-size',type=int,default=131072);args=p.parse_args();out=args.selector_study/'multi-expert';out.mkdir(parents=True,exist_ok=True)
    store=EndpointStore(args.dataset);index=np.load(args.selector_study/'indices.npz');targets=np.load(args.selector_study/'validation-targets.npz')
    fitpos=index['fit_cache_positions'];testpos=index['test_cache_positions']
    groups={}
    for split in ['selection','calibration','test']:
        positions=fitpos if split=='selection' else testpos if split=='test' else None
        ids=index['fit_dataset_indices'] if split=='selection' else index['tune_dataset_indices'] if split=='calibration' else index['test_dataset_indices']
        ep=store.get(ids);pred=read_predictions(args.evaluation,split,positions)
        if split=='test':truth=np.load(args.evaluation/'cohort.npz')['truth'][testpos]
        else:truth=targets['selection_y'][fitpos] if split=='selection' else targets['calibration_y']
        base=np.linalg.norm(ep.wm-truth,axis=1);x=features(ep,pred);gains=base[:,None]-np.column_stack([pred[k]['error'] for k in KINDS])
        groups[split]=dict(x=x,pred=pred,base=base,gains=gains,available=ep.count>0)
        print('Prepared multi-expert',split,x.shape,flush=True)
    fit,tuning,test=[groups[s] for s in ['selection','calibration','test']]
    common=dict(max_iter=180,max_leaf_nodes=15,learning_rate=.08,min_samples_leaf=200,l2_regularization=2.,early_stopping=False,random_state=50572)
    estimators=[]
    for i,kind in enumerate(KINDS):
        model=HistGradientBoostingRegressor(**common).fit(fit['x'],fit['gains'][:,i]);estimators.append(model);print('Fitted expert gain',kind,flush=True)
    scores=np.column_stack([m.predict(tuning['x']) for m in estimators]);policies=tune(scores,tuning['gains'],tuning['available'])
    testscore=np.column_stack([m.predict(test['x']) for m in estimators]);result=dict(policies={},tuning=policies,
        metadata=dict(fit_games='selection10',tune_games='calibration10',test_games='old test20 exploratory reuse',
                      features='71 point-selector observer/output features plus Gaussian/mixture correction and covariance, mixture maximum weight/entropy; no true states/errors as inputs',
                      target='expected improvement versus WM for each frozen model',fresh_policy='retain WM whenever its position count is0'))
    for name,policy in policies.items():
        choice=choose(testscore,policy['threshold'],test['available']);error=used_error(choice,test['base'],test['pred'])
        result['policies'][name]=dict(metrics=summarize(error,test['base'],choice!=0),choice_pct={k:float(100*(choice==i).mean()) for i,k in enumerate(['wm']+KINDS)})
    with (out/'selector.pkl').open('wb') as f:pickle.dump(dict(estimators=estimators,policies=policies),f)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    cohort=np.load(args.evaluation/'cohort.npz');allids=cohort['indices'];base=cohort['baseline'];allpred=read_predictions(args.evaluation,'test')
    choices={name:np.empty(len(allids),np.uint8) for name in policies};errors={name:np.empty(len(allids),np.float32) for name in policies}
    start=time.monotonic()
    for begin in range(0,len(allids),args.batch_size):
        end=min(begin+args.batch_size,len(allids));ep=store.get(allids[begin:end]);pred={k:{field:a[begin:end] for field,a in values.items()} for k,values in allpred.items()}
        x=features(ep,pred);score=np.column_stack([m.predict(x) for m in estimators])
        for name,policy in policies.items():
            action=choose(score,policy['threshold'],ep.count>0);choices[name][begin:end]=action;errors[name][begin:end]=used_error(action,base[begin:end],pred)
        if begin%(20*args.batch_size)==0:print('Multi-expert full test',end,'/',len(allids),'seconds',time.monotonic()-start,flush=True)
    result['full_test']={}
    for name in policies:
        result['full_test'][name]=dict(metrics=summarize(errors[name],base,choices[name]!=0),choice_pct={k:float(100*(choices[name]==i).mean()) for i,k in enumerate(['wm']+KINDS)})
        np.save(out/('full-'+name+'-choice.npy'),choices[name]);np.save(out/('full-'+name+'-error.npy'),errors[name])
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print('Completed multi-expert',result['full_test'],flush=True)

if __name__=='__main__':main()
