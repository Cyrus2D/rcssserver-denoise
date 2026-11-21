#!/usr/bin/env python3
"""Exploratory runtime-observable WM-versus-point selector comparison.

Fit: 10 selection games. Tune: separate 10 calibration games. Evaluate:
old 20 test games, explicitly reused exploratorily. Truth is target only.
"""
import argparse,json,time,pickle,hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsRegressor
from data import Data

THRESHOLDS=[0,.05,.1,.25,.5,1.]
FEATURE_NAMES=None

def selector_features(data,indices,prediction):
    """Only observer inputs, known unum, predicted position and stock WM."""
    x=data.x[indices]
    correction=prediction-data.wm[indices]
    gap=data.age[indices].astype(np.float32)
    speed=np.linalg.norm(x[:,2:4]*1.5,axis=1)
    wm_speed=np.linalg.norm(x[:,20:22]*1.5,axis=1)
    move_speed=np.linalg.norm(x[:,12:14]*1.5,axis=1)
    norm=np.linalg.norm(correction,axis=1)
    d=np.column_stack([correction[:,0],correction[:,1],norm,
                       norm/np.maximum(gap,1),speed,wm_speed,move_speed,
                       np.linalg.norm(x[:,2:4]-x[:,12:14],axis=1),
                       np.linalg.norm(x[:,30:32]*30-x[:,2:4]*1.5*gap[:,None],axis=1)])
    role=np.eye(11,dtype=np.float32)[data.unum[indices]-1]
    return np.nan_to_num(np.concatenate([x,d,role],axis=1),nan=0,posinf=10,neginf=-10).astype(np.float32)

EXTRA_NAMES=['correction_dx','correction_dy','correction_norm','correction_per_cycle',
             'last_speed','wm_speed','two_sighting_speed','last_vs_mean_velocity','estimated_ball_distance']
# Low-dimensional local competence features, all runtime-observable.
NEIGHBOR_FEATURES=[10,19,2,3,20,21,6,7,37,51+2,51+4,51+8]

def summarize(error,baseline,accept=None):
    e=np.asarray(error,dtype=np.float64);b=np.asarray(baseline,dtype=np.float64);delta=e-b
    n=len(e)
    return dict(n=n,mean=float(e.mean()),baseline_mean=float(b.mean()),gain=float((b-e).mean()),
                gain_pct=float(100*(b-e).mean()/b.mean()),p95=float(np.quantile(e,.95)),p99=float(np.quantile(e,.99)),
                max_error=float(e.max()),max_added_error=float(max(0,delta.max())),
                hurt_pct={str(t):float(100*(delta>max(t,1e-5)).mean()) for t in THRESHOLDS},
                help_over_1cm_pct=float(100*(delta<-.01).mean()),
                accept_pct=float(100*np.mean(accept)) if accept is not None else None)

def threshold_policy(score,gain,available,kind):
    """Predefined threshold grid; mean and severity-aware decisions on tune only."""
    if kind in ['win_classifier','cost_classifier']:
        choices=np.r_[-1e9,.1,.25,.4,.5,.6,.7,.8,.9,.95,.99,1e9]
    else:
        choices=np.r_[-1e9,-.25,-.1,0,.02,.05,.1,.2,.5,1.,2.,1e9]
    policies={}
    for name,penalty in [('mean',0.),('risk',3.)]:
        vals=[]
        for threshold in choices:
            accept=(score>threshold)&available
            utility=np.where(accept,gain-penalty*np.maximum(-gain,0),0)
            vals.append(float(utility.mean()))
        i=int(np.argmax(vals));policies[name]=dict(threshold=float(choices[i]),penalty=penalty,
                                                  validation_utility=vals[i],accept_pct=float(100*((score>choices[i])&available).mean()))
    return policies

def predict_score(candidate,x):
    kind=candidate['kind']
    if kind=='speed_rule':return np.maximum(x[:,55],x[:,56])
    if kind=='errors':return candidate['wm'].predict(x)-candidate['nn'].predict(x)
    if kind in ['win_classifier','cost_classifier']:return candidate['model'].predict_proba(x)[:,1]
    if kind=='cluster':
        z=candidate['scale'].transform(x[:,NEIGHBOR_FEATURES]);labels=candidate['model'].predict(z)
        return candidate['gain'][labels]
    if kind=='knn':return candidate['model'].predict(candidate['scale'].transform(x[:,NEIGHBOR_FEATURES]))
    return candidate['model'].predict(x)

def make_selector(kind,x,stock_error,model_error,seed):
    gain=stock_error-model_error
    common=dict(max_iter=180,max_leaf_nodes=15,learning_rate=.08,min_samples_leaf=200,l2_regularization=2.,
                early_stopping=False,random_state=seed)
    result={'kind':kind}
    if kind=='speed_rule':return result
    if kind=='errors':
        result['wm']=HistGradientBoostingRegressor(**common).fit(x,stock_error)
        result['nn']=HistGradientBoostingRegressor(**common).fit(x,model_error)
    elif kind in ['win_classifier','cost_classifier']:
        weight=None if kind=='win_classifier' else np.abs(gain)
        result['model']=HistGradientBoostingClassifier(**common).fit(x,gain>0,sample_weight=weight)
    elif kind in ['cluster','knn']:
        scaler=StandardScaler().fit(x[:,NEIGHBOR_FEATURES]);z=scaler.transform(x[:,NEIGHBOR_FEATURES]);result['scale']=scaler
        if kind=='cluster':
            result['model']=MiniBatchKMeans(n_clusters=128,batch_size=4096,n_init=3,max_iter=150,random_state=seed).fit(z)
            labels=result['model'].predict(z);count=np.bincount(labels,minlength=128)
            total=np.bincount(labels,weights=gain,minlength=128)
            # Shrink tiny neighborhoods toward global mean instead of overfitting rare labels.
            result['gain']=(total+200*gain.mean())/(count+200)
        else:
            rng=np.random.default_rng(seed);take=rng.choice(len(x),min(30000,len(x)),replace=False)
            result['model']=KNeighborsRegressor(n_neighbors=32,weights='distance',algorithm='kd_tree',leaf_size=40,n_jobs=8).fit(z[take],gain[take])
    else:
        if kind=='gain_quantile':common.update(loss='quantile',quantile=.25)
        result['model']=HistGradientBoostingRegressor(**common).fit(x,gain)
    return result

def dataset_indices(data):
    rng=np.random.default_rng(50502)
    return {k:np.sort(rng.choice(data.split_indices[k],min(500000,len(data.split_indices[k])),replace=False)) for k in ['selection','calibration']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path);parser.add_argument('evaluation',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--fit-samples',type=int,default=250000);parser.add_argument('--test-samples',type=int,default=1000000)
    parser.add_argument('--batch-size',type=int,default=131072);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=Data(args.dataset,['validation','test']);ids=dataset_indices(data)
    cohort=np.load(args.evaluation/'cohort.npz');testids=cohort['indices'];stock_test=cohort['baseline']
    assert np.array_equal(testids,data.split_indices['test'])
    rng=np.random.default_rng(50571);fit_pos=np.sort(rng.choice(len(ids['selection']),min(args.fit_samples,len(ids['selection'])),replace=False))
    test_pos=np.sort(rng.choice(len(testids),min(args.test_samples,len(testids)),replace=False))
    np.savez(args.output/'indices.npz',fit_dataset_indices=ids['selection'][fit_pos],fit_cache_positions=fit_pos,
             tune_dataset_indices=ids['calibration'],test_cache_positions=test_pos,test_dataset_indices=testids[test_pos])
    np.savez(args.output/'validation-targets.npz',selection_y=data.y[ids['selection']],calibration_y=data.y[ids['calibration']],
             selection_indices=ids['selection'],calibration_indices=ids['calibration'],
             selection_game=data.game[ids['selection']],calibration_game=data.game[ids['calibration']])
    print('Validation targets ready:',args.output/'validation-targets.npz',flush=True)
    names=data.feature_names+EXTRA_NAMES+['unum_'+str(i) for i in range(1,12)]
    cache={}
    for split in ['selection','calibration','test']:
        p=np.load(args.evaluation/('point-'+split+'.npz'))
        pos=fit_pos if split=='selection' else test_pos if split=='test' else np.arange(len(ids['calibration']))
        di=ids[split][pos] if split!='test' else testids[pos]
        pred=p['pred'][pos];error=p['error'][pos]
        stock=np.linalg.norm(data.wm[di]-data.y[di],axis=1)
        cache[split]=dict(x=selector_features(data,di,pred),e=error,b=stock,indices=di,
                          available=data.count[di]>0,pred=pred)
    fit=cache['selection'];tune=cache['calibration'];test=cache['test'];gain_tune=tune['b']-tune['e']
    meta=dict(feature_names=names,feature_policy='Original observation-only features, known unum and geometric differences between frozen point prediction and WM. No truth/errors/distance/group truth flags as inputs.',
              train_games=data.selection_games,tune_games=data.calibration_games,test_games=sorted(g for g,s in data.manifest['split'].items() if s=='test'),
              fit_n=len(fit_pos),tune_n=len(ids['calibration']),test_subset_n=len(test_pos),full_test_n=len(testids),
              seed=50571,limitations='Exploratory reuse of the previous test20. Selector fit uses validation-selection10 rather than point-model-training160. Need new external games for confirmation.',
              fresh_policy='Never override WM when WM position count is 0. No truth-dependent identity rule.')
    result=dict(metadata=meta,candidates={},baselines={})
    result['baselines']['wm']=summarize(test['b'],test['b'],np.zeros(len(test_pos),bool))
    result['baselines']['always_point']=summarize(test['e'],test['b'],np.ones(len(test_pos),bool))
    result['baselines']['protect_fresh_only']=summarize(np.where(test['available'],test['e'],test['b']),test['b'],test['available'])
    result['baselines']['oracle_not_deployable']=summarize(np.minimum(test['e'],test['b']),test['b'],test['e']<test['b'])
    start=time.monotonic();candidates={}
    for kind in ['speed_rule','gain','errors','win_classifier','cost_classifier','gain_quantile','cluster','knn']:
        stamp=time.monotonic();candidate=make_selector(kind,fit['x'],fit['b'],fit['e'],50571)
        scores=predict_score(candidate,tune['x']);policies=threshold_policy(scores,gain_tune,tune['available'],kind)
        test_scores=predict_score(candidate,test['x']);records={}
        for mode,p in policies.items():
            accept=(test_scores>p['threshold'])&test['available'];error=np.where(accept,test['e'],test['b'])
            record=dict(tuning=p,metrics=summarize(error,test['b'],accept))
            # True-error subgroups are evaluation only, not selector inputs.
            near=(test['x'][:,34]>.5)&(test['x'][:,59]<2)
            for label,mask in [('stock_under_10cm',test['b']<.1),('estimated_near_ball_2m',near),('teammate',test['x'][:,37]>.5),('opponent',test['x'][:,37]<=.5)]:
                if mask.any():record[label]=summarize(error[mask],test['b'][mask],accept[mask])
            records[mode]=record
        candidate['policies']=policies;candidates[kind]=candidate
        with (args.output/(kind+'.pkl')).open('wb') as f:pickle.dump(candidate,f)
        result['candidates'][kind]=dict(policies=records,fit_and_eval_seconds=time.monotonic()-stamp)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(kind,'validation policies',policies,'test subset mean/risk',records['mean']['metrics']['mean'],records['risk']['metrics']['mean'],'seconds',time.monotonic()-stamp,flush=True)
    best={mode:max(candidates,key=lambda k:candidates[k]['policies'][mode]['validation_utility']) for mode in ['mean','risk']}
    result['validation_selected']=best
    # Full test is evaluated only for frozen validation-selected choices, never selected by test rank.
    p=np.load(args.evaluation/'point-test.npz');point=p['pred'];point_error=p['error'];totals={}
    winner_names=sorted(set(best.values()))
    error_arrays={mode:np.empty(len(testids),np.float32) for mode in best};accept_arrays={mode:np.empty(len(testids),bool) for mode in best}
    for begin in range(0,len(testids),args.batch_size):
        end=min(begin+args.batch_size,len(testids));di=testids[begin:end];x=selector_features(data,di,point[begin:end]);available=data.count[di]>0
        scores={name:predict_score(candidates[name],x) for name in winner_names}
        for mode,name in best.items():
            accept=(scores[name]>candidates[name]['policies'][mode]['threshold'])&available
            accept_arrays[mode][begin:end]=accept;error_arrays[mode][begin:end]=np.where(accept,point_error[begin:end],stock_test[begin:end])
        if begin%(args.batch_size*20)==0:print('Full test',end,'/',len(testids),flush=True)
    result['full_test']={mode:dict(selector=best[mode],metrics=summarize(error_arrays[mode],stock_test,accept_arrays[mode])) for mode in best}
    for mode in best:
        np.save(args.output/('full-'+mode+'-error.npy'),error_arrays[mode]);np.save(args.output/('full-'+mode+'-accept.npy'),accept_arrays[mode])
        by_game={}
        for g in np.unique(data.game[testids]):
            mask=data.game[testids]==g;by_game[data.games[int(g)]]=summarize(error_arrays[mode][mask],stock_test[mask],accept_arrays[mode][mask])
        result['full_test'][mode]['per_game']=by_game
    result['elapsed_seconds']=time.monotonic()-start
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Finished',result['full_test'],flush=True)

if __name__=='__main__':main()
