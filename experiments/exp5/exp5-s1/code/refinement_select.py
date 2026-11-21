"""Small readable trees, WM/model routing, and bounded/blended corrections.

All fitting uses selection10; policy tuning uses calibration10. Test20 is
exploratory reuse. True observer distance is used only for reported subgroups.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,pickle,time
import numpy as np
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier,export_text
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from selector_study import selector_features
from refinement_models import atomic_json

MODES={'mean':(0.,False,None),'risk':(3.,False,None),
       'near50_mean':(0.,True,None),'near50_risk':(3.,True,None),
       'harm1':(0.,False,1.),'harm3':(0.,False,3.)}
BASIC_NAMES=['pos_count','named_sighting_age','last_speed','velocity_age','velocity_known',
             'wm_speed','wm_velocity_known','correction_distance','estimated_observer_distance',
             'estimated_ball_distance','ball_known','teammate','view_degrees','mean_sighting_speed']

def load_data(out,split):
    return {k:v for k,v in np.load(out/(split+'-data.npz')).items()}

def features(d,p):
    x=d['x'];speed=np.linalg.norm(x[:,2:4]*1.5,axis=1)
    distance=np.linalg.norm(d['wm']-x[:,24:26]*50,axis=1)
    ball=np.linalg.norm(d['wm']-x[:,30:32]*30,axis=1)
    basic=np.column_stack([d['count'],d['age'],speed,x[:,5]*30,x[:,4],np.linalg.norm(x[:,20:22]*1.5,axis=1),x[:,22],
         np.linalg.norm(p-d['wm'],axis=1),distance,np.where(x[:,34]>.5,ball,100),x[:,34],x[:,37],x[:,28]*180,np.linalg.norm(x[:,12:14]*1.5,axis=1)]).astype(np.float32)
    obj=SimpleNamespace(**{k:d[k] for k in ['x','wm','age','unum']})
    strong=np.column_stack([selector_features(obj,np.arange(len(x)),p),basic[:,[8,9,10]]]).astype(np.float32)
    return basic,strong

def summary(e,b,accept=None,quantiles=True):
    e=np.asarray(e,np.float64);b=np.asarray(b,np.float64);delta=e-b
    if not len(e):return None
    return dict(n=len(e),mean=float(e.mean()),stock=float(b.mean()),gain_pct=float(100*(b-e).mean()/b.mean()),
        p95=float(np.quantile(e,.95)) if quantiles else None,p99=float(np.quantile(e,.99)) if quantiles else None,max_error=float(e.max()),max_added=float(max(0,delta.max())),
        use_pct=float(100*np.mean(accept)) if accept is not None else 100.,
        worse={str(cm):float(100*(delta>cm/100+1e-5).mean()) for cm in [0,1,5,10,25,50,100]},
        better_pct=float(100*(delta<-1e-5).mean()))

def groups(e,d,accept=None):
    b=np.linalg.norm(d['wm']-d['y'],axis=1);near=(d['distance']>=0)&(d['distance']<50)
    result={'all':summary(e,b,accept),'under50':summary(e[near],b[near],None if accept is None else accept[near])}
    return result

def tune(score,gain,available,estimated50,error,b,classification=False):
    grid=[-1e9,.1,.25,.4,.5,.6,.7,.8,.9,.95,.99,1e9] if classification else [-1e9,-.25,-.1,0,.02,.05,.1,.2,.5,1.,2.,1e9]
    # Threshold-grid binning computes the same strict score>threshold policies
    # without repeatedly scanning all500k rows for every threshold and objective.
    bins=np.searchsorted(grid,score,side='left');size=len(grid)+1
    def suffix(weight,mask):
        sums=np.bincount(bins[mask],weights=weight[mask],minlength=size)
        return np.cumsum(sums[::-1])[::-1][1:]
    ones=np.ones(len(gain));use=100*suffix(ones,available)/len(gain)
    hurt=100*suffix(ones,available&(error-b>.25))/len(gain);best={}
    for mode,(penalty,near,budget) in MODES.items():
        mask=estimated50 if near else np.ones(len(gain),bool)
        utility=suffix(gain.astype(float)-penalty*np.maximum(-gain.astype(float),0),available&mask)/max(1,mask.sum())
        eligible=np.arange(len(grid)) if budget is None else np.flatnonzero(hurt<=budget)
        i=eligible[np.argmax(utility[eligible])]
        best[mode]=dict(threshold=float(grid[i]),utility=float(utility[i]),calibration_hurt25=float(hurt[i]),calibration_use_pct=float(use[i]))
    return best

def score_model(m,z,classification):
    return m.predict_proba(z)[:,1] if classification else m.predict(z)

def candidate_record(name,policies,score_test,error_test,test):
    result={'policies':{}}
    for mode,r in policies.items():
        accept=(score_test>r['threshold'])&(test['count']>0)
        b=np.linalg.norm(test['wm']-test['y'],axis=1);used=np.where(accept,error_test,b)
        result['policies'][mode]=dict(tuning=r,metrics=groups(used,test,accept))
    return result

def fit_model(args,name):
    path=args.out/(name+'-selectors.json')
    if path.exists():print('Reuse selectors',name,flush=True);return
    selection=load_data(args.out,'selection');cal=load_data(args.out,'calibration');test=load_data(args.out,'test')
    pred={s:np.load(args.out/(name+'-'+s+'-pred.npy')) for s in ['selection','calibration','test']}
    datasets={'selection':selection,'calibration':cal,'test':test}
    feature={s:features(datasets[s],pred[s]) for s in datasets}
    errors={s:np.linalg.norm(pred[s]-datasets[s]['y'],axis=1) for s in datasets}
    baseline={s:np.linalg.norm(datasets[s]['wm']-datasets[s]['y'],axis=1) for s in datasets}
    gain={s:baseline[s]-errors[s] for s in datasets}
    ix=np.sort(np.random.default_rng(50571).choice(len(selection['x']),250000,replace=False))
    available=cal['count']>0;estimated50=feature['calibration'][0][:,8]<50
    raw=groups(errors['test'],test);result=dict(model=name,raw=raw,candidates={},selected={})
    models={};started=time.monotonic()
    # Transparent nested rules, chosen by calibration utility only.
    for family in ['count','count_speed','count_speed_correction']:
        scores={};test_accepts={};best={}
        for count in [0,1,2,3,5,10]:
            for speed in ([0] if family=='count' else [0,.05,.15,.3,.6,1.]):
                for correction in ([1e9] if family!='count_speed_correction' else [.25,.5,1.,2.,5.,1e9]):
                    z=feature['calibration'][0];zt=feature['test'][0]
                    accept=available&(z[:,0]>count)&(z[:,2]>=speed)&(z[:,7]<=correction)
                    for mode,(penalty,near,budget) in MODES.items():
                        mask=estimated50 if near else np.ones(len(accept),bool)
                        utility=float(np.where(accept[mask],gain['calibration'][mask]-penalty*np.maximum(-gain['calibration'][mask],0),0).mean())
                        hurt=float(100*(accept&(errors['calibration']-baseline['calibration']>.25)).mean())
                        if budget is not None and hurt>budget:continue
                        if mode not in best or utility>best[mode]['utility']:
                            best[mode]=dict(count=count,speed=speed,correction=correction,utility=utility,calibration_hurt25=hurt)
        rec={'policies':{}}
        for mode,r in best.items():
            z=feature['test'][0];a=(test['count']>0)&(z[:,0]>r['count'])&(z[:,2]>=r['speed'])&(z[:,7]<=r['correction'])
            rec['policies'][mode]=dict(tuning=r,metrics=groups(np.where(a,errors['test'],baseline['test']),test,a))
        result['candidates'][family]=rec
    for kind,depth in [('tree_gain',k) for k in [2,3,4,5]]+[('tree_cost',k) for k in [2,3,4,5]]+[('strong_gain',0),('strong_cost',0)]:
        key=kind+(str(depth) if depth else '');classification='cost'in kind;strong=kind.startswith('strong')
        if strong:
            common=dict(max_iter=180,max_leaf_nodes=15,learning_rate=.08,min_samples_leaf=200,l2_regularization=2.,early_stopping=False,random_state=50571)
            model=HistGradientBoostingClassifier(**common) if classification else HistGradientBoostingRegressor(**common)
        else:
            common=dict(max_depth=depth,min_samples_leaf=2000,random_state=50571)
            model=DecisionTreeClassifier(**common) if classification else DecisionTreeRegressor(**common)
        xx=feature['selection'][int(strong)][ix];g=gain['selection'][ix]
        if classification:model.fit(xx,g>0,sample_weight=np.abs(g))
        else:model.fit(xx,g)
        cs=score_model(model,feature['calibration'][int(strong)],classification)
        ts=score_model(model,feature['test'][int(strong)],classification)
        policies=tune(cs,gain['calibration'],available,estimated50,errors['calibration'],baseline['calibration'],classification)
        rec=candidate_record(key,policies,ts,errors['test'],test)
        if not strong:rec.update(tree_text=export_text(model,feature_names=BASIC_NAMES,decimals=3),nodes=model.tree_.node_count,depth=model.get_depth())
        result['candidates'][key]=rec;models[key]=dict(model=model,classification=classification,strong=strong)
        print(name,key,[(m,round(v['utility'],4)) for m,v in policies.items()],flush=True)
    for mode in MODES:
        eligible=[k for k,v in result['candidates'].items() if mode in v['policies']]
        winner=max(eligible,key=lambda k:result['candidates'][k]['policies'][mode]['tuning']['utility'])
        result['selected'][mode]=dict(candidate=winner,**result['candidates'][winner]['policies'][mode])
    result['seconds']=time.monotonic()-started
    with (args.out/(name+'-selectors.pkl')).open('wb') as f:pickle.dump(models,f)
    atomic_json(path,result);print('Finished selectors',name,flush=True)

def aggregate(args):
    records={p.name.removesuffix('-selectors.json'):json.loads(p.read_text()) for p in args.out.glob('*-selectors.json')}
    winners={}
    for mode in MODES:
        name=max(records,key=lambda k:records[k]['selected'][mode]['tuning']['utility'])
        winners[mode]=dict(model=name,**records[name]['selected'][mode])
    result=dict(models=records,winners=winners,note='Validation-selected exploratory policies; raw test metrics never choose candidates. All <50m report filters use true distance; near50 objectives use estimated WM distance.')
    atomic_json(args.out/'routing-results.json',result)

def corrected(pred,wm,alpha=1.,cap=1e9):
    delta=(pred-wm)*alpha;scale=np.minimum(1.,cap/np.maximum(np.linalg.norm(delta,axis=1),1e-12))
    return wm+delta*scale[:,None]

def blend(args):
    requirements=args.out/'study-requirements.json'
    if requirements.exists():
        expected=json.loads(requirements.read_text())['models']
        while not all((args.out/(n+'-selectors.json')).exists() for n in expected):
            print('Waiting for prespecified additional model selectors before freezing:',[n for n in expected if not (args.out/(n+'-selectors.json')).exists()],flush=True)
            time.sleep(30)
        aggregate(args)
    routing=json.loads((args.out/'routing-results.json').read_text())
    # Only validation-selected models enter the finer blending grid.
    names=sorted(set(v['model'] for v in routing['winners'].values())|{'wide128'})
    cal=load_data(args.out,'calibration');test=load_data(args.out,'test');cb=np.linalg.norm(cal['wm']-cal['y'],axis=1)
    results={};globalbest={}
    for name in names:
        cp=np.load(args.out/(name+'-calibration-pred.npy'));tp=np.load(args.out/(name+'-test-pred.npy'))
        cf=features(cal,cp);tf=features(test,tp)
        models=pickle.load((args.out/(name+'-selectors.pkl')).open('rb'));route=routing['models'][name]
        # Keep one learned score per objective, frozen before testing blended rules.
        gate_names=sorted(set(v['candidate'] for v in route['selected'].values() if v['candidate'] in models))
        scores={'always':(np.ones(len(cp)),np.ones(len(tp)),False)}
        for gate in gate_names:
            m=models[gate];scores[gate]=(score_model(m['model'],cf[int(m['strong'])],m['classification']),score_model(m['model'],tf[int(m['strong'])],m['classification']),m['classification'])
        for alpha in [.25,.5,.75,1.]:
            choices=[(cap,None) for cap in [.25,.5,1.,2.,4.,1e9]]
            if alpha in [.5,1.]:choices += [(cap,rate) for cap in [1.,2.,4.] for rate in [.1,.25,.5,1.]]
            for cap,rate in choices:
                ccap=cap if rate is None else np.minimum(cap,rate*cal['age'])
                tcap=cap if rate is None else np.minimum(cap,rate*test['age'])
                cc=corrected(cp,cal['wm'],alpha,ccap);tc=corrected(tp,test['wm'],alpha,tcap)
                ce=np.linalg.norm(cc-cal['y'],axis=1);te=np.linalg.norm(tc-test['y'],axis=1)
                for gate,(cs,ts,classification) in scores.items():
                    policies=tune(cs,cb-ce,cal['count']>0,cf[0][:,8]<50,ce,cb,classification)
                    for mode,r in policies.items():
                        if mode not in globalbest or r['utility']>globalbest[mode]['tuning']['utility']:
                            a=(ts>r['threshold'])&(test['count']>0);b=np.linalg.norm(test['wm']-test['y'],axis=1)
                            globalbest[mode]=dict(model=name,candidate=gate,alpha=alpha,cap_m=cap,cap_per_cycle=rate,tuning=r,
                                 metrics=groups(np.where(a,te,b),test,a))
                        if mode=='mean':
                            for safe,limit in [('cap25',.25),('cap50',.5)]:
                                if cap<=limit and (safe not in globalbest or r['utility']>globalbest[safe]['tuning']['utility']):
                                    a=(ts>r['threshold'])&(test['count']>0);b=np.linalg.norm(test['wm']-test['y'],axis=1)
                                    globalbest[safe]=dict(model=name,candidate=gate,alpha=alpha,cap_m=cap,cap_per_cycle=rate,tuning=r,
                                         metrics=groups(np.where(a,te,b),test,a))
        print('Blending searched',name,flush=True)
    atomic_json(args.out/'blend-results.json',dict(winners=globalbest,model_grid=names,alphas=[.25,.5,.75,1.],caps_m=[.25,.5,1.,2.,4.,1e9],adaptive_caps=dict(alpha=[.5,1.],maximum_m=[1.,2.,4.],meters_per_named_age_cycle=[.1,.25,.5,1.]),note='Validation-tuned blending of frozen predictors and scores; no refitting to test. A cap bounds added distance error per replacement, not total error.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','aggregate','blend']);p.add_argument('out',type=Path);p.add_argument('--model',default='wide128');a=p.parse_args()
    if a.action=='fit':fit_model(a,a.model)
    elif a.action=='aggregate':aggregate(a)
    else:blend(a)
