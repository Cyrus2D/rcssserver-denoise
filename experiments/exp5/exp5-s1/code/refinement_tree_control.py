"""Exploratory same-input selector controls; original validation/test only."""
from pathlib import Path
import argparse,pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,export_text
from sklearn.ensemble import HistGradientBoostingClassifier
from refinement_models import atomic_json
from refinement_select import load_data,features,tune,score_model,candidate_record,BASIC_NAMES
from prepare import FEATURES
from selector_study import EXTRA_NAMES

def main(a):
    datasets={s:load_data(a.out,s) for s in ['selection','calibration','test']};result={};models={}
    ix=np.sort(np.random.default_rng(50571).choice(500000,250000,replace=False))
    names=FEATURES+EXTRA_NAMES+[f'unum_{i}' for i in range(1,12)]+['wm_observer_distance','wm_ball_distance','ball_known_again']
    for name in ['continue128','wmresidual128_l1']:
        pp={s:np.load(a.out/(name+'-'+s+'-pred.npy')) for s in datasets}
        ff={s:features(d,pp[s]) for s,d in datasets.items()}
        ee={s:np.linalg.norm(pp[s]-d['y'],axis=1) for s,d in datasets.items()}
        bb={s:np.linalg.norm(d['wm']-d['y'],axis=1) for s,d in datasets.items()};gain=bb['selection']-ee['selection']
        records={}
        specs=[('tree74_cost',3,True,1),('tree74_cost',5,True,1),('tree74_gain',3,False,1),('tree74_gain',5,False,1),('boost14_cost',0,True,0)]
        for kind,depth,classification,which in specs:
            key=kind+(str(depth) if depth else '')
            if depth:
                opts=dict(max_depth=depth,min_samples_leaf=2000,random_state=50571)
                model=DecisionTreeClassifier(**opts) if classification else DecisionTreeRegressor(**opts)
            else:model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.08,min_samples_leaf=200,l2_regularization=2.,early_stopping=False,random_state=50571)
            if classification:model.fit(ff['selection'][which][ix],gain[ix]>0,sample_weight=np.abs(gain[ix]))
            else:model.fit(ff['selection'][which][ix],gain[ix])
            cs=score_model(model,ff['calibration'][which],classification);ts=score_model(model,ff['test'][which],classification)
            c=datasets['calibration'];policies=tune(cs,bb['calibration']-ee['calibration'],c['count']>0,ff['calibration'][0][:,8]<50,ee['calibration'],bb['calibration'],classification)
            r=candidate_record(key,policies,ts,ee['test'],datasets['test']);r['features']=74 if which else 14
            if depth:r.update(nodes=model.tree_.node_count,tree_text=export_text(model,feature_names=names,decimals=3))
            else:r.update(trees=180)
            records[key]=r;models[name+'/'+key]=model
            print('Controlled selector',name,key,flush=True)
        result[name]=records
        atomic_json(a.out/'tree-controls.json',dict(models=result,note='Additional exploratory same-input controls on the original common1M test subset. Fit and thresholds use only old validation games. Added after primary new-game confirmation; not used to retune frozen policies or claimed as new independent confirmation.'))
    with (a.out/'tree-controls.pkl').open('wb') as f:pickle.dump(models,f)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);main(p.parse_args())
