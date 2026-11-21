"""Collapse equal-decision subtrees at each fixed validation-chosen threshold."""
from pathlib import Path
import argparse,json,pickle
import numpy as np
from refinement_models import atomic_json
from refinement_select import features,load_data
from prepare import FEATURES
from selector_study import EXTRA_NAMES

NAMES=FEATURES+EXTRA_NAMES+[f'unum_{i}' for i in range(1,12)]+['wm_observer_distance','wm_ball_distance','ball_known_again']

def prune(model,cut,node=0):
    tree=model.tree_
    if tree.children_left[node]<0:
        values=tree.value[node].ravel();prob=float(values[1]/values.sum())
        return {'use_model':prob>cut}
    left=prune(model,cut,int(tree.children_left[node]));right=prune(model,cut,int(tree.children_right[node]))
    if 'use_model'in left and left==right:return left
    return dict(feature=int(tree.feature[node]),name=NAMES[tree.feature[node]],threshold=float(tree.threshold[node]),left=left,right=right)

def predict(tree,x):
    if 'use_model'in tree:return np.full(len(x),tree['use_model'],bool)
    yes=x[:,tree['feature']]<=tree['threshold'];result=np.empty(len(x),bool)
    result[yes]=predict(tree['left'],x[yes]);result[~yes]=predict(tree['right'],x[~yes]);return result

def count(tree):return 1 if 'use_model'in tree else 1+count(tree['left'])+count(tree['right'])

def pretty(tree,indent=''):
    if 'use_model'in tree:return indent+('USE MODEL' if tree['use_model'] else 'KEEP WM')
    return indent+f'if {tree["name"]} <= {tree["threshold"]:.6g}:\n'+pretty(tree['left'],indent+'  ')+'\n'+indent+'else:\n'+pretty(tree['right'],indent+'  ')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
    models=pickle.load((a.out/'tree-controls.pkl').open('rb'));controls=json.loads((a.out/'tree-controls.json').read_text())
    d=load_data(a.out,'calibration');export={}
    for name,key,objective in [('continue128','tree74_cost3','mean'),('wmresidual128_l1','tree74_cost5','near50_risk')]:
        original=models[name+'/'+key];cut=controls['models'][name][key]['policies'][objective]['tuning']['threshold'];tree=prune(original,cut)
        x=features(d,np.load(a.out/(name+'-calibration-pred.npy')))[1]
        expected=original.predict_proba(x)[:,1]>cut;actual=predict(tree,x)
        assert np.array_equal(actual,expected),'Pruning changed a decision'
        export[name]=dict(selector=key,threshold=cut,original_nodes=original.tree_.node_count,pruned_nodes=count(tree),
            validation_equivalence_rows=len(x),mismatches=int((actual!=expected).sum()),tree=tree,text=pretty(tree),
            runtime_guard='Keep WM for posCount0, fresh named sighting, invalid or unknown target, missing history, age outside1–30 or reset boundary.',
            numeric_contract='Use float32 feature values and full-precision stored double split thresholds. Do not replace thresholds by their rounded display values.')
        print(name,'pruned',original.tree_.node_count,'to',count(tree),'nodes; exact on',len(x),'calibration rows',flush=True)
    atomic_json(a.out/'pruned-trees.json',export)
