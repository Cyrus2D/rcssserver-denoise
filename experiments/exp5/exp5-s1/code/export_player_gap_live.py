"""Export the frozen practical Exp5 model and its numeric boosted selector."""
from pathlib import Path
import argparse,hashlib,json,pickle,struct,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from refinement_models import get_model
from refinement_select import features,load_data
from data import Data

def main(a):
    frozen=json.loads((a.study/'frozen-policies.json').read_text())[a.policy]
    name=frozen['model'];checkpoint=a.study/name/'best.pt'
    if name=='wide128':checkpoint=a.study.parent/'scaling-study/wide128/best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    selector=pickle.load((a.study/(name+'-selectors.pkl')).open('rb'))[frozen['candidate']]['model']
    a.output.mkdir(parents=True,exist_ok=True);tensors={};offset=0
    with (a.output/'weights.bin').open('wb') as f:
        for key,value in state['state'].items():
            arr=value.detach().numpy().astype('<f4');tensors[key]=dict(shape=list(arr.shape),offset=offset,count=arr.size)
            f.write(arr.tobytes());offset+=arr.size
    counts=[]
    with (a.output/'selector.bin').open('wb') as f:
        for stage in selector._predictors:
            nodes=stage[0].nodes;counts.append(len(nodes))
            for n in nodes:
                assert not n['is_categorical']
                f.write(struct.pack('<ddiiiBB',float(n['num_threshold']),float(n['value']),int(n['feature_idx']),int(n['left']),int(n['right']),int(n['is_leaf']),int(n['missing_go_to_left'])))
    manifest=dict(format='rcsc-player-gap-v1',history=32,features=51,hidden=128,embedding=4,output=2,
       wm_residual=bool(state['record'].get('wm_residual')),threshold=frozen['tuning']['threshold'],
       baseline=float(selector._baseline_prediction[0,0]),trees=counts,tensors=tensors,float_count=offset,
       policy=a.policy,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
       guard='unquantized finite-view no-hear/no-fade PlayOn; preserve fresh observations; age1..30',
       weights_sha256=hashlib.sha256((a.output/'weights.bin').read_bytes()).hexdigest(),selector_sha256=hashlib.sha256((a.output/'selector.bin').read_bytes()).hexdigest())
    manifest['bundle_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if a.dataset:
        torch.set_num_threads(1);data=Data(a.dataset,['validation','test']);small=load_data(a.study,'calibration')
        pos=np.sort(np.random.default_rng(50701).choice(len(small['indices']),256,replace=False));ids=small['indices'][pos]
        x,u,y=data.batch(ids,32,'cpu');model=get_model(checkpoint,'cpu')
        with torch.inference_mode():point=model(x,u)['point'].numpy()
        stock=data.wm[ids];dd={k:v[pos] for k,v in small.items()};feat=features(dd,point)[1]
        scores=selector.predict_proba(feat)[:,1]
        with (a.output/'golden.bin').open('wb') as f:
            f.write(struct.pack('<I',len(ids)))
            for j,i in enumerate(ids):
                f.write(struct.pack('<ii',int(data.unum[i]),int(data.age[i])))
                f.write(x[j].numpy().astype('<f4').tobytes());f.write(stock[j].astype('<f4').tobytes())
                f.write(point[j].astype('<f4').tobytes());f.write(struct.pack('<d',scores[j]))
    print('Exported',a.output,'bundle',manifest['bundle_id'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('output',type=Path);p.add_argument('--dataset',type=Path);p.add_argument('--policy',default='near50_risk');main(p.parse_args())
