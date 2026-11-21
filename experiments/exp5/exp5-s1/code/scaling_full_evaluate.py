#!/usr/bin/env python3
"""Full old-test evaluation of the candidate chosen only by selection error."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from data import Data,batches
from scaling_study import create,summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('study',type=Path)
    args=p.parse_args();file=args.study/'results.json';d=json.loads(file.read_text())
    name=min(d['models'],key=lambda k:d['models'][k]['selection_mean_m']);record=d['models'][name]
    print('Frozen selection winner:',name,flush=True);torch.set_num_threads(8)
    device='cuda' if torch.cuda.is_available() else 'cpu';data=Data(args.dataset,['validation','test'])
    ids=data.split_indices['test'];stock=np.linalg.norm(data.wm[ids]-data.y[ids],axis=1)
    saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    model=create(len(data.feature_names),record['hidden'],record['layers']).to(device);model.load_state_dict(saved['state']);model.eval()
    error=np.empty(len(ids),np.float32);start=time.monotonic()
    with torch.inference_mode():
        for begin in range(0,len(ids),4096):
            end=min(begin+4096,len(ids));x,u,y=data.batch(ids[begin:end],32,device)
            error[begin:end]=(model(x,u)['point']-y).norm(dim=-1).cpu().numpy()
            if begin%(4096*500)==0:print('Full scaling test',end,'/',len(ids),'seconds',time.monotonic()-start,flush=True)
    np.save(args.study/(name+'-full-test-error.npy'),error)
    record['full_test']=summary(error,stock);record['full_test_seconds']=time.monotonic()-start
    record['full_test_already_accurate']=summary(error[stock<.1],stock[stock<.1])
    d['full_test_selection_winner']=name;file.write_text(json.dumps(d,indent=2)+'\n')
    print('Full scaling result',name,record['full_test'],flush=True)

if __name__=='__main__':main()
