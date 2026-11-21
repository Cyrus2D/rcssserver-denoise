"""RAM-backed causal windows: padding never reads across a PlayOn boundary."""
import json
from pathlib import Path
import numpy as np
import torch

class Data:
    def __init__(self,root,splits=None):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        obs=[r for r in self.manifest['observers'] if not splits or r['split'] in splits]
        self.feature_names=self.manifest['features'];self.rows=sum(r['rows']*21 for r in obs)
        self.x=np.empty((self.rows,len(self.feature_names)),np.float32)
        self.y=np.empty((self.rows,2),np.float32);self.wm=np.empty_like(self.y)
        self.age=np.empty(self.rows,np.uint16);self.flags=np.empty(self.rows,np.uint8)
        self.count=np.empty(self.rows,np.uint8);self.unum=np.empty(self.rows,np.uint8)
        self.start=np.empty(self.rows,np.uint32);self.distance=np.empty(self.rows,np.float32)
        self.game=np.empty(self.rows,np.uint16);self.side=np.empty(self.rows,np.uint8)
        self.split_indices={k:[] for k in ['train','selection','calibration','test']}
        val=sorted(g for g,s in self.manifest['split'].items() if s=='validation')
        self.calibration_games=val[::2];self.selection_games=val[1::2]
        self.games=sorted({r['game'] for r in obs});offset=0
        for n,r in enumerate(obs):
            path=Path(r['path']);t=r['rows'];size=t*21;sl=slice(offset,offset+size)
            for name in ['x','y','wm','age','flags','count','distance']:
                a=np.load(path/(name+'.npy'));getattr(self,name)[sl]=a.reshape((size,)+a.shape[2:])
            ids=np.load(path/'ids.npy');self.unum[sl]=np.broadcast_to((ids-1)%11+1,(t,21)).ravel()
            blocks=np.load(path/'block.npy');starts=np.maximum.accumulate(np.where(np.r_[True,np.diff(blocks)!=0],np.arange(t),0))
            self.start[sl]=(offset+starts[:,None]*21+np.arange(21)[None,:]).ravel()
            self.game[sl]=self.games.index(r['game']);self.side[sl]=int(r['observer'].startswith('r'))
            eligible=((self.flags[sl]&3)==3)&((self.flags[sl]&4)==0)&(self.age[sl]>0)&(self.age[sl]<=30)
            split=r['split']
            if split=='validation':split='calibration' if r['game'] in self.calibration_games else 'selection'
            self.split_indices[split].append(offset+np.flatnonzero(eligible))
            offset+=size
            if (n+1)%220==0:print(f'Loaded {n+1}/{len(obs)} observers into RAM',flush=True)
        self.split_indices={k:np.concatenate(v) if v else np.empty(0,np.int64) for k,v in self.split_indices.items()}
        self.tx=torch.from_numpy(self.x)

    def batch(self,indices,history,device):
        indices=np.asarray(indices,np.int64)
        prev=indices[:,None]-np.arange(history-1,-1,-1)[None,:]*21
        padding=prev<self.start[indices,None]
        prev=np.maximum(prev,self.start[indices,None])
        x=self.x[prev].copy();x[padding]=0
        return (torch.from_numpy(x).to(device),torch.from_numpy(self.unum[indices].astype(np.int64)).to(device),
                torch.from_numpy(self.y[indices]).to(device))

def batches(indices,size):
    for begin in range(0,len(indices),size):yield indices[begin:begin+size]
