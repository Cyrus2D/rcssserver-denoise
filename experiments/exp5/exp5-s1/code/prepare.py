#!/usr/bin/env python3
"""Causal, observation-only Exp5 player position dataset, all observers.

Truth and logger truth-derived diagnostics are stored only as labels.
Features use named visual localizations and the observer's current WM.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

VERSION=2
FEATURES='last_x last_y last_vx last_vy velocity_known velocity_age body_cos body_sin body_known body_age gap named_now mean_move_x mean_move_y mean_move_known unnamed_wm_refresh wm_dx wm_dy wm_pos_known wm_pos_age wm_vx wm_vy wm_vel_known wm_vel_age observer_dx observer_dy observer_face_cos observer_face_sin view_width see_used ball_dx ball_dy ball_vx ball_vy ball_pos_known ball_vel_known ball_pos_age teammate nearest_team_dx nearest_team_dy nearest_team_known nearby_team_count nearest_opp_dx nearest_opp_dy nearest_opp_known nearby_opp_count context_vx context_vy context_fraction closest_distance observer_pos_age'.split()
OBS_COLS='cycle stopped object_id seen unum_visible match_method ls_pos_x ls_pos_y ls_vel_x ls_vel_y dist_chg dir_chg body_rel track_match trk_pos_x trk_pos_y trk_vel_x trk_vel_y trk_pos_count trk_vel_count'.split()
LABEL_COLS='truth_valid tru_on_field tru_pos_x tru_pos_y true_dist n_trk_pos_d'.split()
AGENT_COLS='cycle stopped decision_made game_mode see_used sb_view_width view_width est_pos_x est_pos_y est_pos_count est_face'.split()

def last_indices(mask,block):
    t=np.arange(len(mask));last=np.maximum.accumulate(np.where(mask,t[:,None],-1),axis=0)
    starts=np.maximum.accumulate(np.where(np.r_[True,np.diff(block)!=0],t,0))
    return last,last>=starts[:,None]

def gather(a,idx,valid):
    out=a[np.maximum(idx,0),np.arange(a.shape[1])]
    return np.where(valid[...,None],out,0) if out.ndim==3 else np.where(valid,out,0)

def make_features(a,d,ids,side):
    """d contains ONLY OBS_COLS: changing truth cannot alter these features."""
    t=len(a);j=len(ids);sgn=1 if side=='l' else -1
    def arr(name):return d[name][:,ids]
    def vec(prefix):return np.stack([arr(prefix+'_x'),arr(prefix+'_y')],-1)*sgn
    cycle=a.cycle.to_numpy();block=a.block.to_numpy()
    pos=vec('ls_pos');direct=arr('seen')==1
    direct &= (arr('unum_visible')==1)&(arr('match_method')==1)&np.isfinite(pos).all(-1)
    last,history=last_indices(direct,block)
    anchor=gather(pos,last,history);age=np.where(history,cycle[:,None]-cycle[np.maximum(last,0)],0)
    previous=np.r_[np.full((1,j),-1),last[:-1]]
    prevok=direct&(previous>=0)&(a.block.to_numpy()[:,None]==block[np.maximum(previous,0)])
    prevpos=gather(pos,previous,prevok)
    dt=np.maximum(1,cycle[:,None]-cycle[np.maximum(previous,0)])
    move=(pos-prevpos)/dt[...,None]
    avg=gather(np.nan_to_num(move),last,history)
    avgok=gather(prevok,last,history)
    vel=vec('ls_vel')
    hasvel=direct&np.isfinite(vel).all(-1)&np.isfinite(arr('dist_chg'))&np.isfinite(arr('dir_chg'))
    vi,vok=last_indices(hasvel,block);v=gather(vel,vi,vok)
    # A fallback from two actual sightings is available to a live observer.
    v=np.where(vok[...,None],v,np.where(avgok[...,None],avg,0))
    vage=np.where(vok,cycle[:,None]-cycle[np.maximum(vi,0)],60)
    body=np.deg2rad(arr('body_rel')+a.est_face.to_numpy()[:,None])
    bi,bok=last_indices(direct&np.isfinite(body),block)
    bc=gather(np.cos(body)*sgn,bi,bok);bs=gather(np.sin(body)*sgn,bi,bok)
    bage=np.where(bok,cycle[:,None]-cycle[np.maximum(bi,0)],60)
    wm=vec('trk_pos');wv=vec('trk_vel');namedtrack=arr('track_match')==1
    wpok=namedtrack&np.isfinite(wm).all(-1);wvok=namedtrack&np.isfinite(wv).all(-1)
    base=v*age[...,None];center=anchor+base
    selfpos=a[['est_pos_x','est_pos_y']].to_numpy()*sgn
    face=np.deg2rad(a.est_face.to_numpy());width=a.sb_view_width.fillna(a.view_width).map(dict(narrow=60,normal=120,wide=180)).fillna(120).to_numpy()
    ball=np.stack([d['trk_pos_x'][:,0],d['trk_pos_y'][:,0]],-1)*sgn
    ballv=np.stack([d['trk_vel_x'][:,0],d['trk_vel_y'][:,0]],-1)*sgn
    bpok=np.isfinite(ball).all(-1);bvok=np.isfinite(ballv).all(-1)
    team=(ids<=11) if side=='l' else (ids>=12)
    diff=wm[:,None,:,:]-center[:,:,None,:]
    dist=np.linalg.norm(diff,axis=-1)
    available=np.broadcast_to(wpok[:,None,:],dist.shape).copy()
    available &= ~np.eye(j,dtype=bool)[None]
    nearby=available&(dist<10)
    near=[]
    for same in [True,False]:
        candidate=available&((team[None,:,None]==team[None,None,:])==same)
        # Group relative to observer team, not target team, for stable semantics.
        candidate=available&np.broadcast_to((team if same else ~team)[None,None,:],dist.shape)
        score=np.where(candidate,dist,np.inf);ni=score.argmin(-1);ok=np.isfinite(score.min(-1))
        relative=diff[np.arange(t)[:,None],np.arange(j)[None,:],ni]
        near.extend([np.where(ok,relative[...,0]/20,0),np.where(ok,relative[...,1]/20,0),ok,
                     (nearby&candidate).sum(-1)/10])
    vm=nearby&wvok[:,None,:];nv=vm.sum(-1)
    pooled=(np.where(vm[...,None],np.nan_to_num(wv)[:,None,:,:],0)).sum(2)/np.maximum(nv,1)[...,None]
    def broad(x):return np.broadcast_to(x[:,None],(t,j))
    def broadvec(x):return np.broadcast_to(x[:,None,:],(t,j,2))
    observer=broadvec(selfpos)-anchor
    ballrel=broadvec(ball)-anchor
    wmrel=wm-center
    values=[anchor[...,0]/52.5,anchor[...,1]/34,v[...,0]/1.5,v[...,1]/1.5,vok,vage/30,
            bc,bs,bok,bage/30,age/30,direct,avg[...,0]/1.5,avg[...,1]/1.5,avgok,
            (arr('trk_pos_count')==0)&~direct&wpok,
            np.where(wpok,wmrel[...,0]/10,0),np.where(wpok,wmrel[...,1]/10,0),wpok,
            np.where(wpok,arr('trk_pos_count')/30,1),np.where(wvok,wv[...,0]/1.5,0),np.where(wvok,wv[...,1]/1.5,0),wvok,
            np.where(wvok,arr('trk_vel_count')/10,1),observer[...,0]/50,observer[...,1]/50,
            broad(np.cos(face)*sgn),broad(np.sin(face)*sgn),broad(width/180),broad(a.see_used.to_numpy()),
            np.where(broad(bpok),ballrel[...,0]/30,0),np.where(broad(bpok),ballrel[...,1]/30,0),
            broad(np.where(bvok,ballv[...,0]/3,0)),broad(np.where(bvok,ballv[...,1]/3,0)),broad(bpok),broad(bvok),
            broad(np.nan_to_num(d['trk_pos_count'][:,0],nan=30)/30),np.broadcast_to(team,(t,j)),*near,
            pooled[...,0]/1.5,pooled[...,1]/1.5,available.sum(-1)/20,
            np.minimum(np.where(available,dist,np.inf).min(-1),100)/20,
            broad(a.est_pos_count.fillna(30).to_numpy()/3)]
    x=np.nan_to_num(np.stack(values,-1),nan=0,posinf=5,neginf=-5).astype('float32')
    assert x.shape[-1]==len(FEATURES)
    return x,anchor,base,age,history,direct,wpok,wm

def build_observer(path,out):
    meta=json.loads((path/'meta.json').read_text());sp=meta['server_param']
    assert meta['final'] and meta['status']=='ok' and not meta['config']['use_fullstate']
    assert sp['see_no_quantization']=='true' and sp['full_view']=='false' and sp['say_msg_size']=='0'
    fingerprint=hashlib.sha256((str(VERSION)+(path/'meta.json').read_text()).encode()).hexdigest()
    dest=out/path.parent.name/path.name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'meta.json').exists():
        old=json.loads((dest/'meta.json').read_text())
        if old.get('fingerprint')==fingerprint:return old
    a=pd.read_csv(path/'agent_cycle.csv.gz',usecols=AGENT_COLS)
    a=a[a.decision_made.eq(1)].copy()
    continuous=a.stopped.eq(0)&a.stopped.shift().eq(0)&a.cycle.diff().eq(1)&a.game_mode.eq('play_on')&a.game_mode.shift().eq('play_on')
    a['block']=(~continuous).cumsum()
    a=a[a.stopped.eq(0)&a.game_mode.eq('play_on')].reset_index(drop=True)
    if a.duplicated(['cycle','stopped']).any():raise ValueError('Duplicate agent keys')
    # Schema 4 decision overlays must never feed back into model inputs.
    # Stock columns apply to player rows only; the ball has no overlay.
    stock=['trk_stock_pos_x','trk_stock_pos_y','trk_stock_pos_count']
    header=pd.read_csv(path/'objects.csv.gz',nrows=0).columns
    extra=stock if all(k in header for k in stock) else []
    o=pd.read_csv(path/'objects.csv.gz',usecols=OBS_COLS+LABEL_COLS+extra,low_memory=False)
    if extra:
        for key in ['pos_x','pos_y','pos_count']:
            o.loc[o.object_id.gt(0),'trk_'+key]=o.loc[o.object_id.gt(0),'trk_stock_'+key]
    o=o[o.stopped.eq(0)].merge(a[['cycle']].assign(row=np.arange(len(a))),on='cycle',validate='many_to_one')
    r=o.row.to_numpy(int);oid=o.object_id.to_numpy(int);d={}
    for name in OBS_COLS[3:]+LABEL_COLS:
        values=o[name]
        if name in ['match_method','track_match']:values=values.eq('unum').astype(float)
        grid=np.full((len(a),23),np.nan,np.float32);grid[r,oid]=values.to_numpy(float);d[name]=grid
    ownid=int(meta['unum'])+(0 if meta['side']=='l' else 11)
    ids=np.array([i for i in range(1,23) if i!=ownid])
    # This explicit whitelist enforces a one-way truth/feature boundary.
    inputs={k:d[k] for k in OBS_COLS[3:]}
    x,anchor,base,age,history,direct,wpok,wm=make_features(a,inputs,ids,meta['side'])
    sgn=1 if meta['side']=='l' else -1
    truth=np.stack([d['tru_pos_x'][:,ids],d['tru_pos_y'][:,ids]],-1)*sgn
    valid=history&(d['truth_valid'][:,ids]==1)&(d['tru_on_field'][:,ids]==1)&np.isfinite(truth).all(-1)
    flags=(valid.astype(np.uint8)|wpok.astype(np.uint8)*2|direct.astype(np.uint8)*4|
           (d['seen'][:,ids]==1).astype(np.uint8)*8)
    arrays=dict(x=x,y=np.nan_to_num(truth-anchor).astype('float32'),wm=np.nan_to_num(wm-anchor).astype('float32'),
                age=np.minimum(age,65535).astype('uint16'),flags=flags,block=a.block.to_numpy(np.int32),
                cycle=a.cycle.to_numpy(np.int16),ids=ids.astype(np.uint8),
                count=np.nan_to_num(d['trk_pos_count'][:,ids],nan=255).clip(0,255).astype(np.uint8),
                distance=np.nan_to_num(d['true_dist'][:,ids],nan=-1).astype('float32'))
    for name,value in arrays.items():np.save(dest/(name+'.npy'),value)
    result=dict(game=path.parent.name,observer=path.name,fingerprint=fingerprint,rows=len(a),
                path=str(dest.resolve()),eligible=int((valid&~direct&(age>0)&(age<=30)&wpok).sum()),
                fresh=int((valid&direct).sum()),feature_count=len(FEATURES))
    temp=dest/'meta.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(dest/'meta.json')
    return result

def job(args):return build_observer(*map(Path,args))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('logs',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    games=[]
    for g in sorted(args.logs.iterdir()):
        files=list(g.glob('*/meta.json'))
        if len(files)==22 and all((m:=json.loads(f.read_text())).get('final') and m.get('status')=='ok' for f in files):games.append(g)
    rng=np.random.default_rng(50501);order=rng.permutation(len(games));n=len(games)
    split={games[i].name:('train' if k<int(n*.8) else 'validation' if k<int(n*.9) else 'test') for k,i in enumerate(order)}
    if args.limit:games=games[:args.limit]
    jobs=[(str(m.parent),str(args.output)) for g in games for m in sorted(g.glob('*/meta.json'))]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(job,jobs),1):
            r['split']=split[r['game']];results.append(r)
            if i%22==0:print(f'Prepared {i//22}/{len(games)} games ({i} observers)',flush=True)
    manifest=dict(version=VERSION,logs=str(args.logs.resolve()),features=FEATURES,observation_columns=OBS_COLS,
                  label_columns=LABEL_COLS,split_seed=50501,split=split,max_gap=30,observers=results)
    if games:
        first=json.loads(next(games[0].glob('*/meta.json')).read_text())
        manifest['server_default_player_decay']=float(first['server_param']['player_decay'])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Dataset complete:',len(results),'observers;',sum(r['eligible'] for r in results),'paired gap samples',flush=True)

if __name__=='__main__':main()
