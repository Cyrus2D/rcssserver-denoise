#!/usr/bin/env python3
"""Turn correlated per-target regressions into observer-cycle/episode diagnostics."""
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import numpy as np
import pandas as pd

VERSION=1

def one(job):
    root,out=map(Path,job);metadata=sorted(root.glob('l_*/meta.json'))
    fp=hashlib.sha256(str(root.resolve()).encode())
    for f in metadata:
        fp.update(f.read_bytes());st=(f.parent/'player_prediction.csv.gz').stat();fp.update(f'{st.st_size}:{st.st_mtime_ns}'.encode())
    fingerprint=fp.hexdigest()
    if out.exists():
        old=json.loads(out.read_text())
        if old.get('version')==VERSION and old.get('fingerprint')==fingerprint:return old
    counts=Counter();max_episode=0
    for file in metadata:
        m=json.loads(file.read_text());assert m['final'] and m['status']=='ok'
        assert m['run_options']['player_gap_integration']=='cache-v2'
        a=pd.read_csv(file.parent/'agent_cycle.csv.gz',usecols=['cycle','stopped','decision_made','game_mode'])
        a=a[a.decision_made.eq(1)].copy()
        continuous=a.stopped.eq(0)&a.stopped.shift().eq(0)&a.cycle.diff().eq(1)&a.game_mode.eq('play_on')&a.game_mode.shift().eq('play_on')
        a['block']=(~continuous).cumsum()
        block=dict(zip(a.loc[a.stopped.eq(0),'cycle'],a.loc[a.stopped.eq(0),'block']))
        events={k:set() for k in ['eligible','applied','better25','worse25','worse50','worse100','worse25_near10','worse25_opponent_near10','worse25_good_wm10cm']}
        bad=[]
        for d in pd.read_csv(file.parent/'player_prediction.csv.gz',usecols=['cycle','stopped','object_id','eligible','applied','truth_valid','true_dist','stock_error','used_error'],chunksize=105000):
            q=d[d.eligible.eq(1)&d.truth_valid.eq(1)&d.true_dist.ge(0)&d.true_dist.lt(50)&d.stock_error.notna()&d.used_error.notna()]
            delta=q.used_error-q.stock_error
            for key,mask in [('eligible',np.ones(len(q),bool)),('applied',q.applied.eq(1)),('better25',delta.lt(-.25)),
                ('worse25',delta.gt(.25)),('worse50',delta.gt(.5)),('worse100',delta.gt(1)),
                ('worse25_near10',delta.gt(.25)&q.true_dist.lt(10)),
                ('worse25_opponent_near10',delta.gt(.25)&q.true_dist.lt(10)&q.object_id.ge(12)),
                ('worse25_good_wm10cm',delta.gt(.25)&q.stock_error.lt(.1))]:
                events[key].update(q.loc[mask,'cycle'].astype(int))
            take=q[delta.gt(1)]
            bad.extend((int(row.object_id),int(block[int(row.cycle)]),int(row.cycle)) for row in take.itertuples())
        for key,values in events.items():counts['observer_cycles_'+key]+=len(values)
        counts['observer_cycles_both_better25_and_worse25']+=len(events['better25']&events['worse25'])
        episode=[];previous=None;length=0
        for target,phase,cycle in sorted(bad):
            if previous and (target,phase)==previous[:2] and cycle==previous[2]+1:length+=1
            else:
                if length:episode.append(length)
                length=1
            previous=(target,phase,cycle)
        if length:episode.append(length)
        counts['target_episodes_added_over1m']+=len(episode)
        counts['target_episodes_added_over1m_at_least3cycles']+=sum(n>=3 for n in episode)
        max_episode=max(max_episode,max(episode,default=0))
    result=dict(version=VERSION,fingerprint=fingerprint,game=root.name,observers=len(metadata),counts=dict(counts),max_consecutive_cycles_added_over1m=max_episode)
    out.write_text(json.dumps(result,indent=2)+'\n');return result

def main(a):
    source=json.loads(a.games.read_text());ids=sorted(r['game'] for r in source['per_game'])
    a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'games';cache.mkdir(exist_ok=True)
    counts=Counter();records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,(str(a.logs/g),str(cache/(g+'.json')))) for g in ids]
        for f in as_completed(futures):
            r=f.result();records.append(r);counts.update(r['counts']);print('Completed risk games',len(records),'/',len(ids),flush=True)
    n=counts['observer_cycles_eligible']
    result=dict(games=len(records),observers=sum(r['observers'] for r in records),counts=dict(counts),
        percent_of_observer_cycles_with_eligible_stale_under50={k:v/n*100 for k,v in counts.items() if k.startswith('observer_cycles_')},
        max_consecutive_cycles_added_over1m=max(r['max_consecutive_cycles_added_over1m'] for r in records),
        note='A cycle may have both better and worse targets. These observer-cycle and episode counts are correlated diagnostic exposures, not independent errors or failed actions.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('Finished risk summary',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('logs',type=Path);p.add_argument('games',type=Path);p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=6);main(p.parse_args())
