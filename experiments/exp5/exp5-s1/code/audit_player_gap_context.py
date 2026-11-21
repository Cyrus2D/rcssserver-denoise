#!/usr/bin/env python3
"""Label-only ball proximity and one-step diagnostics on a fixed game sample."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd

class Stats:
    def __init__(self):
        self.n=0;self.stock=0.;self.used=0.;self.max_added=-float('inf');self.better=0;self.worse={k:0 for k in [0,1,5,10,25,50,100]}
    def add(self,stock,used):
        stock=np.asarray(stock);used=np.asarray(used);ok=np.isfinite(stock)&np.isfinite(used)
        stock=stock[ok];used=used[ok];delta=(used-stock)*100
        if not len(stock):return
        self.n+=len(stock);self.stock+=float(stock.sum());self.used+=float(used.sum());self.better+=int((delta < -1).sum())
        self.max_added=max(self.max_added,float(delta.max()))
        for k in self.worse:self.worse[k]+=int((delta>k).sum())
    def report(self):
        if not self.n:return None
        return dict(samples=self.n,stock_mean_cm=self.stock/self.n*100,used_mean_cm=self.used/self.n*100,
                    gain_percent=(1-self.used/self.stock)*100 if self.stock else None,
                    better_1cm_percent=self.better/self.n*100,max_added_cm=self.max_added,
                    **{f'added_over_{k}cm_percent':v/self.n*100 for k,v in self.worse.items()})

def one(path):
    path=Path(path);a=pd.read_csv(path/'agent_cycle.csv.gz',usecols=['cycle','stopped','game_mode','decision_made','tru_pos_x','tru_pos_y'])
    cols=['cycle','stopped','object_id','tru_pos_x','tru_pos_y','trk_vel_x','trk_vel_y']
    o=pd.read_csv(path/'objects.csv.gz',usecols=cols)
    b=o[o.object_id.eq(0)][['cycle','stopped','tru_pos_x','tru_pos_y']].rename(columns={'tru_pos_x':'ball_x','tru_pos_y':'ball_y'})
    # The logger omits the observer's own row in objects; self truth is in agent_cycle.
    observer=a.rename(columns={'tru_pos_x':'observer_x','tru_pos_y':'observer_y'})
    labels=b.merge(observer,on=['cycle','stopped'],validate='one_to_one')
    # Current and next-cycle truth are labels only, never model inputs.
    nxt=o[o.stopped.eq(0)][['cycle','object_id','tru_pos_x','tru_pos_y']].copy();nxt.cycle-=1
    nxt=nxt.rename(columns={'tru_pos_x':'next_x','tru_pos_y':'next_y'})
    play=a[a.stopped.eq(0)&a.game_mode.eq('play_on')&a.decision_made.eq(1)][['cycle']]
    nxt=nxt.merge(play,on='cycle',validate='many_to_one')
    next_play=play.copy();next_play.cycle-=1
    nxt=nxt.merge(next_play,on='cycle',validate='many_to_one')
    stats=defaultdict(Stats)
    cols=['cycle','stopped','object_id','eligible','selected','applied','true_dist','stock_error','used_error','truth_valid',
          'stock_x','stock_y','used_x','used_y','true_x','true_y','named_age','stock_count','score']
    worst_good=None
    for d in pd.read_csv(path/'player_prediction.csv.gz',usecols=cols,chunksize=105000):
        q=d[d.eligible.eq(1)&d.truth_valid.eq(1)&d.true_dist.lt(50)].copy()
        if not len(q):continue
        q=q.merge(labels,on=['cycle','stopped'],validate='many_to_one')
        target_ball=np.hypot(q.true_x-q.ball_x,q.true_y-q.ball_y)
        observer_ball=np.hypot(q.observer_x-q.ball_x,q.observer_y-q.ball_y)
        for k,mask in [('target_ball_under2m',target_ball<2),('target_ball_2to5m',(target_ball>=2)&(target_ball<5)),
                       ('target_ball_5mplus',target_ball>=5),('observer_ball_under3m',observer_ball<3),('observer_ball_3mplus',observer_ball>=3)]:
            stats[k].add(q.loc[mask,'stock_error'],q.loc[mask,'used_error'])
        v=q.merge(o[['cycle','stopped','object_id','trk_vel_x','trk_vel_y']],on=['cycle','stopped','object_id'],validate='one_to_one')
        v=v[v.stopped.eq(0)].merge(nxt,on=['cycle','object_id'],validate='one_to_one')
        stock=np.hypot(v.stock_x+v.trk_vel_x-v.next_x,v.stock_y+v.trk_vel_y-v.next_y)
        used=np.hypot(v.used_x+v.trk_vel_x-v.next_x,v.used_y+v.trk_vel_y-v.next_y)
        stats['next_cycle_using_unchanged_valid_wm_velocity'].add(stock,used)
        good=q[q.stock_error.lt(.1)]
        if len(good):
            delta=good.used_error-good.stock_error;j=delta.idxmax();row=good.loc[j]
            if worst_good is None or float(delta.loc[j])>worst_good['added_error_m']:
                worst_good=dict(game=path.parent.name,observer=path.name,cycle=int(row.cycle),object_id=int(row.object_id),
                                stock_error_m=float(row.stock_error),used_error_m=float(row.used_error),
                                named_age=int(row.named_age),stock_count=int(row.stock_count),score=float(row.score),
                                observer_distance_m=float(row.true_dist),target_ball_m=float(target_ball.loc[j]),
                                added_error_m=float(delta.loc[j]))
    return dict(version=2,path=str(path),stats={k:v.report() for k,v in stats.items() if v.n},worst_accurate_wm=worst_good)

def main(a):
    summary=json.loads(a.batch.read_text());games=sorted(x['game'] for x in summary['per_game'])
    chosen=[games[i] for i in np.linspace(0,len(games)-1,min(a.games,len(games))).astype(int)]
    a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'observers';cache.mkdir(exist_ok=True)
    records=[];jobs=[]
    for game in chosen:
        for p in sorted((a.logs/game).glob('l_*/meta.json')):
            dest=cache/(game+'-'+p.parent.name+'.json')
            if dest.exists() and (cached:=json.loads(dest.read_text())).get('version')==2:records.append(cached)
            else:jobs.append((p.parent,dest))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,p):dest for p,dest in jobs}
        for f in as_completed(futures):
            r=f.result();futures[f].write_text(json.dumps(r,indent=2)+'\n');records.append(r)
            print('Completed observers',len(records),'/',len(chosen)*11,flush=True)
    from audit_player_gap_batch import combine_metrics
    labels={k for r in records for k in r['stats']}
    result=dict(games=chosen,observers=len(records),selection='Ten evenly spaced sorted complete game IDs, fixed without consulting outcomes; descriptive label-only subgroups.',
                stats={k:combine_metrics([r['stats'][k] for r in records if k in r['stats']]) for k in sorted(labels)},
                worst_accurate_wm=max((r['worst_accurate_wm'] for r in records if r['worst_accurate_wm']),key=lambda w:w['added_error_m'],default=None))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Finished',a.output/'summary.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('logs',type=Path);p.add_argument('batch',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--games',type=int,default=10);p.add_argument('--workers',type=int,default=4);main(p.parse_args())
