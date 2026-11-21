#!/usr/bin/env python3
"""Parallel full-game live audit; exact sample-weighted means and tail rates.

Per-game percentiles are retained separately and never averaged as a global
percentile. Cached summaries make interrupted audits resumable.
"""
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from collections import Counter
import json
import hashlib
import math
import os
from summarize_player_gap_live import summarize

VERSION=2

def one(job):
    source,cache,max_distance=job
    source=Path(source);cache=Path(cache)
    fingerprint=hashlib.sha256(str(source.resolve()).encode())
    for meta in sorted(source.glob('*/meta.json')):
        fingerprint.update(meta.read_bytes())
        for name in ['player_prediction.csv.gz','agent_cycle.csv.gz']:
            st=(meta.parent/name).stat()
            fingerprint.update(f'{meta.parent.name}/{name}:{st.st_size}:{st.st_mtime_ns}'.encode())
    fingerprint=fingerprint.hexdigest()
    if cache.exists():
        existing=json.loads(cache.read_text())
        if existing.get('audit_version')==VERSION and existing.get('max_true_distance_m')==max_distance and existing.get('source_fingerprint')==fingerprint:return existing
    with open(os.devnull,'w') as sink,contextlib.redirect_stderr(sink):
        result=summarize(source,max_distance,True)
    m=json.loads((source/'l_1/meta.json').read_text())
    result.update(audit_version=VERSION,source_fingerprint=fingerprint,source_path=str(source.resolve()),game=source.name,score_left=m['score_left'],score_right=m['score_right'])
    tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(cache)
    return result

def combine_metrics(rows):
    n=sum(v['samples'] for v in rows)
    if not n:return None
    weighted=lambda key:sum(v[key]*v['samples'] for v in rows)/n
    stock=weighted('stock_mean_cm');used=weighted('used_mean_cm')
    return dict(samples=n,stock_mean_cm=stock,used_mean_cm=used,
                gain_percent=(1-used/stock)*100 if stock else None,
                better_1cm_percent=weighted('better_1cm_percent'),
                **{f'added_over_{k}cm_percent':weighted(f'added_over_{k}cm_percent') for k in [0,1,5,10,25,50,100]},
                max_added_cm=max(v['max_added_cm'] for v in rows))

def aggregate(records,distance):
    config=[];checks=Counter();cycles=Counter();groups={}
    for r in records:
        config.extend(r['config_issues']);checks.update(r['checks']);cycles.update(r['cycles'])
        for g in r['groups']:
            key=(g['mode'],g['count_policy'],g['side'])
            groups.setdefault(key,[]).append((r,g))
    result=dict(audit_version=VERSION,games=len(records),max_true_distance_m=distance,
                observers=sum(g['observers'] for r in records for g in r['groups']),
                config_issues=config,checks=dict(checks),cycles=dict(cycles),groups=[],
                score=dict(left=sum(r['score_left'] for r in records),right=sum(r['score_right'] for r in records),
                           wins=sum(r['score_left']>r['score_right'] for r in records),
                           draws=sum(r['score_left']==r['score_right'] for r in records),
                           losses=sum(r['score_left']<r['score_right'] for r in records)),
                per_game=[],note='Means and tail rates give each target-cycle equal weight; global p95 is intentionally not reconstructed from per-game percentiles.')
    for key,entries in sorted(groups.items()):
        gs=[g for _,g in entries]
        out=dict(mode=key[0],count_policy=key[1],side=key[2],games=len(gs),
                 observers=sum(g['observers'] for g in gs),applied_all=sum(g['applied_all'] for g in gs),
                 eligible_all=sum(g['eligible_all'] for g in gs),statuses=dict(sum((Counter(g['statuses']) for g in gs),Counter())),
                 bundles=sorted({x for g in gs for x in g['bundles']}),
                 view_policies=sorted({x for g in gs for x in g['view_policies']}),
                 integration_versions=sorted({x for g in gs for x in g.get('integration_versions',['legacy-unspecified'])}))
        for key2 in ['actually_used','proposed']:
            rows=[g[key2] for g in gs if key2 in g]
            if rows:out[key2]=combine_metrics(rows)
        for key2 in ['by_named_age','by_stock_count','by_distance','by_target_team','by_original_error']:
            labels={x for g in gs for x in g.get(key2,{})}
            out[key2]={label:combine_metrics([g[key2][label] for g in gs if label in g.get(key2,{})]) for label in sorted(labels)}
        if 'actually_used' in out:
            out['applied_under_distance_filter']=sum(g.get('applied_under_distance_filter',0) for g in gs)
            out['selected_stock_count_thresholds']={str(k):sum(g['selected_stock_count_thresholds'][str(k)] for g in gs) for k in [4,6,11,15]}
            out['worst']=max((g['worst'] for g in gs if g['worst']),key=lambda w:w['added_error_m'])
            out['games_with_lower_mean']=sum(g['actually_used']['used_mean_cm']<g['actually_used']['stock_mean_cm'] for g in gs)
            for r,g in entries:
                result['per_game'].append(dict(game=r['game'],left=r['score_left'],right=r['score_right'],
                    **g['actually_used'],applied_under_distance_filter=g['applied_under_distance_filter']))
        times=[g['inference_ms'] for g in gs if 'inference_ms' in g]
        if times:out['timing_per_game']=dict(median_range_ms=[min(t['median'] for t in times),max(t['median'] for t in times)],
            p95_range_ms=[min(t['p95'] for t in times),max(t['p95'] for t in times)],maximum_ms=max(t['maximum'] for t in times))
        result['groups'].append(out)
    n=result['score']['wins']+result['score']['losses'];w=result['score']['wins']
    if n:
        p=w/n;z=1.95996398454;den=1+z*z/n;center=(p+z*z/(2*n))/den
        margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        result['score'].update(decisive_win_rate_percent=p*100,wilson95_decisive_win_rate_percent=[(center-margin)*100,(center+margin)*100])
    return result

def main(a):
    a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'games';cache.mkdir(exist_ok=True)
    games=[]
    for p in sorted(a.logs.iterdir()):
        if not p.is_dir():continue
        fs=list(p.glob('*/meta.json'))
        if len(fs)==22 and all((m:=json.loads(f.read_text())).get('final') and m.get('status')=='ok' for f in fs):games.append(p)
    if a.limit:games=games[:a.limit]
    jobs=[(str(p),str(cache/(p.name+'.json')),a.max_distance) for p in games]
    records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,job) for job in jobs]
        for f in as_completed(futures):
            r=f.result();records.append(r)
            print(f"Completed {len(records)}/{len(games)}: {r['game']}",flush=True)
    records.sort(key=lambda r:r['game']);result=aggregate(records,a.max_distance)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Finished:',a.output/'summary.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('logs',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0);p.add_argument('--max-distance',type=float,default=50)
    main(p.parse_args())
