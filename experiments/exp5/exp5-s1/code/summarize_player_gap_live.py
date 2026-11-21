#!/usr/bin/env python3
"""Sample-weighted Exp5 live checks with bounded CSV-reading memory."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(stock, used):
    added = (used - stock) * 100
    return dict(samples=len(stock), stock_mean_cm=float(stock.mean() * 100),
                used_mean_cm=float(used.mean() * 100), used_p95_cm=float(np.quantile(used, .95) * 100),
                gain_percent=float((1 - used.mean() / stock.mean()) * 100) if stock.mean() else None,
                better_1cm_percent=float((added < -1).mean() * 100),
                **{f'added_over_{x}cm_percent': float((added > x).mean() * 100)
                   for x in [0, 1, 5, 10, 25, 50, 100]}, max_added_cm=float(added.max()))


def summarize(root, max_distance, complete_games_only=False, limit=0):
    groups = {}
    skipped = 0
    metadata = []
    per_game = defaultdict(list)
    for file in sorted(root.rglob('meta.json')):
        if not (file.parent/'player_prediction.csv.gz').exists():
            continue
        meta = json.loads(file.read_text())
        metadata.append((file, meta))
        per_game[meta['game_id']].append(meta)
    complete = {k for k, v in per_game.items()
                if len(v) == 22 and all(m.get('final') and m.get('status') == 'ok' for m in v)}
    if limit:
        complete = set(sorted(complete)[:limit])
    config_issues = []
    expected = dict(see_no_quantization='true', full_view='false', see_no_fade_all='true',
                    see_no_fade_unum='true', say_msg_size='0', synch_mode='true',
                    randomize_player_commands='true')
    checks = Counter()
    cycles = Counter()
    for file, meta in metadata:
        if not meta.get('final') or meta.get('status') != 'ok' or (complete_games_only and meta['game_id'] not in complete):
            skipped += 1
            continue
        options = meta.get('run_options', {})
        mode = options.get('player_gap', 'off')
        policy = options.get('player_gap_pos_count', 'preserve')
        key = mode, policy, meta['side']
        if key not in groups:
            groups[key] = dict(frames=[], times=[], statuses=Counter(), observers=0, games=set(), applied=0,
                               eligible=0, selected=0, worst=None, bundles=set(), view_policies=set(), integration_versions=set())
        g = groups[key]
        g['observers'] += 1
        g['games'].add(meta['game_id'])
        g['bundles'].add(options.get('player_gap_bundle', 'off'))
        g['view_policies'].add(options.get('view_policy', 'unknown'))
        g['integration_versions'].add(options.get('player_gap_integration','legacy-unspecified'))
        for k, v in expected.items():
            if str(meta['server_param'].get(k)) != v:
                config_issues.append(dict(game=meta['game_id'], observer=file.parent.name, key=k, expected=v,
                                          actual=meta['server_param'].get(k)))
        if meta['config'].get('use_fullstate') or not meta['config'].get('debug_fullstate'):
            config_issues.append(dict(game=meta['game_id'], observer=file.parent.name, key='fullstate_reference'))
        agent = pd.read_csv(file.parent/'agent_cycle.csv.gz', usecols=['decision_made','game_mode','see_used'])
        cycles['decision_cycles'] += int(agent.decision_made.eq(1).sum())
        cycles['missing_decision_placeholders'] += int(agent.decision_made.eq(0).sum())
        cycles['play_on_decisions'] += int((agent.decision_made.eq(1)&agent.game_mode.eq('play_on')).sum())
        cols = ['cycle','stopped','object_id','status','active','eligible','selected','applied','named_age',
                'stock_count','decision_count','score','inference_us','truth_valid','true_dist',
                'stock_error','pred_error','used_error']
        last_time = None
        for d in pd.read_csv(file.parent/'player_prediction.csv.gz', usecols=cols, chunksize=105000):
            if not len(d):
                continue
            applied = d.applied.eq(1)
            eligible = d.eligible.eq(1)
            selected = d.selected.eq(1)
            checks['applied_ineligible'] += int((applied&~eligible).sum())
            checks['applied_unselected'] += int((applied&~selected).sum())
            checks['applied_fresh'] += int((applied&d.stock_count.le(0)).sum())
            checks['applied_outside_age'] += int((applied&~d.named_age.between(1,30)).sum())
            checks['applied_stopped'] += int((applied&d.stopped.ne(0)).sum())
            checks['applied_score_not_above_threshold'] += int((applied&d.score.le(.8)).sum())
            checks['selected_ineligible'] += int((selected&~eligible).sum())
            checks['wrong_effective_count'] += int((applied&d.decision_count.ne(0 if policy=='zero' else (d.stock_count.floordiv(2).clip(lower=1) if policy=='half' else d.stock_count))).sum())
            checks['wrong_unapplied_count'] += int((eligible&~applied&d.decision_count.ne(d.stock_count)).sum())
            if mode != 'apply':
                checks['applied_while_not_apply'] += int(applied.sum())
            else:
                checks['selected_but_not_applied'] += int((selected&~applied).sum())
            checks['used_error_mismatch'] += int((d.used_error.notna() &
                (np.abs(d.used_error-np.where(applied,d.pred_error,d.stock_error))>1e-7)).sum())
            g['applied'] += int(applied.sum());g['eligible'] += int(eligible.sum());g['selected'] += int(selected.sum())
            g['statuses'].update(d.status.value_counts().to_dict())
            timing = d[d.active.eq(1)].drop_duplicates(['cycle','stopped'])
            if last_time is not None:
                timing = timing[~(timing.cycle.eq(last_time[0])&timing.stopped.eq(last_time[1]))]
            if len(timing):
                last_time = tuple(timing[['cycle','stopped']].iloc[-1])
                g['times'].append(timing.inference_us.to_numpy()/1000)
            valid = eligible & d.truth_valid.eq(1) & d.stock_error.notna() & d.pred_error.notna()
            if max_distance > 0:
                valid &= d.true_dist.ge(0)&d.true_dist.lt(max_distance)
            q = d.loc[valid]
            if len(q):
                stock = q.stock_error.to_numpy()
                proposed = np.where(q.selected.eq(1),q.pred_error,q.stock_error)
                g['frames'].append(np.column_stack([stock,proposed,q.used_error,q.named_age,q.selected,
                    q.stock_count,q.true_dist,(q.object_id.le(11)==(meta['side']=='l'))]))
                j = int(np.argmax(proposed-stock));candidate = q.iloc[j]
                if g['worst'] is None or proposed[j]-stock[j] > g['worst']['added_error_m']:
                    g['worst'] = dict(game=meta['game_id'],observer=file.parent.name,cycle=int(candidate.cycle),
                                      object_id=int(candidate.object_id),named_age=int(candidate.named_age),
                                      stock_count=int(candidate.stock_count),score=float(candidate.score),
                                      distance_m=float(candidate.true_dist),stock_error_m=float(stock[j]),
                                      used_error_m=float(proposed[j]),added_error_m=float(proposed[j]-stock[j]))
        print(f"Read {meta['game_id']}/{file.parent.name}", file=sys.stderr, flush=True)
    result = dict(max_true_distance_m=max_distance, complete_games_only=complete_games_only,
                  complete_games=sorted(complete), skipped_unfinished_or_excluded_observers=skipped,
                  skipped_unfinished_observers=skipped,config_issues=config_issues,
                  checks=dict(checks),cycles=dict(cycles),groups=[])
    for (mode,policy,side),g in sorted(groups.items()):
        r = dict(mode=mode,count_policy=policy,side=side,observers=g['observers'],games=len(g['games']),
                 statuses=dict(g['statuses']),applied_all=g['applied'],eligible_all=g['eligible'],selected_all=g['selected'],
                 bundles=sorted(g['bundles']),view_policies=sorted(g['view_policies']),integration_versions=sorted(g['integration_versions']),worst=g['worst'])
        if g['times']:
            timing = np.concatenate(g['times'])
            r['inference_ms'] = dict(median=float(np.median(timing)),p95=float(np.quantile(timing,.95)),
                                     maximum=float(timing.max()),cycles=len(timing))
        if g['frames']:
            data = np.concatenate(g['frames']);g['frames'].clear()
            stock,proposed,used,age,selected,count,distance,teammate = data.T
            r['selected_percent'] = float(selected.mean()*100)
            r['proposed'] = metrics(stock,proposed);r['actually_used'] = metrics(stock,used)
            r['by_named_age'] = {}
            for label,lo,hi in [('1',1,1),('2',2,2),('3',3,3),('4–5',4,5),('6–10',6,10),('11–20',11,20),('21–30',21,30)]:
                take = (age>=lo)&(age<=hi)
                if take.any():r['by_named_age'][label]=metrics(stock[take],proposed[take])

            r['by_stock_count'] = {}
            for label,lo,hi in [('1',1,1),('2',2,2),('3',3,3),('4–5',4,5),('6–10',6,10),('11–14',11,14),('15–30',15,30)]:
                take=(count>=lo)&(count<=hi)
                if take.any():r['by_stock_count'][label]=metrics(stock[take],used[take])
            r['by_distance'] = {}
            for label,lo,hi in [('0–10',0,10),('10–20',10,20),('20–35',20,35),('35–50',35,50),('50+',50,float('inf'))]:
                take=(distance>=lo)&(distance<hi)
                if take.any():r['by_distance'][label]=metrics(stock[take],used[take])
            r['by_target_team'] = {label:metrics(stock[take],used[take]) for label,take in
                [('teammates',teammate==1),('opponents',teammate==0)] if take.any()}
            r['by_original_error'] = {}
            for label,lo,hi in [('0–10cm',0,.1),('10–25cm',.1,.25),('25–50cm',.25,.5),('50–100cm',.5,1),('1–2m',1,2),('2m+',2,float('inf'))]:
                take=(stock>=lo)&(stock<hi)
                if take.any():r['by_original_error'][label]=metrics(stock[take],used[take])
            r['applied_under_distance_filter']=int(selected.sum()) if mode=='apply' else 0
            r['selected_stock_count_thresholds']={str(k):int(((selected==1)&(count>=k)).sum()) for k in [4,6,11,15]}
        result['groups'].append(r)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',type=Path)
    parser.add_argument('--max-distance',type=float,default=50,help='true report distance only; <=0 means all')
    parser.add_argument('--complete-games-only',action='store_true')
    parser.add_argument('--limit',type=int,default=0,help='maximum sorted complete games; use with --complete-games-only')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.limit and not args.complete_games_only:parser.error('--limit requires --complete-games-only')
    result=summarize(args.logs,args.max_distance,args.complete_games_only,args.limit)
    output=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(output)
    print(output,end='')
