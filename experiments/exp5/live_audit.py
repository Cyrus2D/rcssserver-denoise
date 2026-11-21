#!/usr/bin/env python3
"""Sample-weighted Exp5 state audit; finalized games only, PlayOn only.

Fullstate is used solely by the logger's precomputed same-cycle error labels.
This script never trains a model or changes any agent state.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import hashlib
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd

VERSION = 2
AGES = ['0', '1', '2', '3', '4', '5', '>5']
OBJECTS = ['self', 'ball', 'teammate', 'opponent']
VIEWS = ['narrow', 'normal', 'wide']
METRICS = ['position', 'velocity', 'body']
THRESHOLDS = dict(position=1., velocity=.3, body=30.)
SENSOR_TOLERANCE = 5e-6


def age_labels(values):
    v = np.asarray(values, float)
    return np.select([v == i for i in range(6)] + [v > 5], AGES, default='missing')


def accumulate(result, key, values, threshold=0.):
    a = np.abs(np.asarray(values, float))
    a = a[np.isfinite(a)]
    if not len(a):
        return
    v = [int(len(a)), float(a.sum()), float(a @ a), float(a.max()), int((a > threshold).sum())]
    if key not in result:
        result[key] = v
    else:
        old = result[key]
        for i in [0, 1, 2, 4]:
            old[i] += v[i]
        old[3] = max(old[3], v[3])


def merge_stats(target, source):
    for key, values in source.items():
        if key not in target:
            target[key] = values.copy()
        else:
            for i in [0, 1, 2, 4]:
                target[key][i] += values[i]
            target[key][3] = max(target[key][3], values[3])


def state_stats(stats, counts, kind, f, prefix, age_names):
    view = f['actual_view'].fillna('unknown').to_numpy(str)
    pc = age_labels(f[prefix + 'pos_count'])
    counts.update({f'age_rows/{kind}/{width}/{age}': int(((view == width) & (pc == age)).sum())
                   for width in VIEWS for age in AGES})
    for metric, field, agefield in zip(METRICS, ['pos_d', 'vel_d', 'body'], age_names):
        errors = f['n_' + prefix + field].to_numpy(float)
        own = age_labels(f[prefix + agefield])
        counts[f'eligible/{kind}/{metric}'] += len(f)
        counts[f'measured/{kind}/{metric}'] += int(np.isfinite(errors).sum())
        accumulate(stats, f'state/{kind}/{metric}/all/all/all', errors, THRESHOLDS[metric])
        for axis, labels in [('own', own), ('pos', pc)]:
            for age in AGES:
                mask = labels == age
                accumulate(stats, f'state/{kind}/{metric}/{axis}/{age}/all', errors[mask], THRESHOLDS[metric])
                for width in VIEWS:
                    accumulate(stats, f'state/{kind}/{metric}/{axis}/{age}/{width}', errors[mask & (view == width)], THRESHOLDS[metric])
        for width in VIEWS:
            accumulate(stats, f'state/{kind}/{metric}/view/all/{width}', errors[view == width], THRESHOLDS[metric])
        for seen in [0, 1]:
            mask = f['see_used'].eq(seen).to_numpy() if kind == 'self' else f['seen'].eq(seen).to_numpy()
            accumulate(stats, f'state/{kind}/{metric}/seen/{seen}/all', errors[mask], THRESHOLDS[metric])
        if kind != 'self':
            dist = f.true_dist.to_numpy(float)
            for name, lo, hi in [('0-3', 0, 3), ('3-10', 3, 10), ('10-20', 10, 20), ('20-40', 20, 40), ('40+', 40, np.inf)]:
                for age in AGES:
                    accumulate(stats, f'state/{kind}/{metric}/distance/{age}/{name}', errors[(dist >= lo) & (dist < hi) & (pc == age)], THRESHOLDS[metric])


def observer(path):
    meta = json.loads((path / 'meta.json').read_text())
    sp = meta['server_param']
    if meta['config']['use_fullstate'] or sp.get('full_view') != 'false' or sp.get('say_msg_size') != '0':
        raise ValueError(f'{path}: fullstate input, full view or hearing enabled')
    if sp.get('see_no_quantization') != 'true' or sp.get('see_no_fade_all') != 'true':
        raise ValueError(f'{path}: unexpected sensor settings')
    acols = '''cycle stopped decision_made game_mode truth_valid see_used view_width sb_view_width est_pos_count est_vel_count est_face_count
n_est_pos_d n_est_vel_d n_est_body n_est_face n_rawfix_pos_d n_rawfix_face n_sb_speed_mag n_sb_speed_dir n_sb_neck'''.split()
    a = pd.read_csv(path / 'agent_cycle.csv.gz', usecols=acols)
    missed = int((a.game_mode.eq('play_on') & a.stopped.eq(0) & a.decision_made.ne(1)).sum())
    a = a[a.decision_made.eq(1) & a.stopped.eq(0) & a.game_mode.eq('play_on') & a.truth_valid.eq(1)].copy()
    if a.duplicated(['cycle', 'stopped']).any():
        raise ValueError(f'{path}: duplicate decision keys')
    a['actual_view'] = a.sb_view_width.fillna(a.view_width)
    stats = {}
    counts = Counter(observers=1, decisions=len(a), see_updates=int(a.see_used.sum()), missed_decision_placeholders=missed)
    state_stats(stats, counts, 'self', a, 'est_', ['pos_count', 'vel_count', 'face_count'])
    for name in ['n_rawfix_pos_d', 'n_rawfix_face', 'n_sb_speed_mag', 'n_sb_speed_dir', 'n_sb_neck']:
        accumulate(stats, 'sensor/self/' + name, a[name], SENSOR_TOLERANCE)
    for name in ['n_est_pos_d', 'n_est_vel_d', 'n_est_body', 'n_est_face']:
        accumulate(stats, 'fresh_wm/self/' + name, a.loc[a.see_used.eq(1), name], SENSOR_TOLERANCE)
    ocols = '''cycle stopped object_id is_teammate truth_valid tru_on_field seen unum_visible match_ambiguous track_match
true_dist in_cone trk_pos_count trk_vel_count trk_body_count trk_seen_pos_count trk_seen_vel_count
n_trk_pos_d n_trk_vel_d n_trk_body n_dist n_dir n_dist_chg n_dir_chg n_body_rel n_head_rel
n_ls_pos_d n_ls_rpos_d n_ls_vel_d n_ls_rvel_d'''.split()
    o = pd.read_csv(path / 'objects.csv.gz', usecols=ocols, low_memory=False)
    o = o[o.stopped.eq(0) & o.truth_valid.eq(1) & o.tru_on_field.eq(1)]
    o = o.merge(a[['cycle', 'stopped', 'actual_view']], on=['cycle', 'stopped'], validate='many_to_one')
    o['kind'] = np.where(o.object_id.eq(0), 'ball', np.where(o.is_teammate.eq(1), 'teammate', 'opponent'))
    for kind in OBJECTS[1:]:
        raw = o[o.kind.eq(kind)]
        identified = raw.seen.eq(1) & raw.match_ambiguous.fillna(0).eq(0)
        if kind != 'ball':
            identified &= raw.unum_visible.eq(1)
        named = raw[identified]
        for name in ['n_dist', 'n_dir', 'n_dist_chg', 'n_dir_chg', 'n_body_rel', 'n_head_rel']:
            accumulate(stats, f'sensor/{kind}/{name}', named[name], SENSOR_TOLERANCE)
        for name in ['n_ls_pos_d', 'n_ls_rpos_d', 'n_ls_vel_d', 'n_ls_rvel_d']:
            accumulate(stats, f'localization/{kind}/{name}', named[name], SENSOR_TOLERANCE)
        for width in VIEWS:
            q = raw[raw.actual_view.eq(width)]
            counts[f'visibility/{kind}/{width}/total'] += len(q)
            counts[f'visibility/{kind}/{width}/seen'] += int(q.seen.eq(1).sum())
            counts[f'visibility/{kind}/{width}/outside_cone'] += int(q.in_cone.eq(0).sum())
        counts[f'potential/{kind}'] += len(raw)
        counts[f'nonidentity_track/{kind}'] += int((~raw.track_match.isin(['unum', 'ball'])).sum())
        # Unknown tracks are matched geometrically using truth; exclude these
        # from the main state cohorts to avoid optimistic identity assignment.
        f = raw[raw.track_match.eq('ball' if kind == 'ball' else 'unum')]
        state_stats(stats, counts, kind, f, 'trk_', ['pos_count', 'vel_count', 'body_count'])
        confirmed = f.seen.eq(1) & f.match_ambiguous.fillna(0).eq(0)
        if kind != 'ball':
            confirmed &= f.unum_visible.eq(1)
        for label, mask in [('confirmed', confirmed), ('no_named_observation', ~confirmed)]:
            fresh = f[f.trk_pos_count.eq(0) & mask]
            for metric, field in zip(METRICS, ['pos_d', 'vel_d', 'body']):
                accumulate(stats, f'fresh_identity/{kind}/{metric}/{label}', fresh['n_trk_' + field], THRESHOLDS[metric])
        counts[f'pos0_unseen/{kind}'] += int((f.trk_pos_count.eq(0) & f.seen.ne(1)).sum())
        counts[f'seen_pos_nonzero/{kind}'] += int((f.seen.eq(1) & f.trk_pos_count.gt(0)).sum())
    return counts, stats


def game_job(job):
    game, cache = map(Path, job)
    paths = sorted(game.glob('*/meta.json'))
    metas = [json.loads(p.read_text()) for p in paths]
    fingerprint = hashlib.sha256(('v' + str(VERSION) + ''.join(p.read_text() for p in paths)).encode()).hexdigest()
    dest = cache / (game.name + '.json')
    if dest.exists():
        data = json.loads(dest.read_text())
        if data.get('fingerprint') == fingerprint:
            return data
    if len(metas) != 22 or not all(m.get('final') and m.get('status') == 'ok' for m in metas):
        raise ValueError(f'{game}: game is not finalized')
    ids = {(m['side'], int(m['unum'])) for m in metas}
    if ids != {(s, n) for s in ['l', 'r'] for n in range(1, 12)}:
        raise ValueError(f'{game}: missing or duplicated observer identities')
    counts, stats = Counter(), {}
    for p in paths:
        c, s = observer(p.parent)
        counts.update(c)
        merge_stats(stats, s)
    data = dict(game=game.name, fingerprint=fingerprint, counts=counts, stats=stats,
                last_cycles=[m.get('last_cycle') for m in metas],
                config=metas[0]['server_param'], policies=sorted({m.get('run_options', {}).get('view_policy', 'unknown') for m in metas}))
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(data, separators=(',', ':')))
    temp.replace(dest)
    return data


def select_games(logs):
    selected, skipped = [], []
    for game in sorted(logs.iterdir()):
        if not game.is_dir():
            continue
        paths = list(game.glob('*/meta.json'))
        try:
            meta = [json.loads(p.read_text()) for p in paths]
        except (json.JSONDecodeError, OSError):
            skipped.append(game.name)
            continue
        if len(meta) == 22 and all(m.get('final') and m.get('status') == 'ok' for m in meta):
            selected.append(game)
        else:
            skipped.append(game.name)
    return selected, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--expected-games', type=int, default=0)
    p.add_argument('--prepare-only', action='store_true', help='Build immutable per-game caches without publishing a partial report')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / 'cache'
    cache.mkdir(exist_ok=True)
    games, skipped = select_games(args.logs)
    if args.limit:
        games = games[:args.limit]
    if args.expected_games and len(games) != args.expected_games:
        raise ValueError(f'Expected {args.expected_games} finalized games, found {len(games)}; skipped {len(skipped)}')
    counts, stats = Counter(), {}
    policies, last_cycles, configurations = set(), [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, d in enumerate(pool.map(game_job, [(g, cache) for g in games]), 1):
            counts.update(d['counts'])
            merge_stats(stats, d['stats'])
            policies.update(d['policies'])
            last_cycles.extend(d['last_cycles'])
            for key in ['see_no_quantization','see_no_fade_all','see_no_fade_unum','see_noise_free','full_view','say_msg_size','fullstate_l','fullstate_r','synch_mode','randomize_player_commands']:
                configurations.setdefault(key,set()).add(d['config'].get(key,'missing'))
            if i % 5 == 0 or i == len(games):
                print(f'Processed {i}/{len(games)} games', flush=True)
    if args.prepare_only:
        print('Caches ready; no partial report published.', flush=True)
        return
    result = dict(version=VERSION, generated_utc=datetime.now(timezone.utc).isoformat(), input_logs=str(args.logs.resolve()),
                  view_policies=sorted(policies), configurations={k:sorted(v) for k,v in configurations.items()},
                  observer_last_cycle_range=[min(last_cycles),max(last_cycles)] if last_cycles else [],
                  games=[g.name for g in games], skipped=skipped, counts=counts,
                  metrics={k: dict(n=int(v[0]), mean=v[1] / v[0], rmse=math.sqrt(v[2] / v[0]), max=v[3], above_threshold=int(v[4])) for k, v in stats.items()},
                  thresholds=THRESHOLDS, sensor_tolerance=SENSOR_TOLERANCE,
                  method='Equal weight per valid observer-target-cycle. PlayOn, stopped=0, decision_made=1, same-cycle truth only. Player WM tracks require unum matching. Missing values excluded independently per metric; no zero filling. Velocity/body own-age plots use their own counters; pos-age tables use position count for all metrics. Current view comparisons are descriptive, not randomized causal comparisons.')
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    from render_live_audit import render
    render(args.output, result)
    print(f'Report: {args.output / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
