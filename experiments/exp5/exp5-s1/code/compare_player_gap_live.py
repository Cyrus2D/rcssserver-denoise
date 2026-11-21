#!/usr/bin/env python3
"""Compare completed preserve/zero live audits without changing runtime policy."""
import argparse,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def active(r):
    return next(g for g in r['groups'] if g['mode']=='apply')


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    p=json.loads((a.preserve/'summary.json').read_text());z=json.loads((a.zero/'summary.json').read_text())
    pg,zg=active(p),active(z);pm,zm=pg['actually_used'],zg['actually_used']
    runtime=json.loads((a.zero/'runtime.json').read_text())
    pc=json.loads((a.preserve/'context/summary.json').read_text())
    zc=json.loads((a.zero/'context/summary.json').read_text())
    assert pg['count_policy']=='preserve' and zg['count_policy']=='zero'
    assert pg['bundles']==zg['bundles'] and pg['view_policies']==zg['view_policies']==['adaptive']
    assert p['max_true_distance_m']==z['max_true_distance_m']==50
    assert not z['config_issues'] and not any(z['checks'].values())
    assert not runtime['runtime_errors'] and not runtime['score_mismatches'] and not runtime['missing_files']
    for r in [p,z]:
        assert r['games']==200 and r['observers']==4400
        off=next(g for g in r['groups'] if g['mode']=='off')
        assert off['side']=='r' and off['applied_all']==0 and len(r['groups'])==2
        g=active(r)
        for k in ['by_named_age','by_stock_count','by_distance','by_target_team','by_original_error']:
            assert sum(v['samples'] for v in g[k].values())==g['actually_used']['samples'],k
    for source,name in [(a.preserve/'summary.json','preserve-metrics.json'),(a.zero/'summary.json','zero-metrics.json'),
        (a.zero/'runtime.json','runtime.json'),(a.preserve/'context/summary.json','preserve-context.json'),
        (a.zero/'context/summary.json','zero-context.json'),(a.zero/'first-output-check.json','first-output-check.json'),
        (a.zero/'last-output-check.json','last-output-check.json')]:shutil.copy2(source,a.output/name)
    checks=[json.loads((a.zero/f).read_text()) for f in ['first-output-check.json','last-output-check.json']]
    baseline=runtime['references']['baseline'];comparison=runtime['references']['preserve'];score=z['score']
    # Whole-game bootstrap intervals: these runs have different seeds, not paired trajectories.
    plot_rows=[('Stock baseline',baseline['mean_goal_difference'],baseline['goal_diff_bootstrap95']),
               ('Apply / preserve',p['score']['left']/p['games']-p['score']['right']/p['games'],comparison['goal_diff_bootstrap95']),
               ('Apply / zero',runtime['current']['mean_goal_difference'],runtime['goal_diff_bootstrap95'])]
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    axes[0].bar(range(3),[r[1] for r in plot_rows],color=['#98a4af','#cf795d','#147e8d'])
    axes[0].axhline(0,color='#233444',lw=1)
    axes[0].set(xticks=range(3),xticklabels=[r[0] for r in plot_rows],ylabel='Left minus right goals per game',title='Goals/game: observed means and 95% bootstrap intervals')
    for i,(_,v,ci) in enumerate(plot_rows):
        axes[0].errorbar(i,v,yerr=[[v-ci[0]],[ci[1]-v]],fmt='none',ecolor='#233444',capsize=5)
    thresholds=[1,5,10,25,50,100];x=np.arange(len(thresholds));w=.36
    axes[1].bar(x-w/2,[pm[f'added_over_{k}cm_percent'] for k in thresholds],w,label='Preserve',color='#98a4af')
    axes[1].bar(x+w/2,[zm[f'added_over_{k}cm_percent'] for k in thresholds],w,label='Zero',color='#147e8d')
    axes[1].set(xticks=x,xticklabels=[f'>{k}cm' for k in thresholds],ylabel='Eligible target-cycles (%)',title='Added position error versus original WM (<50m)')
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'outcomes-and-risk.svg');fig.savefig(a.output/'outcomes-and-risk.png',dpi=120);plt.close(fig)
    labels=['1','2','3','4–5','6–10','11–20','21–30'];x=np.arange(len(labels))
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for g,name,color,style in [(pg,'Preserve','#7f8e9c','o-'),(zg,'Zero','#147e8d','s--')]:
        axes[0].plot(x,[g['by_named_age'][k]['used_mean_cm']/100 for k in labels],style,label=name,color=color)
        axes[1].plot(x,[g['by_named_age'][k]['samples']/g['actually_used']['samples']*100 for k in labels],style,label=name,color=color)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Used mean position error (m)',title='Accuracy remains almost unchanged')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Eligible samples (%)',title='Named-sighting age distribution')
    for ax in axes:ax.set_xlabel('Cycles since confirmed named sighting');ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(a.output/'accuracy-and-age.svg');plt.close(fig)
    context_labels=['target_ball_under2m','target_ball_2to5m','target_ball_5mplus']
    fig,ax=plt.subplots(figsize=(8.5,4.2));x=np.arange(3)
    ax.bar(x-w/2,[pc['stats'][k]['used_mean_cm'] for k in context_labels],w,label='Preserve: 10 sampled games',color='#98a4af')
    ax.bar(x+w/2,[zc['stats'][k]['used_mean_cm'] for k in context_labels],w,label='Zero: 10 sampled games',color='#147e8d')
    ax.set(xticks=x,xticklabels=['Target–ball <2m','Target–ball 2–5m','Target–ball ≥5m'],ylabel='Used mean position error (cm)',title='Ball-proximity diagnostics: separate sampled trajectories')
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.output/'ball-context.svg');plt.close(fig)
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    points=lambda s:3*s['wins']+s['draws']
    policy_rows=[('Stock baseline',baseline['goals_left'],baseline['goals_right'],baseline['wins'],baseline['draws'],baseline['losses']),
                 ('Apply / preserve',p['score']['left'],p['score']['right'],p['score']['wins'],p['score']['draws'],p['score']['losses']),
                 ('Apply / zero',score['left'],score['right'],score['wins'],score['draws'],score['losses'])]
    score_table='| Batch | Goals L–R | W / D / L | Points L–R | Win rate excluding draws |\n|---|---:|---:|---:|---:|\n'
    for name,left,right,wins,draws,losses in policy_rows:
        score_table+=f'| {name} | {left}–{right} | {wins}/{draws}/{losses} | {wins*3+draws}–{losses*3+draws} | {100*wins/(wins+losses):.2f}% |\n'
    ci=comparison['independent_game_bootstrap95_for_change'];base_ci=baseline['independent_game_bootstrap95_for_change']
    w=zg['worst'];good=zg['by_original_error']['0–10cm'];near=zc['stats']['target_ball_under2m']
    report=f'''# Exp5-S1: zero-count live trial versus preserve

**The zero-count setting works. The observed match result is better than the
preserve batch, but the team has not demonstrated a gameplay advantage over
stock. Position accuracy is almost identical in the two model batches.**

Generated {now}. Current logs: `/home/nader/workspace/denoise/logs/exp5-live-zero`.
Both model batches have200 complete games /4,400 finalized observers. Left
uses the same frozen model; right is off. Only effective position-count policy
changes from preserve to zero; both teams retain adaptive views. Game seeds
are different, so comparisons are unpaired and trajectories are not identical.

## 1. Game outcomes

{score_table}

The decisive-game rate excludes draws (zero:85/(85+85)=50%). Its Wilson95%
interval is[{score['wilson95_decisive_win_rate_percent'][0]:.2f}%,
{score['wilson95_decisive_win_rate_percent'][1]:.2f}%]. The ordinary win rate,
including draws in the denominator, is85/200=42.5%; it must not be compared
directly with50% as though every game were decisive.

Left-minus-right goals/game changed **−0.385 →−0.055** from preserve to zero,
an observed +0.330. Independent whole-game bootstrap95% interval for that
change: **[{ci[0]:+.3f},{ci[1]:+.3f}]**. It includes zero, so these200-game
batches do not establish that the improvement was caused by the count switch.
There are no shared simulator/order-seed pairs between the two runs.

Compared with the historical all-stock Exp5 batch, the zero-count change in
goal difference is{baseline['change_in_mean_goal_difference']:+.3f}/game,
95% bootstrap interval[{base_ci[0]:+.3f},{base_ci[1]:+.3f}]. This also includes zero.
The historical stock batch is a separate run, not a matched-seed control.

![Outcomes and regression risk](outcomes-and-risk.svg)

## 2. Position accuracy and mistakes below50 metres

Each eligible target-cycle has equal weight. The denominator requires a
valid identified stale WM position, original posCount>0, named-sighting age1–30,
PlayOn and same-cycle truth. Unselected rows retain WM and remain included.
Fresh/invalid/unsupported tracks are not included in these accuracy means.

| Metric | Preserve | Zero |
|---|---:|---:|
| Paired eligible samples | {pm['samples']:,} | {zm['samples']:,} |
| Original WM mean error | {pm['stock_mean_cm']:.2f}cm | {zm['stock_mean_cm']:.2f}cm |
| Used mean error | **{pm['used_mean_cm']:.2f}cm** | **{zm['used_mean_cm']:.2f}cm** |
| Mean-error reduction | {pm['gain_percent']:.2f}% | {zm['gain_percent']:.2f}% |
| Improved by >1cm | {pm['better_1cm_percent']:.2f}% | {zm['better_1cm_percent']:.2f}% |
| Added error >1cm | {pm['added_over_1cm_percent']:.2f}% | {zm['added_over_1cm_percent']:.2f}% |
| Added error >5cm | {pm['added_over_5cm_percent']:.2f}% | {zm['added_over_5cm_percent']:.2f}% |
| Added error >10cm | {pm['added_over_10cm_percent']:.2f}% | {zm['added_over_10cm_percent']:.2f}% |
| Added error >25cm | {pm['added_over_25cm_percent']:.2f}% | {zm['added_over_25cm_percent']:.2f}% |
| Added error >50cm | {pm['added_over_50cm_percent']:.2f}% | {zm['added_over_50cm_percent']:.2f}% |
| Added error >1m | {pm['added_over_100cm_percent']:.2f}% | {zm['added_over_100cm_percent']:.2f}% |
| Largest added error | {pm['max_added_cm']/100:.2f}m | {zm['max_added_cm']/100:.2f}m |
| Games with a lower position-error mean than their own WM | {pg['games_with_lower_mean']}/200 | {zg['games_with_lower_mean']}/200 |

Added error is used position error minus original WM error for the same
observer/target/cycle, not absolute error. The two batches have different
sample mixes. The difference between52.62cm and52.58cm is not evidence of a
better network: the weights and selection threshold did not change.

![Accuracy and age](accuracy-and-age.svg)

### By named-sighting age

| Age | Preserve mean cm | Zero mean cm | Preserve adds >25cm | Zero adds >25cm |
|---|---:|---:|---:|---:|
'''
    for label in labels:
        pv,zv=pg['by_named_age'][label],zg['by_named_age'][label]
        report+=f"| {label} | {pv['used_mean_cm']:.2f} | {zv['used_mean_cm']:.2f} | {pv['added_over_25cm_percent']:.2f}% | {zv['added_over_25cm_percent']:.2f}% |\n"
    report+='''
### By true observer-target distance

| Distance m | Preserve original→used cm | Zero original→used cm | Preserve adds >25cm | Zero adds >25cm |
|---|---:|---:|---:|---:|
'''
    for label in ['0–10','10–20','20–35','35–50']:
        pv,zv=pg['by_distance'][label],zg['by_distance'][label]
        report+=f"| {label} | {pv['stock_mean_cm']:.2f}→{pv['used_mean_cm']:.2f} | {zv['stock_mean_cm']:.2f}→{zv['used_mean_cm']:.2f} | {pv['added_over_25cm_percent']:.2f}% | {zv['added_over_25cm_percent']:.2f}% |\n"
    report+=f'''
## 3. Was posCount really zero, and did it contaminate the next input?

- All2,200 left metadata records use apply/zero/adaptive and the expected
  bundle. All2,200 right records use model-off/adaptive.
- **{zg['applied_all']:,} predictions applied** across all distances. Every
  applied prediction record has decision_count=0, while stock_count stays
  positive. No applied row was fresh, ineligible, unselected, stopped, outside
  age1–30, or at/below routing threshold0.8.
- Effective count of an unselected eligible position remains its original
  count. Zero does not rewrite raw tracker/seen ages or velocity/body state.
- Raw object joins were checked in two full left-observer logs and one right
  observer: {sum(x['routing_rows'] for x in checks):,} valid target rows,
  including {sum(x['routing_applied_rows'] for x in checks):,} applied rows.
  Exposed WM position/count and prediction-log position/count agree; the
  raw stock fields retain their original values. Right has no overlay.
- Reconstructed the observation-only Python pipeline from the same logs:
  {sum(x['output_samples'] for x in checks)} sampled eligible outputs from the
  first and last game, zero routing disagreements; maximum position difference
  {max(x['max_position_difference_m'] for x in checks):.8f}m including CSV rounding.
  Input feature dumps were off, so this is a sampled output replay check.
- No server-configuration violations, model/runtime/unknown-column errors,
  missing console/order files or disagreement between metadata and final
  AutoTest score lines were found.
- No missing-decision placeholder rows in {z['cycles']['decision_cycles']:,}
  recorded decisions. Per-game inference medians were
  {zg['timing_per_game']['median_range_ms'][0]:.2f}–{zg['timing_per_game']['median_range_ms'][1]:.2f}ms;
  per-game p95 range{zg['timing_per_game']['p95_range_ms'][0]:.2f}–
  {zg['timing_per_game']['p95_range_ms'][1]:.2f}ms; maximum
  {zg['timing_per_game']['maximum_ms']:.2f}ms. These are inference times only.
- {runtime['command_order']['nonempty_rounds']:,} nonempty randomized command
  rounds: left first{runtime['command_order']['left_first']:,}, right first
  {runtime['command_order']['right_first']:,}. No fixed side priority is visible.

The zero setting removes posCount-based skips/margins while a forecast is
accepted. It does not remove all age handling: some intercept bonuses and
receiver/look checks use seenPosCount/heardPosCount; velocity/body ages are
unchanged. The original age distribution remains similar in both batches.

## 4. Near-ball and next-cycle checks (10 sampled games per policy)

Each context sample uses110 left observers from10 evenly spaced game IDs,
chosen without using outcomes. These are separate sampled trajectories and
must not be presented as the full200-game context result. True distances are
labels for reporting only; none enters inference.

![Ball context](ball-context.svg)

| Situation | Preserve original→used cm | Zero original→used cm | Preserve adds >25cm | Zero adds >25cm |
|---|---:|---:|---:|---:|
'''
    for label,title in [('target_ball_under2m','Target within2m of ball'),('target_ball_2to5m','Target2–5m from ball'),
                        ('target_ball_5mplus','Target at least5m from ball'),('observer_ball_under3m','Observer within3m of ball'),
                        ('next_cycle_using_unchanged_valid_wm_velocity','Next cycle: current position + unchanged valid WM velocity')]:
        pv,zv=pc['stats'][label],zc['stats'][label]
        report+=f"| {title} | {pv['stock_mean_cm']:.2f}→{pv['used_mean_cm']:.2f} | {zv['stock_mean_cm']:.2f}→{zv['used_mean_cm']:.2f} | {pv['added_over_25cm_percent']:.2f}% | {zv['added_over_25cm_percent']:.2f}% |\n"
    report+=f'''
For zero, target-near-ball error is{near['stock_mean_cm']:.2f}→
{near['used_mean_cm']:.2f}cm ({near['gain_percent']:.1f}% reduction). This is a
smaller gain than on the overall stale sample. The one-step test uses identical
valid WM velocity for both candidate positions and true consecutive PlayOn
positions; it is not the complete planner/intercept simulation.

## 5. Remaining regression risk

On {good['samples']:,} zero-batch eligible samples where WM error was already
below10cm, mean error changed **{good['stock_mean_cm']:.2f}→{good['used_mean_cm']:.2f}cm**;
{good['added_over_25cm_percent']:.2f}% added >25cm. These are truth-defined
diagnostic groups, not a deployable oracle rule.

Worst added-error case below50m: `{w['game']}/{w['observer']}`,
cycle{w['cycle']}, object{w['object_id']}; named age{w['named_age']},
original count{w['stock_count']}, effective count0. Observer distance
{w['distance_m']:.2f}m; error changed **{w['stock_error_m']:.2f}→
{w['used_error_m']:.2f}m**, adding{w['added_error_m']:.2f}m. Routing score
{w['score']:.3f} is not a calibrated correctness probability.

Zero lets decision code use corrected stale players more freely, but also
presents bad accepted forecasts as effective count-zero states. It does not
make their uncertainty equal to a real fresh named measurement.

## 6. What this run establishes, and the next useful diagnostic

1. The selected model still reduces mean current-position error substantially.
   Integration checks and sampled Python replay agree in zero mode.
2. Keeping age and zeroing age produce almost identical position-error metrics.
   Their distinction is decision behavior. The observed zero batch is better
   than preserve, but the unpaired result is not statistically conclusive.
3. Neither live trial has established stronger gameplay than stock. Even
   zero's balanced85/30/85 does not demonstrate a gain over the original team.
4. Current-point mean error is insufficient as the sole success metric. Rare
   wrong corrections, remaining motion uncertainty, and action-dependent costs
   still matter. Existing logs do not isolate a single cause of lost goals.

Do not treat zero as a confirmed winning policy or enlarge the network based
on these results alone. If continuing research, a useful next diagnostic is
a clearly labelled **oracle-position ceiling test**: on the same eligible/
selected stale players, provide same-cycle true position for testing only,
with stock velocity/body and an explicit count policy. Keep it separate from
training/normal model inference, use controlled seeds and swap treatment sides.
This would help distinguish residual position-estimation failures from limits
of current-point-only information and the planner's use of it. It is a proposed
experiment; no oracle inference or new runtime behavior was implemented here.

## Reproduce

```bash
cd /home/nader/workspace/denoise
exp=rcssserver-denoise/experiments/exp5/exp5-s1
.venv-exp5-s1/bin/python "$exp/code/audit_player_gap_batch.py" \\
  logs/exp5-live-zero logs/exp5-s1/live-analysis/zero-200 --workers 6
.venv-exp5-s1/bin/python "$exp/code/audit_player_gap_context.py" \\
  logs/exp5-live-zero logs/exp5-s1/live-analysis/zero-200/summary.json \\
  logs/exp5-s1/live-analysis/zero-200/context --workers 6
.venv-exp5-s1/bin/python "$exp/code/audit_player_gap_runtime.py" \\
  logs/exp5-live-zero AutoTest2D/out/exp5-live-zero \\
  logs/exp5-s1/live-analysis/zero-200/runtime.json \\
  --reference preserve logs/exp5-live-preserve --reference baseline logs/exp5-logs
.venv-exp5-s1/bin/python "$exp/code/compare_player_gap_live.py" \\
  logs/exp5-s1/live-analysis/preserve-200 logs/exp5-s1/live-analysis/zero-200 \\
  "$exp/reports/live-zero-200"
```

Per-game/observer caches are under logs/exp5-s1 outside Git. JSON evidence is
alongside this report. Means and worsening rates are weighted by exact sample
counts; per-game percentiles are not averaged into a global percentile.
Game-outcome bootstrap uses whole games. No model weights, server, librcsc or
Helios runtime code was changed by this review.
'''
    report=report.replace("\n\n", "\n\n"+'> **Later integration finding:** the earlier apply path duplicated derived\n> player-cache entries and could repeat identity inference. The current cache-v2\n> build fixes this. These historical match outcomes are not a clean gameplay\n> evaluation of the intended prediction policy. See the\n> [final report](../../../reports/final/report.html).'+"\n\n",1)
    (a.output/'README.md').write_text(report)
    style='body{font:16px/1.65 system-ui,sans-serif;max-width:1140px;margin:35px auto;padding:0 24px;color:#193343;background:#f7f9fb}h1{font-size:30px}h2{margin-top:38px}table{border-collapse:collapse;display:block;overflow:auto;background:white}td,th{padding:9px 13px;border:1px solid #d8e1e8}th{background:#dfedf2}img{width:100%;background:white}pre{background:#193343;color:#eef7fa;padding:20px;overflow:auto}a{color:#05698b}code{font-size:.9em}'
    (a.output/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp5-S1 zero versus preserve</title><style>'+style+'</style></head><body>'+markdown.markdown(report,extensions=['tables','fenced_code'])+'</body></html>')
    print(a.output/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('preserve',type=Path);p.add_argument('zero',type=Path);p.add_argument('output',type=Path);main(p.parse_args())
