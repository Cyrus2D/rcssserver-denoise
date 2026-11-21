#!/usr/bin/env python3
"""Render the preserve-count audit and optional context/runtime diagnostics."""
from pathlib import Path
import argparse,json
from datetime import datetime,timezone
import shutil
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[5]

def table(groups,labels):
    rows=[]
    for label in labels:
        if label not in groups:continue
        v=groups[label]
        rows.append(f"| {label} | {v['samples']:,} | {v['stock_mean_cm']:.2f} | {v['used_mean_cm']:.2f} | {v['gain_percent']:.1f}% | {v['added_over_25cm_percent']:.2f}% |")
    return '| Group | Samples | Original mean cm | Used mean cm | Mean reduction | Added error >25cm |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)

def plot_bars(ax,groups,labels,title):
    x=np.arange(len(labels));w=.36
    ax.bar(x-w/2,[groups[k]['stock_mean_cm']/100 for k in labels],w,label='Original WM',color='#9aa6b2')
    ax.bar(x+w/2,[groups[k]['used_mean_cm']/100 for k in labels],w,label='Used position',color='#087f8c')
    ax.set(xticks=x,xticklabels=labels,ylabel='Mean position error (m)',title=title)
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)

def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    r=json.loads((a.analysis/'summary.json').read_text())
    g=next(x for x in r['groups'] if x['mode']=='apply');m=g['actually_used'];policy=g['count_policy'];score=r['score']
    if policy!='preserve':raise ValueError('This report template documents the preserve-count trial; zero needs a separate comparison report.')
    runtime=json.loads((a.analysis/'runtime.json').read_text()) if (a.analysis/'runtime.json').exists() else None
    context=json.loads((a.analysis/'context/summary.json').read_text()) if (a.analysis/'context/summary.json').exists() else None
    for source,name in [(a.analysis/'summary.json','metrics.json'),(a.analysis/'runtime.json','runtime.json'),
                        (a.analysis/'context/summary.json','context.json'),
                        (a.analysis/'worst-output-check.json','worst-output-check.json'),
                        (a.analysis/'accurate-stock-output-check.json','accurate-stock-output-check.json')]:
        if source.exists():shutil.copy2(source,a.output/name)
    ages=['1','2','3','4–5','6–10','11–20','21–30']
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5))
    plot_bars(axes[0],g['by_named_age'],ages,'Error by cycles since named sighting')
    plot_bars(axes[1],g['by_distance'],['0–10','10–20','20–35','35–50'],'Error by true observer distance (m)')
    fig.tight_layout();fig.savefig(a.output/'accuracy.svg');fig.savefig(a.output/'accuracy.png',dpi=120);plt.close(fig)
    selected=g['applied_under_distance_filter'];flags=g['selected_stock_count_thresholds']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    cuts=[1,5,10,25,50,100]
    axes[0].bar(np.arange(len(cuts)),[m[f'added_over_{x}cm_percent'] for x in cuts],color='#cb6d4c')
    axes[0].set(xticks=np.arange(len(cuts)),xticklabels=[f'>{x}cm' for x in cuts],ylabel='Eligible samples (%)',title='Added error compared with original WM')
    thresholds=[4,6,11,15];shares=[flags[str(k)]/selected*100 for k in thresholds]
    axes[1].bar(np.arange(4),shares,color='#4775a7')
    axes[1].set(xticks=np.arange(4),xticklabels=[f'count ≥{x}' for x in thresholds],ylabel='Applied positions (%)',title='Original age of applied positions (<50m)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    for i,v in enumerate(shares):axes[1].text(i,v+.4,f'{v:.1f}%',ha='center')
    fig.tight_layout();fig.savefig(a.output/'risks-and-counts.svg');plt.close(fig)
    worst=g['worst'];good=g['by_original_error']['0–10cm'];gate_share=flags['15']/selected*100
    ci=score['wilson95_decisive_win_rate_percent']
    report=f'''# Exp5-S1: {r['games']} live games, position-count policy {policy}

**Position accuracy improved, but the recorded game results worsened.** The
application is active and consistent with offline accuracy. This batch does
not establish that lower average position error makes the team stronger.

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.
All {r['games']} games and {r['observers']:,} observer logs finalized. Primary
accuracy uses true observer-target distance **below50m**, PlayOn, valid
identified stale tracks and confirmed named-sighting age1–30. Fullstate is
labels only. Every eligible target-cycle has equal weight; games are not
given equal weight for position averages.

## 1. Main results

| Metric | Result |
|---|---:|
| Left/right goals | {score['left']} / {score['right']} |
| Left wins / draws / losses | {score['wins']} / {score['draws']} / {score['losses']} |
| Paired stale target-cycles below50m | {m['samples']:,} |
| Original WM mean position error | {m['stock_mean_cm']:.2f}cm |
| Used mean position error | **{m['used_mean_cm']:.2f}cm** |
| Reduction in mean position error | {m['gain_percent']:.2f}% |
| Games with lower position-error mean | {g['games_with_lower_mean']} / {r['games']} |
| Improved by >1cm | {m['better_1cm_percent']:.2f}% |
| Added error >1cm | {m['added_over_1cm_percent']:.2f}% |
| Added error >25cm | {m['added_over_25cm_percent']:.2f}% |
| Added error >50cm | {m['added_over_50cm_percent']:.2f}% |
| Added error >1m | {m['added_over_100cm_percent']:.2f}% |
| Maximum added error | {m['max_added_cm']/100:.2f}m |

“Added error” means used error minus original WM error on the same target and
cycle; it is not absolute position error. Unselected rows retain WM and stay
in the denominator. Fresh and unsupported tracks are excluded from these paired
accuracy figures. Selected positions below50m: {selected:,}; applied positions
across all distances: {g['applied_all']:,}. The frozen offline confirmation
was134.98→52.30cm, with3.64% adding >25cm and1.70% adding >50cm.

![Error by age and distance](accuracy.svg)

### By time since confirmed named sighting

{table(g['by_named_age'],ages)}

### By observer distance

{table(g['by_distance'],['0–10','10–20','20–35','35–50'])}

### Teammates and opponents

{table(g['by_target_team'],['teammates','opponents'])}

## 2. Why preserve-count and better positions can still interact badly

About **{gate_share:.2f}%** of applied positions below50m still had original
`posCount≥15`. The current librcsc player-intercept simulator returns1000 for
such a player before simulating it. Other paths also use position age to skip
players or expand their effective reach. Better coordinates do not remove
those gates when the age is preserved.

These counters measure exposure to code gates, not how often a particular
decision actually entered each branch. Observations alone do not prove these
branches caused the match losses.

| Original count condition | Applied positions below50m | Share of applied positions |
|---|---:|---:|
'''
    for k in thresholds:report+=f"| ≥{k} | {flags[str(k)]:,} | {flags[str(k)]/selected*100:.2f}% |\n"
    report+=f'''
Code checked in the current checkout:

- `librcsc-denoise/rcsc/player/intercept_simulator_player.cpp`: the `posCount≥15`
  gate; prediction routing through decisionPos is present.
- `helios-base-denoise/src/player/planner/strict_check_pass_generator.cpp`:
  skips stale opponents above10 and uses age-based opponent reach bonuses.
- `planner/short_dribble_generator.cpp`: opponent bonus up to8 or4 cycles.
- `planner/field_analyzer.cpp`: virtual dash distance depends on the smaller
  of seenPosCount and posCount.

For the next batch use **`--player-gap-pos-count zero`** on the left only.
It makes effective posCount zero only on accepted overlays and preserves raw
tracking/visual ages. It therefore tests age-based decision behavior while
keeping the same model and threshold.

Zero does **not** remove every stale-data effect: the player-intercept bonus
still reads seenPosCount/heardPosCount, and velocity/body ages remain original.
An accurate current point also does not establish accurate future motion.
Zero may help or may make the agent too optimistic about bad predictions;
its performance has not yet been measured.

![Regression risk and retained ages](risks-and-counts.svg)

## 3. When WM was already accurate

For {good['samples']:,} eligible rows with original WM error below10cm,
the mean changed **{good['stock_mean_cm']:.2f}→{good['used_mean_cm']:.2f}cm**.
{good['added_over_25cm_percent']:.2f}% of this subgroup added >25cm error.
The mean remains small, but rare severe regressions are hidden by that average.
These are truth-defined diagnostic groups, not deployable oracle gates.

{table(g['by_original_error'],['0–10cm','10–25cm','25–50cm','50–100cm','1–2m','2m+'])}

Worst overall below50m regression: `{worst['game']}/{worst['observer']}`,
cycle{worst['cycle']}, object{worst['object_id']}. Named age{worst['named_age']},
original count{worst['stock_count']}, observer distance{worst['distance_m']:.2f}m.
Original error **{worst['stock_error_m']:.2f}m**, used error
**{worst['used_error_m']:.2f}m**, added **{worst['added_error_m']:.2f}m**.
Its routing score{worst['score']:.3f} is not a calibrated correctness probability.

The worst-error cycle and a separate already-accurate-WM failure were replayed
through the original PyTorch pipeline. Across36 eligible outputs in those
cycles, position differences stayed below0.000066m, including CSV rounding,
with zero routing disagreements. The severe model proposals reproduce in
Python; they are not explained by a different C++ calculation in these cases.
'''
    if context:
        c=context['stats'];near=c['target_ball_under2m'];far=c['target_ball_5mplus']
        total=sum(c[k]['samples'] for k in ['target_ball_under2m','target_ball_2to5m','target_ball_5mplus'])
        labels=['target_ball_under2m','target_ball_2to5m','target_ball_5mplus']
        pretty=['Target–ball <2m','Target–ball 2–5m','Target–ball ≥5m']
        fig,ax=plt.subplots(figsize=(8.6,4.4));plot_bars(ax,c,labels,'Ball proximity: 10 representative games, stale targets <50m');ax.set_xticklabels(pretty)
        fig.tight_layout();fig.savefig(a.output/'ball-context.svg');plt.close(fig)
        report+=f'''
## 4. Important situations near the ball

This section uses **10 evenly spaced game IDs /110 left observers**, selected
without looking at game outcomes. These are descriptive sampled-game results,
not all200 games. Distances to the ball use truth for reporting only.

![Error near the ball](ball-context.svg)

{table(c,labels+['observer_ball_under3m','observer_ball_3mplus'])}

The largest average gains are away from the ball. For stale targets within2m
of it, mean error changed **{near['stock_mean_cm']:.2f}→{near['used_mean_cm']:.2f}cm**
({near['gain_percent']:.1f}% reduction), compared with
{far['gain_percent']:.1f}% for targets at least5m from it.
Only{near['samples']/total*100:.2f}% of the eligible stale rows in this sample
were within2m of the ball. Fresh rows are excluded, so this percentage must
not be interpreted as the fraction of all players near the ball. Large gains
on many less immediate states do not translate directly into goals.

### One-step motion with unchanged velocity

For {c['next_cycle_using_unchanged_valid_wm_velocity']['samples']:,} rows with
valid current WM velocity and consecutive PlayOn truth, evaluate
`current_position + current_WM_velocity` against the true next-cycle position.
The same velocity is used for original and corrected positions. Mean one-step
error changed **{c['next_cycle_using_unchanged_valid_wm_velocity']['stock_mean_cm']:.2f}→
{c['next_cycle_using_unchanged_valid_wm_velocity']['used_mean_cm']:.2f}cm**.
This simple forecast improved on that matched subset; it is not the complete
intercept/planner simulation and does not validate stale velocity/body estimates.
'''
    report+=f'''
## 5. Integration and scheduling checks

- {r['observers']:,} finalized logs; all expected server parameters and
  reference-only fullstate settings matched. Left uses apply/{policy}/adaptive;
  right is model-off/adaptive. All left logs use the same expected bundle.
- No fresh/ineligible/unselected/out-of-horizon or stopped rows were applied.
  All selected rows applied, effective counts matched policy, and used errors
  matched their recorded original/predicted path.
- {r['cycles']['missing_decision_placeholders']} missing-decision placeholders
  in{r['cycles']['decision_cycles']:,} recorded decisions. This does not measure
  the full CPU action duration.
- Per-game inference medians were{g['timing_per_game']['median_range_ms'][0]:.2f}–
  {g['timing_per_game']['median_range_ms'][1]:.2f}ms; per-game p95s were
  {g['timing_per_game']['p95_range_ms'][0]:.2f}–{g['timing_per_game']['p95_range_ms'][1]:.2f}ms.
  Largest recorded inference was{g['timing_per_game']['maximum_ms']:.2f}ms.
  Per-game percentiles are kept separate; they were not averaged into a global p95.
'''
    if runtime:
        order=runtime['command_order'];b=runtime['baseline'];n=order['nonempty_rounds']
        report+=f'''
- No model/runtime/unknown-column errors in the team console logs. All expected
  console and command-order files were found.
- {n:,} nonempty randomized command rounds: left first{order['left_first']:,}
  ({order['left_first']/n*100:.2f}%), right first{order['right_first']:,}
  ({order['right_first']/n*100:.2f}%). No fixed first-command advantage is visible.

## 6. Gameplay interpretation and next trial

The earlier all-stock Exp5 batch ({b['games']} games) had goals
{b['left_goals']}–{b['right_goals']} and wins/draws/losses
{b['wins']}/{b['draws']}/{b['losses']} for the left side. This batch has
{score['left']}–{score['right']} and{score['wins']}/{score['draws']}/{score['losses']}.
The batch means of left-minus-right goals changed from+0.09 to−0.385 per game.
An independent-game bootstrap95% interval for that change was approximately
[{runtime['change_from_baseline_bootstrap95'][0]:.3f},
{runtime['change_from_baseline_bootstrap95'][1]:.3f}] goals/game. This is an
unpaired, historical-batch comparison, not a matched-seed causal estimate.

Among decisive games in the current batch, the left win rate is
{score['decisive_win_rate_percent']:.2f}%, Wilson95% interval
[{ci[0]:.2f}%,{ci[1]:.2f}%]. This differs from the ordinary{score['wins']}/{r['games']} win rate:
draws must be handled consistently when comparing with50%.

The recorded gameplay direction is worse and should be taken seriously.
Position-error reduction is not an adequate success criterion by itself.
The data identify retained age gates, occasional destruction of already-good
WM states, and smaller gains on immediate ball-near targets as concrete issues;
they do not isolate a single cause of the losses.
'''
    report+='''
For the zero-count trial keep the model, threshold and server/view configuration
fixed. Use left-only `--player-gap-pos-count zero` and a separate
`logs/exp5-live-zero` directory. Compare gameplay, position accuracy, regression
tails and count-gate exposure with this preserve batch. A zero-count result is
still pending; no benefit is promised from that switch.

## Reproduce

```bash
cd /home/nader/workspace/denoise
exp=rcssserver-denoise/experiments/exp5/exp5-s1
.venv-exp5-s1/bin/python "$exp/code/audit_player_gap_batch.py" \\
  logs/exp5-live-preserve logs/exp5-s1/live-analysis/preserve-200 --workers 6
.venv-exp5-s1/bin/python "$exp/code/audit_player_gap_context.py" \\
  logs/exp5-live-preserve logs/exp5-s1/live-analysis/preserve-200/summary.json \\
  logs/exp5-s1/live-analysis/preserve-200/context --workers 6
.venv-exp5-s1/bin/python "$exp/code/report_player_gap_live.py" \\
  logs/exp5-s1/live-analysis/preserve-200 "$exp/reports/live-preserve-200"
```

Per-game summaries and detailed context observer caches are under logs/exp5-s1,
outside Git. Means and worsening rates aggregate exact sample counts and sums;
per-game p95 values are retained separately. Runtime/baseline bootstrap evidence
is in runtime.json. Severe-cycle PyTorch checks are in the two output-check
JSONs. Current runtime inference/model/weights were not changed by this audit.
'''
    report=report.replace("\n\n", "\n\n"+'> **Later integration finding:** the earlier apply path duplicated derived\n> player-cache entries and could repeat identity inference. The current cache-v2\n> build fixes this. These historical match outcomes are not a clean gameplay\n> evaluation of the intended prediction policy. See the\n> [final report](../../../reports/final/report.html).'+"\n\n",1)
    (a.output/'README.md').write_text(report)
    style='body{font:16px/1.65 system-ui,sans-serif;max-width:1140px;margin:35px auto;padding:0 24px;color:#193343;background:#f7f9fb}h1{font-size:30px}h2{margin-top:38px}table{border-collapse:collapse;display:block;overflow:auto;background:white}td,th{padding:9px 13px;border:1px solid #d8e1e8}th{background:#dfedf2}img{width:100%;background:white}pre{background:#193343;color:#eef7fa;padding:20px;overflow:auto}a{color:#05698b}code{font-size:.9em}'
    (a.output/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp5-S1 live batch audit</title><style>'+style+'</style></head><body>'+markdown.markdown(report,extensions=['tables','fenced_code'])+'</body></html>')
    print(a.output/'report.html')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('analysis',type=Path);p.add_argument('output',type=Path);main(p.parse_args())
