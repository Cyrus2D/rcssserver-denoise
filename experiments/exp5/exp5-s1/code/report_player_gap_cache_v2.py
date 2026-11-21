#!/usr/bin/env python3
"""Report a fresh cache-v2 batch separately from legacy runs sharing its log name."""
import argparse,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def table(groups,labels):
    s='| Group | Original mean cm | Used mean cm | Reduction | Added error >25cm |\n|---|---:|---:|---:|---:|\n'
    names = {
        'target_ball_under2m': 'Target within 2 m of ball',
        'target_ball_2to5m': 'Target 2–5 m from ball',
        'target_ball_5mplus': 'Target at least 5 m from ball',
        'observer_ball_under3m': 'Observer within 3 m of ball',
        'next_cycle_using_unchanged_valid_wm_velocity': 'Next-cycle forecast, same valid WM velocity',
    }
    for k in labels:
        label = names.get(k, k)
        v=groups[k];s+=f"| {label} | {v['stock_mean_cm']:.2f} | {v['used_mean_cm']:.2f} | {v['gain_percent']:.1f}% | {v['added_over_25cm_percent']:.2f}% |\n"
    return s


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    r=json.loads((a.analysis/'summary.json').read_text());g=next(x for x in r['groups'] if x['mode']=='apply');m=g['actually_used']
    runtime=json.loads((a.analysis/'runtime.json').read_text());context=json.loads((a.analysis/'context/summary.json').read_text())
    risks=json.loads((a.analysis/'risk-cycles/summary.json').read_text());risk=risks['percent_of_observer_cycles_with_eligible_stale_under50']
    replay=[json.loads((a.analysis/f).read_text()) for f in ['worst-output-check.json','good-stock-output-check.json']]
    assert g['integration_versions']==['cache-v2'] and g['count_policy']=='preserve'
    assert r['games']==200 and r['observers']==4400 and g['games_with_lower_mean']==200
    assert not r['config_issues'] and not any(r['checks'].values())
    assert not runtime['runtime_errors'] and not runtime['score_mismatches'] and not runtime['missing_files']
    assert all(v['selection_mismatches']==0 for v in replay)
    for name in ['by_named_age','by_stock_count','by_distance','by_target_team','by_original_error']:
        assert sum(v['samples'] for v in g[name].values())==m['samples']
    for source,dest in [(a.analysis/'summary.json','metrics.json'),(a.analysis/'runtime.json','runtime.json'),
        (a.analysis/'context/summary.json','context.json'),(a.analysis/'risk-cycles/summary.json','cycle-risk.json'),
        (a.analysis/'worst-output-check.json','worst-output-check.json'),(a.analysis/'good-stock-output-check.json','good-stock-output-check.json')]:
        shutil.copy2(source,a.output/dest)
    labels=['1','2','3','4–5','6–10','11–20','21–30'];x=np.arange(len(labels));w=.36
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,groups,ks,title in [(axes[0],g['by_named_age'],labels,'Cycles since confirmed named sighting'),
        (axes[1],g['by_distance'],['0–10','10–20','20–35','35–50'],'True observer distance (m)')]:
        xx=np.arange(len(ks));ax.bar(xx-w/2,[groups[k]['stock_mean_cm']/100 for k in ks],w,color='#98a4af',label='Original WM')
        ax.bar(xx+w/2,[groups[k]['used_mean_cm']/100 for k in ks],w,color='#087f8c',label='Used position')
        ax.set(xticks=xx,xticklabels=ks,ylabel='Mean position error (m)',title=title);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(a.output/'accuracy.svg');plt.close(fig)
    thresholds=[25,50,100];xx=np.arange(3)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar(xx,[m[f'added_over_{k}cm_percent'] for k in thresholds],color='#cb7657')
    axes[0].set(xticks=xx,xticklabels=[f'>{k}cm' for k in thresholds],ylabel='Eligible target-cycles (%)',title='A target becomes worse than its original WM')
    axes[1].bar(xx,[risk[f'observer_cycles_worse{k}'] for k in thresholds],color='#087f8c')
    axes[1].set(xticks=xx,xticklabels=[f'≥1 target\n>{k} cm worse' for k in thresholds],ylabel='Observer-cycles with eligible stale targets (%)',title='An observer sees at least one worsened target')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'object-versus-cycle-risk.svg');fig.savefig(a.output/'object-versus-cycle-risk.png',dpi=120);plt.close(fig)
    examples=[]
    for name in ['accurate-wm-damaged','largest-added-error']:
        p=a.output/'replays'/name/'provenance.json';d=json.loads(p.read_text());examples.append(d)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,d,title in zip(axes,examples,['Originally accurate WM, model regression','Largest added-error case']):
        rec=d['records'];ax.plot([v['cycle'] for v in rec],[v['stock_error_m'] for v in rec],label='Original WM',color='#f39c12')
        ax.plot([v['cycle'] for v in rec],[v['used_error_m'] for v in rec],label='Used position',color='#087f8c')
        ax.axvline(d['focus_cycle'],ls=':',color='#374957');ax.set(xlabel='Recorded cycle',ylabel='Position error (m)',title=title);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(a.output/'recorded-failures.svg');plt.close(fig)
    c=context['stats'];near=c['target_ball_under2m'];far=c['target_ball_5mplus'];good=g['by_original_error']['0–10cm'];worst=g['worst']
    gate=g['selected_stock_count_thresholds']['15']/g['applied_under_distance_filter']*100
    score=r['score'];ci=score['wilson95_decisive_win_rate_percent'];now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    text=f'''# Exp5: corrected cache-v2 preserve run,200 games

**The visual improvement is real, but the model is not uniformly better and
the recorded team result is still worse than its opponent.** This fresh batch
uses cache-v2. Position accuracy improved in all200 games; gameplay benefit has
not been established. The previous duplicate-cache defect is fixed in the
current code and this run's metadata identifies that revision throughout.

Generated {now}. Source is the replacement `logs/exp5-live-preserve` run,
with scores635–684 and78/22/100 W/D/L. Old live logs were removed; their archived
reports are historical evidence only. This analysis uses a new dated output
directory and validates source fingerprints, so old cached results cannot be
silently reused under the repeated log-directory name. The new zero run is
in progress and is not included in this report.

## 1. Actual position-error result

Primary denominator: **{m['samples']:,}** left-observer target-cycles in PlayOn,
true observer distance below50m, identified valid stale WM, original count>0,
named-sighting age1–30 and same-cycle truth. Unselected rows retain WM and
remain in the denominator. Every sample has equal weight; game averages are
not averaged equally.

| Metric | Current cache-v2 preserve |
|---|---:|
| Original WM mean error | {m['stock_mean_cm']:.2f}cm |
| Used mean error | **{m['used_mean_cm']:.2f}cm** |
| Mean reduction | {m['gain_percent']:.2f}% |
| Games with lower mean than their own raw WM | {g['games_with_lower_mean']}/200 |
| Improved by >1cm | {m['better_1cm_percent']:.2f}% |
| Added error >25cm | {m['added_over_25cm_percent']:.2f}% |
| Added error >50cm | {m['added_over_50cm_percent']:.2f}% |
| Added error >1m | {m['added_over_100cm_percent']:.2f}% |
| Maximum added error | {m['max_added_cm']/100:.2f}m |

Added error means used error minus raw WM error for the same observer/target/
cycle. The paired comparison scores both states on the actual treatment
trajectory; it is not a counterfactual game without the model. Fresh measurements
and unsupported/invalid tracks are not in these means.

![Error by age and distance](accuracy.svg)

Named-sighting age in cycles (first column):

{table(g['by_named_age'],labels)}

## 2. Why the display can look excellent while mistakes remain

Per-target rates hide how often a multi-object scene contains at least one bad
estimate. Across **{risks['counts']['observer_cycles_eligible']:,}** observer-cycles
with at least one eligible stale target below50m:

| Scene contains… | Observer-cycles |
|---|---:|
| At least one target improved by >25cm | {risk['observer_cycles_better25']:.2f}% |
| At least one target worsened by >25cm | {risk['observer_cycles_worse25']:.2f}% |
| At least one target worsened by >50cm | {risk['observer_cycles_worse50']:.2f}% |
| At least one target worsened by >1m | {risk['observer_cycles_worse100']:.2f}% |
| Both an improvement >25cm and a regression >25cm | {risk['observer_cycles_both_better25_and_worse25']:.2f}% |
| A target within10m of observer worsened by >25cm | {risk['observer_cycles_worse25_near10']:.2f}% |

![Different target and scene denominators](object-versus-cycle-risk.svg)

Both improved and worsened objects can be present in the same frame. These are
descriptive observer-target exposures, **not percentages of wrong decisions or
goals caused by the model**. They are correlated across objects, time and agents.
The model can visibly correct many positions and still put one tactically
important player on the wrong side of a pass/interception decision.

There were{risks['counts']['target_episodes_added_over1m']:,} observer-target
episodes with added error>1m;{risks['counts']['target_episodes_added_over1m_at_least3cycles']:,}
lasted at least3 consecutive cycles, maximum
{risks['max_consecutive_cycles_added_over1m']} cycles. These are not independent
real-world incidents; multiple observers can describe the same physical play.

## 3. Already-good WM and two real recorded examples

When the original WM error was below10cm, mean error changed
**{good['stock_mean_cm']:.2f}→{good['used_mean_cm']:.2f}cm** across
{good['samples']:,} eligible rows;{good['added_over_25cm_percent']:.2f}% added
>25cm. Large improvements on already-bad states dominate the global average,
while some already-good states are damaged. These truth-defined groups are
diagnostics, not directly available live gates.

{table(g['by_original_error'],['0–10cm','10–25cm','25–50cm','50–100cm','1–2m','2m+'])}

![Actual recorded failures around the selected cycles](recorded-failures.svg)

- **Accurate WM damaged:** game202611180055_20604, observer left5, teammate6,
  cycle5202: error **0.050m→6.451m**; original count11; routing score0.962.
  The team actually won that game5–3. This example proves a bad estimate, not
  that it caused a particular lost game or goal.
- **Largest added error:** game{worst['game']}, observer{worst['observer']},
  opponent9, cycle{worst['cycle']}: **{worst['stock_error_m']:.3f}m→
  {worst['used_error_m']:.3f}m**; observer distance{worst['distance_m']:.2f}m,
  original count{worst['stock_count']}, score{worst['score']:.3f}.

Both selected cycles were replayed with the original PyTorch pipeline:
{sum(v['output_samples'] for v in replay)} eligible outputs, maximum difference
{max(v['max_position_difference_m'] for v in replay):.8f}m including CSV rounding,
zero selection disagreements. These failures reproduce in the learned model;
the checked cases are not explained by a different native computation.
The routing score is not a calibrated probability that a position is correct.

### Open these examples in SoccerWindow without rerunning games

The batch did not retain original RCGs at the checked output location. I exported
**reconstructed positioning clips** from the22 agents' same-cycle truth columns,
plus the actual recorded raw/used positions as dlog. Both41-frame clips pass
librcsc's native RCG parser. They are not original server RCGs: action counters
and unrecorded action-state bits are placeholders, so do not use them for action
analysis. No model was rerun to generate the drawn overlays.

1. Open [accurate-WM clip](replays/accurate-wm-damaged/reconstructed.rcg), load
   `replays/accurate-wm-damaged` as the **left** Debug Message directory, enable
   **WM**, select observer5, and seek cycle5202.
2. Open [largest-error clip](replays/largest-added-error/reconstructed.rcg), load
   `replays/largest-added-error` as left debug directory, select observer5,
   seek cycle5036.

Orange circle=raw WM; cyan square=used estimate; monitor players=reconstructed
truth. The clip README/provenance files record sources and limitations.

## 4. Immediately relevant situations near the ball

This section samples10 evenly spaced complete games /110 left observers,
selected without looking at outcomes. It is not all200 games. Distances to
the ball are labels for reporting, never model inputs from truth.

{table(c,['target_ball_under2m','target_ball_2to5m','target_ball_5mplus','observer_ball_under3m','next_cycle_using_unchanged_valid_wm_velocity'])}

For stale targets within2m of the ball, the gain is
**{near['stock_mean_cm']:.2f}→{near['used_mean_cm']:.2f}cm**, about
{near['gain_percent']:.1f}%. At least5m from the ball it is
{far['gain_percent']:.1f}%. The aggregate percentage gain therefore overstates the percentage improvement
for this ball-near subgroup; sample counts and tactical relevance differ. Fresh accurate positions
are preserved, so this is a statement about eligible stale rows, not all ball
contacts. The one-step forecast uses the same valid WM velocity for both points
and consecutive PlayOn truth; it is not a full planner simulation.

## 5. Current integration and scheduling checks

- All4,400 metadata records identify cache-v2. Left2200 use apply/preserve,
  right2200 use off; both adaptive, same frozen expected bundle for the left.
  No server-config/fullstate-reference violations were found.
- Current code clears all derived containers and lookup arrays and resets
  kickability before rebuilding; overlay refresh does not rerun identity
  inference. The earlier native duplicate-cache regression test passed.
- Batch CSVs do not contain every derived pointer-container size. Thus uniqueness
  is supported by the revision/source and native regression checks, not a direct
  per-cycle cache-size trace over these200 debug-off games.
- {g['applied_all']:,} corrections applied across all distances; no fresh,
  unselected, ineligible, stopped or out-of-horizon rows were applied. All
  effective counts match preserve; raw/used error routing is consistent.
- In the two replayed full observer logs, {sum(v['routing_rows'] for v in replay):,}
  valid rows were joined with objects; actual used positions/counts and raw
  positions/counts agree with the prediction records.
- No model/runtime errors, missing AutoTest console/order files, mismatched
  final scores or missing-decision placeholders were found in available logs.
- Random command order was balanced: left first
  {runtime['command_order']['left_first']:,}, right first
  {runtime['command_order']['right_first']:,}. Per-game inference median range
  {g['timing_per_game']['median_range_ms'][0]:.2f}–{g['timing_per_game']['median_range_ms'][1]:.2f}ms;
  largest inference{g['timing_per_game']['maximum_ms']:.2f}ms. These are not full
  action times and no global percentile is inferred from per-game percentiles.

No new activation/routing defect was identified in this review. This is not a
proof that every possible planner interaction is correct; the visual scene and
point errors cannot alone validate tactical choices.

## 6. What preserve still means for decision code

{gate:.2f}% of selected/applied positions below50m retain original count≥15.
The current player-intercept simulator skips such a player; pass/dribble/shot
paths also apply age gates or extra reach margins. These counts measure exposure
to gates, not how often a particular branch determined an action. Position is
corrected but velocity, body and seen ages are unchanged.

The in-progress new zero batch has cache-v2 /left apply-zero /right off metadata
in the checked sample. Its gameplay result is pending. It tests a different
decision-age policy, not a different network. No live model, library, threshold
or game configuration was changed during this review.

## 7. Gameplay conclusion and next action

Scores are **635–684**, wins/draws/losses **78/22/100**. Mean left-minus-right
goals are−0.245/game; whole-game bootstrap95% interval
[{runtime['goal_diff_bootstrap95'][0]:.3f},{runtime['goal_diff_bootstrap95'][1]:.3f}].
Decisive-game win rate is43.82%, Wilson95% interval[{ci[0]:.2f}%,{ci[1]:.2f}%].
The ordinary39% win rate includes draws and should not be compared directly
with50% as though all games were decisive.

The observed result is worse and there is no established gameplay gain. The
intervals still include a balanced result; these data do not isolate a single
cause for the lost goals. Better point accuracy, nonzero tail risk, stale-motion
information and age-dependent action rules can coexist.

Finish the already-running cache-v2 zero batch with the same weights/threshold
and compare it with this dated run. Do not compare it with the removed legacy
zero batch as though both used this integration. There is no reason to repeat
this completed preserve batch again from the checks above. Improving scene
mean error further is not by itself a sufficient reason for more large trials.

## Reproduction and provenance

```bash
cd /home/nader/workspace/denoise
exp=rcssserver-denoise/experiments/exp5/exp5-s1
analysis=logs/exp5-s1/live-analysis/preserve-cache-v2-200-20261118
.venv-exp5-s1/bin/python "$exp/code/audit_player_gap_batch.py" \\
  logs/exp5-live-preserve "$analysis" --workers 6
.venv-exp5-s1/bin/python "$exp/code/audit_player_gap_context.py" \\
  logs/exp5-live-preserve "$analysis/summary.json" "$analysis/context" --workers 6
.venv-exp5-s1/bin/python "$exp/code/audit_player_gap_risk_cycles.py" \\
  logs/exp5-live-preserve "$analysis/summary.json" "$analysis/risk-cycles" --workers 6
.venv-exp5-s1/bin/python "$exp/code/report_player_gap_cache_v2.py" \\
  "$analysis" "$exp/reports/live-preserve-cache-v2-200"
```

The batch reader now checks metadata/data-file fingerprints before reusing a
cache and records integration revisions. Whole-game cached summaries stay in
the dated analysis directory. Historical reports remain unchanged as data
snapshots, separate from this new run. The numerical/model/RCG checks and JSON
evidence are alongside this report.
'''
    # Keep file paths/identifiers intact; format specific prose tokens only.
    for old, new in [
        ('all200', 'all 200'), ('all4,400', 'all 4,400'), ('these200', 'these 200'),
        ('the22', 'the 22'), ('Both41', 'Both 41'), ('run,200', 'run, 200'),
        ('below50m', 'below 50 m'), ('below10cm', 'below 10 cm'),
        ('within10m', 'within 10 m'), ('within2m', 'within 2 m'),
        ('samples10', 'samples 10'), ('/110', '/110'),
        ('scores635', 'scores 635'), ('and78/', 'and 78/'),
        ('age1–30', 'age 1–30'), ('count>0', 'count >0'),
        ('There were221,353', 'There were 221,353'),
        (';100,211', '; 100,211'), ('least3', 'least 3'),
        ('game202611', 'game 202611'), ('left5', 'left 5'), ('teammate6', 'teammate 6'),
        ('opponent9', 'opponent 9'), ('observerl_5', 'observer l_5'),
        ('cycle5202', 'cycle 5202'), ('cycle5036', 'cycle 5036'),
        ('observer5', 'observer 5'), ('count11', 'count 11'), ('count29', 'count 29'),
        ('score0.', 'score 0.'), ('distance18.', 'distance 18.'),
        ('game5–3', 'game 5–3'), ('least5m', 'least 5 m'),
        ('Left2200', 'Left 2200'), ('right2200', 'right 2200'),
        ('inference55.', 'inference 55.'), ('are−0.', 'are −0.'),
        ('is43.', 'is 43.'), ('ordinary39%', 'ordinary 39%'), ('with50%', 'with 50%'),
        ('bootstrap95%', 'bootstrap 95%'), ('Wilson95%', 'Wilson 95%'),
        ('interval[', 'interval ['),
    ]:
        text = text.replace(old, new)
    (a.output/'README.md').write_text(text)
    style='body{font:16px/1.65 system-ui,sans-serif;max-width:1140px;margin:35px auto;padding:0 24px;color:#193343;background:#f7f9fb}h1{font-size:30px}h2{margin-top:38px}table{border-collapse:collapse;display:block;overflow:auto;background:white}td,th{padding:9px 13px;border:1px solid #d8e1e8}th{background:#dfedf2}img{width:100%;background:white}pre{background:#193343;color:#eef7fa;padding:20px;overflow:auto}a{color:#05698b}code{font-size:.9em}'
    (a.output/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp5 cache-v2 preserve review</title><style>'+style+'</style></head><body>'+markdown.markdown(text,extensions=['tables','fenced_code'])+'</body></html>')
    print(a.output/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('analysis',type=Path);p.add_argument('output',type=Path);main(p.parse_args())
