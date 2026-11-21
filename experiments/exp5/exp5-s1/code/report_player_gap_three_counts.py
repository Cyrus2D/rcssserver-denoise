#!/usr/bin/env python3
"""Completed cache-v2 age-policy comparison and compact Exp5 closeout."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def render(path, text, title):
    path.mkdir(parents=True, exist_ok=True)
    (path/'README.md').write_text(text)
    css = 'body{font:16px/1.65 system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 24px;color:#193343;background:#f7f9fb}h2{margin-top:34px}table{border-collapse:collapse;display:block;overflow:auto;background:white}td,th{padding:9px 12px;border:1px solid #d8e1e8}th{background:#dfedf2}img{width:100%;background:white}a{color:#05698b}pre{background:#193343;color:#eef7fa;padding:18px;overflow:auto}'
    (path/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+css+'</style></head><body>'+markdown.markdown(text, extensions=['tables','fenced_code'])+'</body></html>')


def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)



def point_uncertainty(batches):
    rng = np.random.default_rng(50918)
    boot, rows = [], []
    for batch in batches:
        games = batch['per_game']
        assert len(games) == len({g['game'] for g in games}) == 200
        signs = np.sign([g['left']-g['right'] for g in games])
        delta = 3*signs  # left minus right league points per whole game
        boot.append(delta[rng.integers(200, size=(20000,200))].mean(1))
        rows.append(float(delta.mean()))
    comparisons = {}
    for i,name in [(0,'preserve'),(1,'zero')]:
        comparisons[name] = dict(change=rows[2]-rows[i],
            bootstrap95=np.quantile(boot[2]-boot[i],[.025,.975]).tolist())
    return dict(means=dict(zip(['preserve','zero','half'],rows)),comparisons=comparisons)


def main(a):
    paths = [a.preserve, a.zero, a.half]
    names = ['Preserve', 'Zero', 'Half']
    batches = [read(d/'summary.json') for d in paths]
    groups = [next(g for g in r['groups'] if g['mode']=='apply') for r in batches]
    metrics = [g['actually_used'] for g in groups]
    contexts = [read(d/'context/summary.json') for d in paths]
    risks = [read(d/'risk-cycles/summary.json') for d in paths]
    runtimes = [read(d/'runtime.json') for d in paths]
    runtime = runtimes[-1]
    checks = [read(a.half/(name+'-output-check.json')) for name in ('first','last','worst','accurate-stock')]
    assert len({tuple(g['bundles']) for g in groups})==1 and len(groups[0]['bundles'])==1
    for r,g,name in zip(batches,groups,names):
        assert r['games']==200 and r['observers']==4400
        assert g['count_policy']==name.lower() and g['integration_versions']==['cache-v2']
        assert g['view_policies']==['adaptive'] and not r['config_issues'] and not any(r['checks'].values())
        off=next(v for v in r['groups'] if v['mode']=='off')
        assert off['side']=='r' and off['applied_all']==0 and len(r['groups'])==2
        for key in ('by_named_age','by_stock_count','by_distance','by_target_team','by_original_error'):
            assert sum(v['samples'] for v in g[key].values())==g['actually_used']['samples']
    for r in runtimes:
        assert not r['runtime_errors'] and not r['missing_files'] and not r['score_mismatches']
    assert all(c['selection_mismatches']==0 and c['max_position_difference_m']<.0003 for c in checks)
    assert all(r['games']==200 and r['observers']==2200 for r in risks)
    a.output.mkdir(parents=True, exist_ok=True)
    for path,name in zip(paths,names):
        for src,suffix in [('summary.json','metrics'),('context/summary.json','context'),('risk-cycles/summary.json','scene-risk'),('runtime.json','runtime')]:
            shutil.copy2(path/src,a.output/(name.lower()+'-'+suffix+'.json'))
    shutil.copy2(a.half/'runtime.json',a.output/'runtime.json')
    for name in ('first','last','worst','accurate-stock'):
        shutil.copy2(a.half/(name+'-output-check.json'),a.output/('half-'+name+'-output-check.json'))
    points = point_uncertainty(batches)
    (a.output/'points-uncertainty.json').write_text(json.dumps(points,indent=2)+'\n')
    baseline=runtime['references']['baseline']
    scores=[('Stock baseline (separate run)',baseline)]+[(n,r['score']) for n,r in zip(names,batches)]
    score_rows=[]
    for n,s in scores:
        w,d,l=s['wins'],s['draws'],s['losses'];left=s.get('left',s.get('goals_left'));right=s.get('right',s.get('goals_right'))
        score_rows.append([n,f'{left}–{right}',f'{w}/{d}/{l}',f'{3*w+d}–{3*l+d}',f'{w/(w+l)*100:.2f}%',f'{(left-right)/200:+.3f}'])
    score_table=table(['Batch','Goals L–R','W / D / L','Points L–R','Wins excluding draws','Goal diff/game'],score_rows)
    colors=['#8d9ca9','#087f8c','#c97a3d']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));x=np.arange(3);width=.36
    axes[0].bar(x-width/2,[m['stock_mean_cm'] for m in metrics],width,label='Original WM',color='#b7c1ca')
    axes[0].bar(x+width/2,[m['used_mean_cm'] for m in metrics],width,label='Used position',color=colors)
    axes[0].set(xticks=x,xticklabels=names,ylabel='Mean position error (cm)',title='Similar accuracy with all three age policies');axes[0].legend(frameon=False)
    means=[baseline['mean_goal_difference']]+[r['current']['mean_goal_difference'] for r in runtimes]
    intervals=[baseline['goal_diff_bootstrap95']]+[r['goal_diff_bootstrap95'] for r in runtimes]
    axes[1].bar(range(4),means,color=['#b7c1ca']+colors)
    for i,(m,ci) in enumerate(zip(means,intervals)):
        axes[1].errorbar(i,m,yerr=[[m-ci[0]],[ci[1]-m]],fmt='none',ecolor='#213b49',capsize=5)
    axes[1].axhline(0,color='#213b49',lw=1)
    axes[1].set(xticks=range(4),xticklabels=['Stock']+names,ylabel='Left minus right goals per game',title='Outcomes and 95% whole-game bootstrap intervals')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'three-policy-overview.svg');fig.savefig(a.output/'three-policy-overview.png',dpi=120);plt.close(fig)
    ages=['1','2','3','4–5','6–10','11–20','21–30']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for g,r,n,color in zip(groups,risks,names,colors):
        axes[0].plot(ages,[g['by_named_age'][k]['used_mean_cm']/100 for k in ages],'o-',label=n,color=color)
        axes[1].plot(['>25 cm','>50 cm','>1 m'],[r['percent_of_observer_cycles_with_eligible_stale_under50'][f'observer_cycles_worse{k}'] for k in (25,50,100)],'o-',label=n,color=color)
    axes[0].set(xlabel='Cycles since confirmed named sighting',ylabel='Mean used position error (m)',title='Accuracy by time since observation')
    axes[1].set(ylabel='Observer-cycles with eligible stale targets (%)',title='At least one target worsened above threshold')
    for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(a.output/'three-policy-age-and-risk.svg');plt.close(fig)
    metric_rows=[]
    for label,key in [('Original WM mean cm','stock_mean_cm'),('Used mean cm','used_mean_cm'),('Mean reduction %','gain_percent'),('Improved by >1 cm %','better_1cm_percent')]+[(f'Added error >{k} cm %',f'added_over_{k}cm_percent') for k in (1,5,10,25,50,100)]:
        metric_rows.append([label]+[f'{m[key]:.2f}' for m in metrics])
    metric_rows.append(['Maximum added error m']+[f"{m['max_added_cm']/100:.2f}" for m in metrics])
    metric_table=table(['Metric']+names,metric_rows)
    scene_rows=[]
    for label,key in [('At least one target improved >25 cm','better25'),('At least one target worsened >25 cm','worse25'),('At least one target worsened >50 cm','worse50'),('At least one target worsened >1 m','worse100'),('Both improved and worsened >25 cm','both_better25_and_worse25'),('Target within 10 m worsened >25 cm','worse25_near10')]:
        scene_rows.append([label]+[f"{r['percent_of_observer_cycles_with_eligible_stale_under50']['observer_cycles_'+key]:.2f}%" for r in risks])
    age_rows=[]
    for age in ages:
        age_rows.append([age]+[f"{g['by_named_age'][age]['stock_mean_cm']:.2f}→{g['by_named_age'][age]['used_mean_cm']:.2f}" for g in groups])
    contexts_rows=[]
    for key,label in [('target_ball_under2m','Target within 2 m of ball'),('target_ball_2to5m','Target 2–5 m from ball'),('target_ball_5mplus','Target at least 5 m from ball'),('observer_ball_under3m','Observer within 3 m of ball'),('next_cycle_using_unchanged_valid_wm_velocity','Next-cycle forecast; same valid WM velocity')]:
        contexts_rows.append([label]+[f"{c['stats'][key]['stock_mean_cm']:.2f}→{c['stats'][key]['used_mean_cm']:.2f}" for c in contexts])
    compare_rows=[]
    for key in ('preserve','zero','baseline'):
        r=runtime['references'][key];ci=r['independent_game_bootstrap95_for_change']
        compare_rows.append(['Half minus '+key,f"{r['change_in_mean_goal_difference']:+.3f}",f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'])
    point_rows=[]
    for name,r in points['comparisons'].items():
        ci=r['bootstrap95']
        point_rows.append(['Half minus '+name,f"{r['change']:+.3f}",f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'])
    good=groups[-1]['by_original_error']['0–10cm'];worst=groups[-1]['worst'];bad=contexts[-1]['worst_accurate_wm']
    now=datetime.now(ZoneInfo('America/Halifax')).strftime('%Y-%m-%d %H:%M %Z')
    wci=batches[-1]['score']['wilson95_decisive_win_rate_percent']
    text=f'''# Exp5: preserve, zero and half — 200 corrected-cache games each

**Half is correctly applied. All three policies reduce stale-player position
error substantially, but no gameplay advantage has been established.** Half
has balanced wins/losses and points; zero has the least negative goal difference.
There is no statistically established winner among the three policies.

Updated {now}. All **600 games /13,200 finalized observers** identify cache-v2.
Left applies the same frozen model/selector; right is off. Views are adaptive,
visual quantization/hearing/fading are disabled, and fullstate is reference
only. Earlier 400 live games used the defective cache refresh and are excluded.

## 1. Game outcomes

{score_table}

Half's 86 wins and 86 losses give 50% among decisive games, Wilson 95% interval
[{wci[0]:.2f}%, {wci[1]:.2f}%]. The ordinary 43% win rate includes draws.
Points are balanced, but the goal difference is still −40; balanced wins do
not imply equal goal margins. These runs have different seeds and trajectories.

{table(['Comparison','Change in goal difference/game','95% independent-game bootstrap'],compare_rows)}

Every interval includes zero. Half improved observed points versus preserve
and zero, while its goal difference is worse than zero. Choosing the best
metric after seeing three runs is not evidence of a causal improvement.
The stock baseline is a separate historical run, not a matched-seed control.

The point difference below is **left points minus right points per game**,
not left points alone. Its independent whole-game bootstrap also includes zero:

{table(['Comparison','Change in points difference/game','95% bootstrap'],point_rows)}

![Accuracy and outcomes](three-policy-overview.svg)

## 2. Paired position errors below 50 m

Scope: PlayOn, valid identified stale raw WM (original count >0), confirmed
named-sighting age 1–30, true observer distance <50 m and same-cycle truth.
Unselected rows retain WM and remain included. Fresh/unsupported tracks are
outside these means. Every eligible target-cycle has equal weight; game means
are not given equal weight. Counts: {metrics[0]['samples']:,} preserve,
{metrics[1]['samples']:,} zero and {metrics[2]['samples']:,} half.

{metric_table}

Added error means used error minus original WM error for the same target and
cycle. All three reduced their paired mean in all 200 games. This scores raw
and corrected estimates along the actual treatment trajectory, not a game
that would have happened without correction. Different sample mixes explain
small accuracy differences; the weights/selection threshold did not change.

{table(['Named-sighting age']+[n+' original→used cm' for n in names],age_rows)}

![Age and scene-level mistakes](three-policy-age-and-risk.svg)

## 3. Mistakes that the scene average hides

Observer-cycle populations with at least one eligible stale target below 50 m:
{risks[0]['counts']['observer_cycles_eligible']:,} preserve,
{risks[1]['counts']['observer_cycles_eligible']:,} zero,
{risks[2]['counts']['observer_cycles_eligible']:,} half.

{table(['Eligible observer-cycle contains…']+names,scene_rows)}

These are correlated exposures, not percentages of wrong actions or goals
caused. A frame can have many improved objects and one important damaged one.
Half's originally accurate WM subgroup (<10 cm) changed mean error
{good['stock_mean_cm']:.2f}→{good['used_mean_cm']:.2f} cm;
{good['added_over_25cm_percent']:.2f}% added >25 cm. Truth defines this diagnostic
group; it is not an available live selection rule.

- Largest added-error half case: game {worst['game']}, observer
  {worst['observer']}, cycle {worst['cycle']}, object {worst['object_id']}:
  original {worst['stock_error_m']:.3f} m→used {worst['used_error_m']:.3f} m,
  age {worst['named_age']}, score {worst['score']:.3f}.
- Accurate-WM failure from the context sample: game {bad['game']}, observer
  {bad['observer']}, cycle {bad['cycle']}, original {bad['stock_error_m']:.3f} m
  →used {bad['used_error_m']:.3f} m.

Both selected cycles reproduce in the original Python model pipeline.
The severity-weighted selector score is not calibrated positional confidence.

## 4. Near-ball and next-cycle checks

Each policy uses ten evenly spaced complete games /110 left observers,
selected without consulting outcomes. These are descriptive subsets, not full
200-game context audits. True distances are reporting labels only.

{table(['Situation']+[n+' original→used cm' for n in names],contexts_rows)}

Ball-near stale-player gains are smaller than aggregate gains. The simple
next-cycle check uses the same valid WM velocity for both starting positions
and consecutive PlayOn truth; it is not a complete action/intercept simulation.
Neither ball state nor player velocity/body is corrected by this model.

## 5. Half implementation and runtime audit

- All 4,400 half records: 2,200 left apply/half and 2,200 right off/preserve,
  cache-v2, adaptive, correct frozen bundle and experiment parameters.
- Effective decision count is max(1,floor(raw trackedPosCount/2)) only while
  an accepted overlay is applied: 1→1, 2→1, 3→1, 4→2, 7→3. Raw/seen/heard
  ages and future model inputs remain unchanged. Preserve is still default.
- {groups[-1]['applied_all']:,} overlays across all distances. Every count,
  eligibility, score and error-routing audit check returned zero violations;
  fresh states were not replaced and unselected counts were not modified.
- Four sampled observer checks (early, late and two failure cycles):
  {sum(c['output_samples'] for c in checks)} outputs against PyTorch, no selection
  disagreements, maximum position difference
  {max(c['max_position_difference_m'] for c in checks):.8f} m including CSV rounding.
  Routing joins checked {sum(c['routing_rows'] for c in checks):,} valid rows.
  Feature dumps were off, so this is sampled output parity, not a full native
  feature trace over all 200 games.
- No missing AutoTest console/order files, final-score mismatches or model/runtime
  errors. Command rounds: left first {runtime['command_order']['left_first']:,},
  right first {runtime['command_order']['right_first']:,}. Inference per-game
  median range {groups[-1]['timing_per_game']['median_range_ms'][0]:.2f}–
  {groups[-1]['timing_per_game']['median_range_ms'][1]:.2f} ms; maximum
  {groups[-1]['timing_per_game']['maximum_ms']:.2f} ms. These are not whole action times.
- Current source and native regressions address the old duplicate-cache defect;
  batch CSVs do not directly trace all pointer-container sizes every frame.
  No new activation/count/routing defect was identified. This does not prove
  every planner interaction correct.

## 6. Conclusion

The three choices mostly alter how decision code treats an inferred position's
age. Half avoids a fake count-zero sighting while reducing some old-age margins;
it neither bounds the position error nor updates velocity/body/seen ages.
No count policy has demonstrated a gameplay advantage. Keep prediction opt-in;
retain it as a strong position estimator, not an established team-strength gain.
Further work should link mistakes to actual passes/intercepts/action changes
and use matched-seed, swapped-side controls before another model-size sweep.

## Reproduction

```bash
cd /home/nader/workspace/denoise
exp=rcssserver-denoise/experiments/exp5/exp5-s1
.venv-exp5-s1/bin/python "$exp/code/report_player_gap_three_counts.py" \\
  logs/exp5-s1/live-analysis/preserve-cache-v2-200-20261118 \\
  logs/exp5-s1/live-analysis/zero-cache-v2-200-20261118 \\
  logs/exp5-s1/live-analysis/half-cache-v2-200-20261118 \\
  "$exp/reports/live-count-cache-v2-200" \\
  --closeout rcssserver-denoise/experiments/exp5/reports/final
```

The batch, context, scene-risk, runtime and Python-parity JSON evidence is
alongside this report. [Half launch details](../half-count-policy.md) ·
[Earlier preserve failure clips](../live-preserve-cache-v2-200/report.html).
'''
    render(a.output,text,'Exp5: three live count policies')
    if a.closeout:
        closeout(a,batches,groups,metrics,score_table,now)
    print(a.output/'report.html')


def closeout(a,batches,groups,metrics,score_table,now):
    link='../../exp5-s1/reports/live-count-cache-v2-200/report.html'
    text=f'''# Exp5 closeout: predicting unseen players

**We substantially reduced position error during visual gaps, but did not
establish a gameplay advantage.** Three corrected-cache policies completed
200 games each; half balanced wins/losses and points, with a negative goal
margin. Prediction remains opt-in. Updated {now}.

## Steps and what each taught us

| Step | What we did | What we learned / achieved |
|---|---|---|
| 0. Isolate visual gaps | Exact visual fields, adaptive 60°/120°/180° views, no hearing/fading; fullstate for labels only | Separate unseen motion from quantization; exact observations do not prevent stale-WM error |
| 1. Baseline audit | 200 games, 4,400 observers; ages, distances, views, position/velocity/body | Checked 819,660,850 raw/fix scalar readings: zero error at saved precision; missing fields and identity errors still matter |
| 2. Causal dataset | Whole-game split: 160 train, 20 validation, 20 test; 51 features ×32 steps and uniform-number embedding | No future/fullstate inputs; position-only output, reset at interruptions; preserve real observations |
| 3. Three predictors | Single point, point+uncertainty, three-position mixture | Same 28.18M test rows: WM 129.43 cm→57.22 /65.47 /73.68 cm; single point best mean |
| 4. Reliability study | Confidence gates, failure cohorts, confidence calibration, capacity/data/training and model/WM selectors | Uncertainty does not automatically identify better-than-WM predictions; some already-good states are damaged |
| 5. Refine and select | GRU128, WM-centered residual learning, harm penalties, boosted/readable tree selection, blends/caps, seed repeats | Practical policy trades a little mean accuracy for fewer harmful replacements; a cap bounds added current error but costs mean gain |
| 6. Independent confirmation | Freeze policies, then score 10 new baseline games | Below 50 m: WM 134.98 cm→49.37 mean-focused /52.30 practical; added >25 cm 7.69% /3.64%. Small 7/11-node trees are credible engineering alternatives, checked exploratorily |
| 7. Native integration | Client-owned network files/inference, librcsc decision overlay; shadow/apply, parity tests | Preserve raw tracking/model inputs; rebuild all derived caches; position changes alone do not update velocity/body |
| 8. Visual/cache diagnosis | Colored dlog and agent debug for every observer; fix duplicated player containers/identity inference | Old 400 live match outcomes are confounded; cache-v2 correction is essential; visual accuracy is not tactical validation |
| 9. Corrected live tests | Preserve, zero and half: 200 games each, same frozen model, right off | About 54 cm mean error with all three; no proven match-strength gain; half-count formula works as requested |

## Final corrected-cache live results

{score_table}

The stock baseline is a separate historical run. The three live model runs
have different seeds/trajectories, so a higher point total is not a causal
policy ranking. Half's goal-difference change versus preserve is +0.045/game
(95% independent bootstrap [−0.455,+0.520]); versus zero −0.080
([−0.580,+0.410]). Both include zero. The old 400 pre-fix live games are excluded.

Paired stale-player error below 50 m:

{table(['Policy','Original WM cm','Used cm','Reduction','Added error >25 cm'],[[n,f"{m['stock_mean_cm']:.2f}",f"{m['used_mean_cm']:.2f}",f"{m['gain_percent']:.2f}%",f"{m['added_over_25cm_percent']:.2f}%"] for n,m in zip(['Preserve','Zero','Half'],metrics)])}

Every eligible target-cycle has equal weight. Means cover PlayOn, identified
valid stale WM, named-sighting age 1–30 and true distance <50 m; unselected rows
stay in the denominator. Offline cohorts and live cohorts differ; do not add
their gains or interpret the mean as the error of all objects in every cycle.

## What is worth keeping

- Strong causal position estimator: final frozen practical model is a WM-centered
  residual GRU128 plus harm-weighted boosted selector (strict score>0.8), predicting
  current x,y from observation history. Self, ball, player velocity/body stay stock.
- Real average gains coexist with rare multi-metre mistakes. Roughly one-third
  of eligible observer scenes contain a target worsened by >25 cm, often alongside
  many improved targets; these are not percentages of wrong actions.
- Exact sensors do not reveal unseen actions. Fresh/anonymous observations,
  identity and missing velocity/body fields need separate masks/ages. A selector
  score is not a guarantee that WM is worse.
- Decision age is a policy, not a visual observation: preserve keeps raw age,
  zero exposes 0, half exposes max(1,floor(raw/2)), only while an overlay is active.
- Correct caches, raw-input isolation, numerical parity and visual tooling are
  reusable achievements. Lower point error alone does not establish better play;
  action-level diagnosis and matched-seed/swapped-side tests are the next useful
  checks if this research is reopened. No new large run is requested here.

[Detailed 600-game audit]({link}) ·
[Baseline](../02-live-200/report.html) ·
[Full model/refinement study](../../exp5-s1/reports/report.html) ·
[Native/cache handoff archive](integration-handoff.html) ·
[SoccerWindow commands](../../debug/README.md).
'''
    render(a.closeout,text,'Exp5 closeout: steps, achievements and lessons')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('preserve','zero','half','output'):p.add_argument(name,type=Path)
    p.add_argument('--closeout',type=Path)
    main(p.parse_args())
