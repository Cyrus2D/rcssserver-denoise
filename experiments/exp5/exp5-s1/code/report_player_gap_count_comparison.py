#!/usr/bin/env python3
"""Compare fresh cache-v2 preserve/zero batches; keep legacy snapshots separate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main(a):
    a.output.mkdir(parents=True, exist_ok=True)
    p, z = [read(d/'summary.json') for d in (a.preserve, a.zero)]
    groups = [next(g for g in r['groups'] if g['mode'] == 'apply') for r in (p, z)]
    pg, zg = groups
    pm, zm = [g['actually_used'] for g in groups]
    runtime = read(a.zero/'runtime.json')
    contexts = [read(d/'context/summary.json') for d in (a.preserve, a.zero)]
    risks = [read(d/'risk-cycles/summary.json') for d in (a.preserve, a.zero)]
    checks = [read(a.zero/(name+'-output-check.json')) for name in ('first', 'last', 'worst', 'accurate-stock')]
    assert [g['count_policy'] for g in groups] == ['preserve', 'zero']
    assert pg['bundles'] == zg['bundles'] and len(zg['bundles']) == 1
    for r, g in zip((p, z), groups):
        assert r['games'] == 200 and r['observers'] == 4400
        assert not r['config_issues'] and not any(r['checks'].values())
        assert g['integration_versions'] == ['cache-v2'] and g['view_policies'] == ['adaptive']
        off = next(v for v in r['groups'] if v['mode'] == 'off')
        assert off['side'] == 'r' and off['applied_all'] == 0
        for k in ('by_named_age', 'by_stock_count', 'by_distance', 'by_target_team', 'by_original_error'):
            assert sum(v['samples'] for v in g[k].values()) == g['actually_used']['samples']
    assert not runtime['runtime_errors'] and not runtime['score_mismatches'] and not runtime['missing_files']
    assert all(c['selection_mismatches'] == 0 and c['max_position_difference_m'] < .0003 for c in checks)
    for d, name in ((a.preserve, 'preserve'), (a.zero, 'zero')):
        for src, suffix in [('summary.json', 'metrics'), ('context/summary.json', 'context'), ('risk-cycles/summary.json', 'scene-risk')]:
            shutil.copy2(d/src, a.output/(name+'-'+suffix+'.json'))
    shutil.copy2(a.zero/'runtime.json', a.output/'runtime.json')
    for name in ('first', 'last', 'worst', 'accurate-stock'):
        shutil.copy2(a.zero/(name+'-output-check.json'), a.output/(name+'-output-check.json'))
    colors = ['#8d9ca9', '#087f8c']
    names = ['Preserve', 'Zero']
    ages = ['1', '2', '3', '4–5', '6–10', '11–20', '21–30']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for g, name, color in zip(groups, names, colors):
        axes[0].plot(ages, [g['by_named_age'][k]['used_mean_cm']/100 for k in ages], 'o-', label=name, color=color)
        axes[1].plot(ages, [g['by_named_age'][k]['added_over_25cm_percent'] for k in ages], 'o-', label=name, color=color)
    axes[0].set(ylabel='Mean used position error (m)', title='Same frozen model; similar accuracy')
    axes[1].set(ylabel='Eligible target-cycles (%)', title='Added error >25 cm versus original WM')
    for ax in axes:
        ax.set_xlabel('Cycles since confirmed named sighting'); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.output/'accuracy-by-age.svg'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    baseline = runtime['references']['baseline']
    prior = runtime['references']['preserve']
    rows = [(baseline['mean_goal_difference'], baseline['goal_diff_bootstrap95']),
            (prior['mean_goal_difference'], prior['goal_diff_bootstrap95']),
            (runtime['current']['mean_goal_difference'], runtime['goal_diff_bootstrap95'])]
    axes[0].bar(range(3), [v for v, _ in rows], color=['#b7c1ca']+colors)
    for i, (value, ci) in enumerate(rows):
        axes[0].errorbar(i, value, yerr=[[value-ci[0]], [ci[1]-value]], fmt='none', ecolor='#213b49', capsize=5)
    axes[0].axhline(0, color='#213b49', lw=1)
    axes[0].set(xticks=range(3), xticklabels=['Stock baseline', 'Preserve', 'Zero'], ylabel='Left minus right goals per game', title='Observed outcomes and 95% bootstrap intervals')
    x = np.arange(3); width = .36
    for i, (risk, name, color) in enumerate(zip(risks, names, colors)):
        pct = risk['percent_of_observer_cycles_with_eligible_stale_under50']
        axes[1].bar(x+(i-.5)*width, [pct[f'observer_cycles_worse{k}'] for k in (25, 50, 100)], width, label=name, color=color)
    axes[1].set(xticks=x, xticklabels=['>25 cm', '>50 cm', '>1 m'], ylabel='Observer-cycles with eligible stale targets (%)', title='At least one target has added error above threshold')
    axes[1].legend(frameon=False)
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(a.output/'outcomes-and-scene-risk.svg'); fig.savefig(a.output/'outcomes-and-scene-risk.png', dpi=120); plt.close(fig)

    scores = '| Batch | Goals L–R | Win / Draw / Loss | Points L–R | Wins excluding draws |\n|---|---:|---:|---:|---:|\n'
    for name, s in [('Stock baseline (separate run)', baseline), ('Preserve / cache-v2', p['score']), ('Zero / cache-v2', z['score'])]:
        l, r = s.get('goals_left', s.get('left')), s.get('goals_right', s.get('right'))
        w, d, loss = s['wins'], s['draws'], s['losses']
        scores += f'| {name} | {l}–{r} | {w}/{d}/{loss} | {3*w+d}–{3*loss+d} | {100*w/(w+loss):.2f}% |\n'
    metrics = '| Metric | Preserve | Zero |\n|---|---:|---:|\n'
    for title, key in [('Original WM mean error, cm', 'stock_mean_cm'), ('Used position mean error, cm', 'used_mean_cm'), ('Mean reduction, %', 'gain_percent'), ('Improved by >1 cm, %', 'better_1cm_percent')]+[(f'Added error >{k} cm, %', f'added_over_{k}cm_percent') for k in (1, 5, 10, 25, 50, 100)]:
        metrics += f'| {title} | {pm[key]:.2f} | {zm[key]:.2f} |\n'
    metrics += f"| Maximum added error, m | {pm['max_added_cm']/100:.2f} | {zm['max_added_cm']/100:.2f} |\n"
    scenes = '| Eligible observer-cycle contains… | Preserve | Zero |\n|---|---:|---:|\n'
    for title, key in [('A target improved by >25 cm', 'better25'), ('A target worsened by >25 cm', 'worse25'), ('A target worsened by >50 cm', 'worse50'), ('A target worsened by >1 m', 'worse100'), ('Both an improvement and regression >25 cm', 'both_better25_and_worse25'), ('A target within 10 m worsened by >25 cm', 'worse25_near10')]:
        values = [r['percent_of_observer_cycles_with_eligible_stale_under50']['observer_cycles_'+key] for r in risks]
        scenes += f'| {title} | {values[0]:.2f}% | {values[1]:.2f}% |\n'
    subgroups = '| Situation | Preserve original→used cm | Zero original→used cm | Preserve adds >25 cm | Zero adds >25 cm |\n|---|---:|---:|---:|---:|\n'
    for key, title in [('target_ball_under2m', 'Target within 2 m of ball'), ('target_ball_2to5m', 'Target 2–5 m from ball'), ('target_ball_5mplus', 'Target at least 5 m from ball'), ('observer_ball_under3m', 'Observer within 3 m of ball'), ('next_cycle_using_unchanged_valid_wm_velocity', 'Next-cycle forecast; same valid WM velocity')]:
        v = [c['stats'][key] for c in contexts]
        subgroups += f"| {title} | {v[0]['stock_mean_cm']:.2f}→{v[0]['used_mean_cm']:.2f} | {v[1]['stock_mean_cm']:.2f}→{v[1]['used_mean_cm']:.2f} | {v[0]['added_over_25cm_percent']:.2f}% | {v[1]['added_over_25cm_percent']:.2f}% |\n"
    age_table = '| Named-sighting age | Preserve original→used cm | Zero original→used cm | Zero adds >25 cm |\n|---|---:|---:|---:|\n'
    for k in ages:
        v, w = pg['by_named_age'][k], zg['by_named_age'][k]
        age_table += f"| {k} | {v['stock_mean_cm']:.2f}→{v['used_mean_cm']:.2f} | {w['stock_mean_cm']:.2f}→{w['used_mean_cm']:.2f} | {w['added_over_25cm_percent']:.2f}% |\n"
    good = zg['by_original_error']['0–10cm']; worst = zg['worst']; near = contexts[1]['stats']['target_ball_under2m']
    sample_bad = contexts[1]['worst_accurate_wm']; ci = prior['independent_game_bootstrap95_for_change']; wci=z['score']['wilson95_decisive_win_rate_percent']
    text = f'''# Exp5: corrected-cache zero versus preserve, 200 games each

**Zero count is correctly applied. Position-error reduction remains substantial,
but there is still no established gameplay gain.** Zero recorded fewer losses
than preserve; the difference between these independent runs is inconclusive.

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Both new
batches identify **cache-v2** throughout. The old pre-fix preserve/zero reports
remain historical snapshots and are not inputs to this comparison. Each new
analysis has a dated cache with source fingerprints.

## 1. Actual game results

{scores}

Preserve→zero changed mean goal difference **{prior['mean_goal_difference']:+.3f}
→{runtime['current']['mean_goal_difference']:+.3f} goals/game**, an observed
{prior['change_in_mean_goal_difference']:+.3f}. The independent whole-game 95%
bootstrap interval for that change is **[{ci[0]:+.3f}, {ci[1]:+.3f}]**.
It includes zero. The simulator/command-order seed pairs do not overlap, so
these are not matched games and the count switch cannot be isolated causally.

Zero has 78 wins, as preserve does; draws rose from 22 to 35 and losses fell
from 100 to 87. Its decisive-game win rate is 78/(78+87)=47.27%, Wilson 95%
interval [{wci[0]:.2f}%, {wci[1]:.2f}%]. The ordinary 39% win rate includes draws
and must not be treated as a decisive-game win probability. No advantage over
stock is established; the historical all-stock baseline is a separate run.

![Outcomes and observer-scene risk](outcomes-and-scene-risk.svg)

## 2. Did zero change position accuracy?

Primary scope: PlayOn, true observer-target distance <50 m, identified valid
stale raw WM, original count >0, confirmed named-sighting age 1–30, and
same-cycle truth. There are {pm['samples']:,} preserve and {zm['samples']:,}
zero paired target-cycles. Unselected forecasts keep WM and remain included.
Each target-cycle has equal weight; game means are not averaged equally.
Fresh/unsupported tracks are outside these means.

{metrics}

Added error = error of the actually used position minus original WM error
for the same observer/target/cycle. The model reduced each game's paired mean
in {pg['games_with_lower_mean']}/200 preserve and {zg['games_with_lower_mean']}/200
zero games. These compare two estimates along the treatment trajectory, not
the hypothetical trajectory of the same game played without a model.

The two policies use the same weights and selector threshold. The small mean
difference reflects different observed trajectories/sample mixes; it is not
evidence of a better network or a direct prediction benefit from zeroing age.

![Accuracy and error regressions by named-sighting age](accuracy-by-age.svg)

{age_table}

## 3. Why good-looking scenes still contain risky mistakes

Observer-cycle denominators are {risks[0]['counts']['observer_cycles_eligible']:,}
for preserve and {risks[1]['counts']['observer_cycles_eligible']:,} for zero;
each has at least one eligible stale target below 50 m.

{scenes}

Several improved players and one worsened player often coexist. These are
correlated scene exposures, not percentages of wrong actions or goals caused.
SoccerWindow can show an impressive overall improvement while a tactically
important target is wrong. Errors are not all isolated: zero has
{risks[1]['counts']['target_episodes_added_over1m']:,} observer-target episodes
with added error >1 m; {risks[1]['counts']['target_episodes_added_over1m_at_least3cycles']:,}
lasted at least three consecutive cycles. Multiple observers may see the same play.

For originally accurate WM (<10 cm error), zero changed the mean
**{good['stock_mean_cm']:.2f}→{good['used_mean_cm']:.2f} cm**;
{good['added_over_25cm_percent']:.2f}% of that group added >25 cm error.
This truth-based subgroup is diagnostic, not an available live oracle gate.

- Largest zero added-error case: game {worst['game']}, observer {worst['observer']},
  cycle {worst['cycle']}, target object {worst['object_id']}: original
  {worst['stock_error_m']:.3f} m→used {worst['used_error_m']:.3f} m, named age
  {worst['named_age']}, score {worst['score']:.3f}.
- Already-good WM damaged in the 10-game context sample: game {sample_bad['game']},
  observer {sample_bad['observer']}, cycle {sample_bad['cycle']}: original
  {sample_bad['stock_error_m']:.3f} m→used {sample_bad['used_error_m']:.3f} m.

Both selected severe cycles reproduce in the original Python model pipeline.
Selector scores are not calibrated probabilities of being correct.

## 4. Near the ball and next-cycle forecasts

Each policy uses ten evenly spaced complete game IDs /110 left observers,
chosen without outcomes. This section is a sample, not all 200 games; true
distances label reporting groups only and never enter live inference.

{subgroups}

Zero's ball-near subgroup gained {near['gain_percent']:.1f}%, versus
{zm['gain_percent']:.1f}% in its full eligible population. Current-player
position changes alone do not correct velocity/body or guarantee correct
interceptions. The one-step check uses the same valid WM velocity for both
positions and consecutive PlayOn truth; it is not a full planner simulation.

## 5. Integration and execution checks

- All 4,400 zero metadata records identify cache-v2: 2,200 left apply/zero and
  2,200 right off/preserve; adaptive views, expected frozen bundle, correct
  server experiment parameters and reference-only fullstate.
- {zg['applied_all']:,} accepted overlays across all distances. Effective
  decision counts are zero on applied rows; rejected/ineligible rows retain
  original counts. No fresh/count-zero original tracks are overwritten.
  All existing routing/count/eligibility audit checks returned zero violations.
- Four observer output checks (early, late and the two selected failure cycles)
  tested {sum(v['output_samples'] for v in checks)} eligible outputs against
  PyTorch: zero selection disagreements, maximum position difference
  {max(v['max_position_difference_m'] for v in checks):.8f} m including CSV rounding.
  Full sampled-observer routing joins cover {sum(v['routing_rows'] for v in checks):,}
  rows. Feature dumps were off in this batch, so this is sampled output parity,
  not a complete native-feature trace of every decision.
- No runtime/model errors, missing console/order files, score disagreements or
  missing-decision placeholders were found. Randomized command rounds:
  left first {runtime['command_order']['left_first']:,}, right first
  {runtime['command_order']['right_first']:,}.
- Per-game inference medians {zg['timing_per_game']['median_range_ms'][0]:.2f}–
  {zg['timing_per_game']['median_range_ms'][1]:.2f} ms; largest inference
  {zg['timing_per_game']['maximum_ms']:.2f} ms. These are not whole action times.
- Current code and previous native regression checks address the duplicate-cache
  defect. Debug-off batch CSVs do not record all pointer-container sizes, so
  uniqueness is not directly traced in every frame. No new activation/routing
  defect was identified; this does not prove every tactical interaction correct.

Zero alters effective posCount only while an overlay is active. Raw/seen/heard
ages, model history, velocity and body remain unchanged. Some neck/intercept
rules still use seen/heard ages; zero does not remove every age-based rule.

## 6. New half-count option: an intermediate decision-age experiment

The requested option is **--player-gap-pos-count half**.
[Exact launch and completed native/live checks](../half-count-policy.md). For a selected and
actually applied forecast only:

**effective count = max(1, floor(original tracked posCount / 2))**

| Original count | Preserve | Half | Zero |
|---|---:|---:|---:|
| 1 | 1 | 1 | 0 |
| 2 | 2 | 1 | 0 |
| 3 | 3 | 1 | 0 |
| 4 | 4 | 2 | 0 |
| 7 | 7 | 3 | 0 |
| 29 | 29 | 14 | 0 |

Half avoids marking a prediction as a fresh count-zero sighting and retains
some age margins while reducing others. It neither changes the model output
nor bounds position error and is not an uncertainty estimate. Preserve remains
default. Real tracker/seen ages are not changed; rejected forecasts, shadow/off
mode and fresh observations retain original counts. No half gameplay result
exists yet; any live trial is a new experiment, not a demonstrated fix.

Use the normal runner and a fresh log directory/game ID. Add the following to
the left team's existing start.sh arguments; keep the right team's model off:

```bash
--player-gap apply \\
--player-gap-model-dir /home/nader/workspace/denoise/logs/exp5-s1/live-model \\
--player-gap-pos-count half
```

## 7. What we learned

The count policy explains neither the entire remaining tail risk nor an
established win-rate gain: zero and preserve both achieve roughly 54 cm mean
error, with occasional severe regressions. Zero's observed outcomes are closer
to balanced, but uncertainty is too wide to isolate an effect. Keep the model
off by default. Half is a reasonable requested comparison of decision-age
behavior; do not advertise it as a more accurate predictor. Further diagnosis
should connect corrections to actual action/receiver/intercept changes before
assuming that a larger network or a visually cleaner scene will improve wins.

## Reproduce this comparison

```bash
cd /home/nader/workspace/denoise
exp=rcssserver-denoise/experiments/exp5/exp5-s1
.venv-exp5-s1/bin/python "$exp/code/report_player_gap_count_comparison.py" \\
  logs/exp5-s1/live-analysis/preserve-cache-v2-200-20261118 \\
  logs/exp5-s1/live-analysis/zero-cache-v2-200-20261118 \\
  "$exp/reports/live-count-cache-v2-200"
```

Source audits: audit_player_gap_batch.py, audit_player_gap_context.py,
audit_player_gap_risk_cycles.py, audit_player_gap_runtime.py and
verify_player_gap_live.py. JSON evidence is alongside this report. The earlier
[corrected preserve review](../live-preserve-cache-v2-200/report.html) includes
two reconstructed positioning clips for SoccerWindow.
'''
    (a.output/'README.md').write_text(text)
    css = 'body{font:16px/1.65 system-ui,sans-serif;max-width:1140px;margin:35px auto;padding:0 24px;color:#193343;background:#f7f9fb}h2{margin-top:38px}table{border-collapse:collapse;display:block;overflow:auto;background:white}td,th{padding:9px 13px;border:1px solid #d8e1e8}th{background:#dfedf2}img{width:100%;background:white}pre{background:#193343;color:#eef7fa;padding:20px;overflow:auto}a{color:#05698b}'
    (a.output/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp5 cache-v2: zero versus preserve</title><style>'+css+'</style></head><body>'+markdown.markdown(text, extensions=['tables','fenced_code'])+'</body></html>')
    print(a.output/'report.html')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('preserve', type=Path); p.add_argument('zero', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--half', type=Path, help='Include completed half batch with the three-policy renderer')
    p.add_argument('--closeout', type=Path, help='Also regenerate compact Exp5 closeout (requires --half)')
    args = p.parse_args()
    if args.half:
        from report_player_gap_three_counts import main as three_counts
        three_counts(args)
    else:
        if args.closeout:
            p.error('--closeout requires --half')
        main(args)
