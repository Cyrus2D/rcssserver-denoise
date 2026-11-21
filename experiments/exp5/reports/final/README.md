# Exp5 closeout: predicting unseen players

**We substantially reduced position error during visual gaps, but did not
establish a gameplay advantage.** Three corrected-cache policies completed
200 games each; half balanced wins/losses and points, with a negative goal
margin. Prediction remains opt-in. Updated 2026-11-18 22:11 AST.

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

| Batch | Goals L–R | W / D / L | Points L–R | Wins excluding draws | Goal diff/game |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock baseline (separate run) | 678–660 | 87/44/69 | 305–251 | 55.77% | +0.090 |
| Preserve | 635–684 | 78/22/100 | 256–322 | 43.82% | -0.245 |
| Zero | 678–702 | 78/35/87 | 269–296 | 47.27% | -0.120 |
| Half | 656–696 | 86/28/86 | 286–286 | 50.00% | -0.200 |


The stock baseline is a separate historical run. The three live model runs
have different seeds/trajectories, so a higher point total is not a causal
policy ranking. Half's goal-difference change versus preserve is +0.045/game
(95% independent bootstrap [−0.455,+0.520]); versus zero −0.080
([−0.580,+0.410]). Both include zero. The old 400 pre-fix live games are excluded.

Paired stale-player error below 50 m:

| Policy | Original WM cm | Used cm | Reduction | Added error >25 cm |
| --- | ---: | ---: | ---: | ---: |
| Preserve | 142.32 | 53.78 | 62.21% | 3.77% |
| Zero | 144.00 | 53.99 | 62.51% | 3.79% |
| Half | 142.87 | 53.70 | 62.41% | 3.80% |


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

[Detailed 600-game audit](../../exp5-s1/reports/live-count-cache-v2-200/report.html) ·
[Baseline](../02-live-200/report.html) ·
[Full model/refinement study](../../exp5-s1/reports/report.html) ·
[Native/cache handoff archive](integration-handoff.html) ·
[SoccerWindow commands](../../debug/README.md).
