# Exp5-S1: position prediction during visual gaps

**Research complete.** Offline refinements and three corrected-cache live
batches are finished. Large position-error gains did not establish better
gameplay. The final model is position-only, client-owned and off by default.

[Short Exp5 closeout](../reports/final/report.html) ·
[600-game preserve/zero/half review](reports/live-count-cache-v2-200/report.html) ·
[Full model study](reports/report.html) ·
[Half launch/verification](reports/half-count-policy.md) ·
[SoccerWindow commands](../debug/README.md).

| Count policy | Original→used error cm | Added error >25 cm | Goals L–R | W / D / L |
|---|---:|---:|---:|---:|
| preserve | 142.32→53.78 | 3.77% | 635–684 | 78/22/100 |
| zero | 144.00→53.99 | 3.79% | 678–702 | 78/35/87 |
| half | 142.87→53.70 | 3.80% | 656–696 | 86/28/86 |

Each count policy has 200 finalized cache-v2 games; same model/selector,
left apply versus right off. Means below50m are sample-weighted eligible
PlayOn stale-player errors. All policies improve each game's paired position
mean; no policy has established a gameplay advantage. Half balanced points,
but its goal difference remained negative and is worse than zero's.
The full report includes uncertainty, tail errors, near-ball situations and
runtime/count checks.

**Historical qualification:** the older400 pre-fix live games had duplicate
derived player containers. Keep their archived reports distinct from these600
corrected-cache games. The sections below retain the offline model progression;
training/evaluation datasets are distinct from live treatment trajectories.

## Latest refinement summary

[Final staged report, sections11–17](reports/report.html#refinement) ·
[Exact simplified decision trees](reports/refinement-pruned-trees.json) ·
[Integration handoff](reports/refinement-handoff.md).

Eight neural variants and simple/boosted selectors were compared. On10 new
baseline games (14,184,868 paired samples), the frozen mean-focused policy
scored45.55cm and the practical policy48.34cm. Restricting true observer distance
to<50m (12,534,441 samples):49.37cm /7.69% made >25cm worse for mean selection;
52.30cm /3.64% for practical selection; stock WM134.98cm.

Additional exploratory tree controls on the same new-game batch found a
seven-node mean selector at49.72cm /7.80%, and an eleven-node practical selector
at53.65cm /3.53%, below50m. These are almost as accurate as their more complex
counterparts and much simpler to represent. Pruning changed zero decisions on
500k calibration rows per tree. The primary policies were frozen before new-game
scoring; later tree controls are clearly labelled exploratory reuse.

Rare severe regressions remain: the frozen practical policy added up to11.82m
of error in a29-cycle gap. Fixed correction ceilings bound this risk but sacrifice
considerable mean accuracy. All results are offline; agents in the new games
used the baseline WM for decisions.

## Original matched-model comparison

[Open the visual comparison](reports/report.html) · [Summary table](reports/report.md) · [All metrics](reports/metrics.json) · [Split and provenance](reports/provenance.json)

Tested on **28,180,368 paired samples in 20 held-out games**. Stock mean position error was **129.43 cm**.

| Position model | Mean error cm | Reduction | Improved samples | Worsened samples |
|---|---:|---:|---:|---:|
| Single position | 57.22 | 55.79% | 66.30% | 27.87% |
| Gaussian uncertainty | 65.47 | 49.42% | 65.09% | 32.28% |
| Three-position mixture | 73.68 | 43.07% | 63.67% | 33.22% |

Improved/worsened uses a 1 cm tolerance. The single-position model is the strongest point predictor in this fixed-budget run; every model improved mean error in all 20 test games. Gaussian and mixture 90% regions achieved about 89.65% and 89.63% test coverage. The mixture improved density/NLL over Gaussian, but not point accuracy.

The average gain does not remove regression risk. For the single-position model, samples where stock error was below 10 cm changed from a 1.19 cm mean to 15.32 cm; 77.31% of that subgroup worsened. Its largest added error was 16.56 m. The worst case included an accurate anonymous count-0 update; stale-only subsets and large failures are retained separately. The 25 cm capped policy reduced the single-model mean to 117.44 cm while bounding added position error to approximately 25 cm, demonstrating the accuracy/risk tradeoff.

**Conclusion:** keep the single-position model as the next accuracy candidate. Before live use, investigate confidence around that accurate point estimate, preservation of anonymous position updates, and high-regression cases. This original comparison did not include runtime integration or new games; the later refinement adds independent baseline-game collection for offline confirmation. These are offline position gains, not win-rate gains.

All three models used 160 training games (3,520 observers), 20 validation games (440 observers; ten selection/ten calibration) and 20 test games (440 observers). Each model sampled 2,048,000 training endpoints across ten epochs from 227,079,477 eligible training samples. The CUDA benchmark measured approximately 55.0 ms/step on CPU versus 11.8 ms/step on the P2200.

## Summary of the comparison

| Model | Output |
|---|---|
| Point | One current position |
| Gaussian | One position and a full 2D covariance |
| Mixture | Three possible position distributions with probabilities |

All three use the same 51 features, uniform-number embedding, 64-unit GRU,
32-cycle causal history, game split, initialization seed and sampled
training windows. Outputs are displacement from the last confirmed named
visual localization, using constant velocity as a prior plus a learned
correction. Output dimensions are position only. Observed velocity and
body remain useful inputs with independent masks and ages.

The deterministic model minimizes smoothed Euclidean error. Probabilistic
models minimize negative log likelihood plus 0.2 times the same distance
loss. These objective differences are part of the approaches being tested.
Mixture readout (weighted mean or most probable component mean) is selected
on validation data, never by picking the component closest to test truth.

## Data and causal boundaries

- Source: /home/nader/workspace/denoise/logs/exp5-logs.
- Generated dataset/runs/evaluation: logs/exp5-s1, outside Git.
- Whole-game split, seed 50501: 160 training, 20 validation, 20 test games.
  Validation is divided into ten checkpoint/gate-selection games and ten
  separate uncertainty-calibration games. Every observer in a game remains
  in its assigned split.
- Fullstate and true distances are labels/reporting groups only. Features
  have an explicit observation-column whitelist. No true states, true
  actions, true player types or future sightings enter the network.
- Inputs include last identified observed position, direct velocity and
  body; ages/masks; movement inferred from previous actual sightings;
  current uncorrected WM position/velocity and ages; observer pose/view;
  available ball WM state; nearest named teammate/opponent, neighbor counts
  and pooled nearby velocity. Context distances use estimates, not truth.
- Reset observed history and recurrent input windows across non-PlayOn,
  stopped-time interruptions and discontinuous cycle numbers.
- A named raw observation is required to establish target history.
  Anonymous observations are not assigned to a named target using truth.
  Current stock WM association errors remain available as features and are
  reported separately.
- Predict when no confirmed named position is available and identified
  sighting age is 1–30 cycles. Confirmed fresh positions are preserved;
  outside the supported horizon the proposed application falls back to WM.
- Primary paired comparison requires a valid known-uniform-number WM
  position and same-cycle truth. Missing/invalid WM positions are excluded
  from paired gains, never counted as model improvements.
- Report identified-sighting age and WM position count separately: an
  anonymous update can reset the latter without confirming identity.
- Every valid paired observer-target-cycle receives equal evaluation
  weight. Training draws uniform eligible endpoints; each epoch samples a
  subset rather than enumerating every possible window.

## Benefit and harm

For each sample, gain = stock position error minus model position error.
Positive means improvement. The report includes mean, p90/p95/p99, fractions
improved/worsened/tied (1 cm reporting tolerance), average improvement and
regression sizes, maximum regression, and regressions exceeding 25/50/100 cm.
Subgroups include gap age, WM count, team, view, distance, association risk,
estimated proximity to the ball, and how accurate stock WM already was.

Two application policies are compared:

1. Raw model position on every eligible sample.
2. A conservative position correction limited to 25 cm from stock WM,
   with age/confidence thresholds selected on validation only. By the
   triangle inequality this limits additional position distance error to
   25 cm (apart from storage precision). It does not guarantee safe actions
   or improved game performance.

For Gaussian/mixture models, held-out calibration checks 50/80/90/95%
coverage and region size. For a common comparison, regions are calibrated
ellipses from total distribution moments; the mixture still retains and
scores its three components through density/NLL and visual examples.
Individual component probabilities are not separate accuracy guarantees.
Calibration is empirical on correlated game data, not an independence
guarantee. This first comparison uses one training seed.

## Reproduce

Run from /home/nader/workspace/denoise. Python environment:
.venv-exp5-s1 (PyTorch 2.5.1 + CUDA 11.8 for the Quadro P2200, NumPy,
pandas, matplotlib and TensorBoard). The original environments are kept.

```sh
step=rcssserver-denoise/experiments/exp5/exp5-s1
py=.venv-exp5-s1/bin/python

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$py" "$step/code/prepare.py" \
  logs/exp5-logs logs/exp5-s1/dataset --workers 16

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 "$py" "$step/code/train.py" \
  logs/exp5-s1/dataset logs/exp5-s1/runs --device auto

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 "$py" "$step/code/evaluate.py" \
  logs/exp5-s1/dataset logs/exp5-s1/runs logs/exp5-s1/evaluation

"$py" "$step/code/report.py" logs/exp5-s1/evaluation "$step/reports"

"$py" "$step/code/test_s1.py"

.venv-exp5-s1/bin/tensorboard --logdir logs/exp5-s1/runs \
  --host 127.0.0.1 --port 6006
```

Data preparation resumes from finalized observer caches. Evaluation caches
are tied to the trained checkpoints/cohort; use a new output directory
when changing weights or data. Checkpoints, per-epoch logs and TensorBoard
events remain under logs/exp5-s1/runs. Keep the frozen split/manifest with
the dataset for future steps.

## Steps

| Step | Purpose |
|---|---|
| 1 | Verify causality, reset boundaries, feature masks and CUDA compatibility |
| 2 | Build all-observer data with a whole-game split |
| 3 | Train the three matched position models |
| 4 | Freeze readouts/checkpoints, calibrate confidence, select conservative gates |
| 5 | Test once on held-out games and report improvements and regressions |

## Reproduction resources

The RAM-backed training loader uses roughly 90–100 GB for this all-observer dataset; evaluation uses about 25 GB. Source logs, the ~96 GB prepared arrays, checkpoints and prediction caches remain outside Git. Do not reuse evaluation caches with changed weights/data/code: the evaluator checks fingerprints. The report generator adds descriptive stale-WM subsets without changing the frozen primary comparison.

Generate the report after evaluation using code/report.py with logs/exp5-s1/evaluation and this step's reports directory as arguments. Nine regression tests cover causality, truth isolation, resets, padding, density math and the correction cap.

## Report labels and detailed change bands

The first chart distinguishes every error decrease, numerical ties, and extra-error bands up to 1, >1–5, >5–25, >25–50 and >50 cm. These disjoint bands are in reports/error-change-buckets.csv and .json. Its Better/Worse columns use all changes beyond 0.001 cm numerical tolerance. The historical summary and later tables explicitly labelled >1 cm retain the original 1 cm threshold. Displayed policy names explain whether the full prediction is used or the movement from librcsc is limited to 25 cm; guarded remains only an internal metrics key.

Section 2 also plots the percentage of full-model predictions that increase position error by more than 25 cm versus librcsc, by confirmed-sighting age; reports/age-regressions.csv contains both 1 cm and 25 cm threshold data.

Section 4 uses green for lower mean error than librcsc, red for higher error, and gray for values equal at the displayed precision. Gain columns retain explicit +/− signs.

The section-4 omission study explicitly compares the union of anonymous_current and unnamed_wm_refresh. They overlap on 116,031 rows: only 122,820 unique rows (0.436%) are excluded. It distinguishes filtering evaluation rows from a full-cohort diagnostic that retains librcsc on them; no retraining occurs. Details are in reports/group-omission-study.json.

## Completed follow-up studies

Summary: confidence-only gates reduce harmful replacements but lose much of the
mean-error improvement. A wider single-position GRU improves full-cohort mean
error from 57.22 to 51.46 cm. A selector around the original GRU reduces harmful
replacements: mean-focused 56.02 cm / 14.88% worsened; risk-focused 59.47 cm /
5.28% worsened. These are separate experiments; width and selector gains have
not been combined.

- Section 5.1:50/80/90/95% confidence thresholds for P(error within 25 cm),
  with 50 cm sensitivity; mixture component-weight thresholds and both fixed
  weighted-mean/highest-weight-center readouts. Rejected rows retain WM.
- Section 9: additional training, width 128, depth 2, and 40 versus 160 source
  training games with a fixed sampled-endpoint budget. Common 500k test subset;
  the validation-selected 128-unit winner is also evaluated on all 28,180,368 rows.
- Section 10: eight WM/model selectors, including clustering and nearest
  neighbors; also selection among all three original networks. Fit and threshold
  selection use separate ten-game validation partitions. Selection never sees
  true states as features.
- [Completed steps and findings](reports/followup-status.md).
- [Verified literature and ranked approaches](reports/selector-research.md).
- [Detailed scaling protocol and learning curves](reports/scaling-study.md).

The original 20 test games had already been inspected. All follow-ups are
exploratory reuse, not a new untouched final test. No live integration is part
of these studies. Additional seeds, a selector around GRU128, targeted hard-case
training and new-game confirmation remain future work.

### Reproduce follow-ups

Run from /home/nader/workspace/denoise after the original frozen-model
evaluation. Selector preparation writes validation truth labels used only by
the confidence calibrator and offline scoring.

```sh
step=rcssserver-denoise/experiments/exp5/exp5-s1
py=.venv-exp5-s1/bin/python

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 "$py" "$step/code/selector_study.py" \
  logs/exp5-s1/dataset logs/exp5-s1/evaluation logs/exp5-s1/selector-study

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 "$py" "$step/code/selector_multi.py" \
  logs/exp5-s1/dataset logs/exp5-s1/evaluation logs/exp5-s1/selector-study

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$py" "$step/code/confidence_gating.py" \
  logs/exp5-s1/evaluation logs/exp5-s1/confidence-study \
  --validation-targets logs/exp5-s1/selector-study/validation-targets.npz --workers 6

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 "$py" "$step/code/scaling_study.py" \
  logs/exp5-s1/dataset logs/exp5-s1/runs/point/best.pt logs/exp5-s1/scaling-study

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 "$py" "$step/code/scaling_full_evaluate.py" \
  logs/exp5-s1/dataset logs/exp5-s1/scaling-study

"$py" "$step/code/scaling_finalize.py" \
  logs/exp5-s1/scaling-study/results.json "$step/reports/scaling-study.md"

"$py" "$step/code/publish_followups.py" "$step" logs/exp5-s1
"$py" "$step/code/test_followups.py"

.venv-exp5-s1/bin/tensorboard --logdir logs/exp5-s1/scaling-study \
  --host 127.0.0.1 --port 6006
```

Training uses CUDA when available. Large arrays, checkpoints, serialized selectors
and TensorBoard events remain under logs/exp5-s1; small exact result snapshots,
source fingerprints, plots and tables are retained in this experiment.
The publisher refreshes the main HTML using existing original plots and metrics;
it does not repeat model inference. Use a new study output directory when
changing data, weights or experiment settings.

## Final refinement research

[Latest visual report, sections11–17](reports/report.html#refinement) ·
[Stage-by-stage findings](reports/refinement-status.md) ·
[Exact frozen policies](reports/refinement-frozen-policies.json).

The follow-up adds readable depth2–5 decision trees; simple count/speed rules;
GRU128 continuation, harm penalties and WM-good sample weighting; a WM-centered
residual architecture; seed repeats; fractional, fixed-cap and age-dependent
corrections. All results include a true-distance<50m comparison. Runtime gates
use estimated WM distance, never true distance. Additional1/2/4m safety ceilings
keep the same practical selector without retuning.

The designated practical objective is near50_risk, set before new-game scoring.
Ten new full-length games were collected with the same baseline teams and
configuration; their data are reserved for independent offline confirmation.
No neural model is applied to either team during collection.

Reproduction from /home/nader/workspace/denoise:

```sh
step=rcssserver-denoise/experiments/exp5/exp5-s1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 .venv-exp5-s1/bin/python \
  "$step/code/run_refinement.py" logs/exp5-s1/dataset logs/exp5-s1

# Refresh the report from saved results, without inference:
.venv-exp5-s1/bin/python "$step/code/publish_followups.py" "$step" logs/exp5-s1

# Focused numerical, truth-isolation and policy checks:
.venv-exp5-s1/bin/python "$step/code/test_refinement.py"

.venv-exp5-s1/bin/tensorboard --logdir logs/exp5-s1/refinement \
  --host 127.0.0.1 --port 6006
```

Training and inference use CUDA when available. Each new fit uses2.048M sampled
endpoints; fine-tuning adds that to the original GRU128 budget. Checkpoints and
TensorBoard events live in logs/exp5-s1/refinement; new raw games and evaluation
arrays stay outside Git. Use a new study directory if changing data, weights,
seeds or methods. Model/threshold selection uses the old selection/calibration
game partitions. The old20 test-game comparisons are exploratory; new-game
confirmation does not establish new-team or corrected-agent performance.

## Live trial integration

[Implementation, complete launcher commands and checks](reports/live-integration.md) ·
[HTML copy](reports/live-integration.html).

The frozen practical GRU128 + boosted selector now runs in Helios/librcsc,
off by default. `--player-gap apply --player-gap-model-dir DIR` enables it;
`--player-gap-pos-count preserve|zero` controls the decision age only (default
preserve). Raw tracker/visual ages never change. Three verified short matches
cover shadow on both sides and left-only apply with each count policy.
All build/check results and paired short-match errors are in the linked report.
The original offline sections above describe their historical collection stage;
they do not imply the new optional live implementation is absent.

## First live pilot: 10 complete games

[Visual report](reports/live-pilot-10/report.html) · [Summary and checks](reports/live-pilot-10/README.md).

Left-only apply/preserve works as configured. On 5.94 million eligible target-cycles below50m, mean error was142.59→52.97cm; 3.71% added >25cm error, close to the offline practical policy. Rare large regressions remain. Ten-game scores were27–36 (3/1/6); gameplay benefit is not established.

## Completed live trial: 200 games with preserved position count

[Full visual audit](reports/live-preserve-200/report.html) · [Findings and reproduction](reports/live-preserve-200/README.md).

On116.21million paired stale target-cycles below50m, mean error was141.55→52.62cm; all200 games improved the position-error mean. Match results were628–705 goals and75/30/95 W/D/L for the left treatment. Rare regressions remain; 14.50% of applied positions below50m retained count≥15, which the current player-intercept simulation excludes. The next isolated test uses --player-gap-pos-count zero. No zero-count gameplay benefit is established yet.
