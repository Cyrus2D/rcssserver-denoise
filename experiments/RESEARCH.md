# Denoise research guide

Start with [Exp1](exp1/README.md) for the standard quantized baseline and
[Exp2](exp2/README.md) for the live self-position tracker. The goal is to
reduce error in self, ball, and player estimates by separating sensor
rounding, gaps in observations, missing player identity or motion fields,
and teammate reports. The original experiment matrix below records how to
isolate those factors.

## What the three repositories add

| Repository | Denoise code and integration |
|---|---|
| rcssserver-denoise | Protocol v19 experiment settings in [serverparam.cpp](../src/serverparam.cpp). [visualsenderplayer.cpp](../src/visualsenderplayer.cpp) selects the distance model and controls distance-based fading for high-quality observations. |
| librcsc-denoise | Reads the server settings and adjusts observation timing and localization. `DenoiseLogger` originally wrote schema-v2 logs; F4 schema 3 adds per-field application flags, original ages and isolated observation history. The self-position estimator hook runs before decision caches are built. |
| helios-base-denoise | `start.sh` configures dataset logging and the Exp2 model modes. Helios loads the weights and owns inference, recurrent state and model logging. Behavior logs include `pre:`, `role:`, `plan:` and `exec:` labels. |

See the server [noise-model notes](../.github/instructions/noise-model.instructions.md)
and [visual protocol notes](../.github/instructions/visual-observation-detail.instructions.md).
In a workspace with the sibling checkouts, see the librcsc [dataset schema](../../librcsc-denoise/.github/instructions/denoise-dataset.instructions.md)
and [observation and localization notes](../../librcsc-denoise/.github/instructions/observation-parsing-localization.instructions.md).

## Server controls

These four boolean settings default to `false`. Exp1 used those defaults,
standard quantized `see`, normal view control, visual fading and teammate
communication.

| Setting | Effect |
|---|---|
| `server::full_view` | Sets the view cone to 360°. With synchronized visual updates, it also sends a `see` every cycle. The client retains its view-width setting. |
| `server::see_no_fade_unum` | Keeps team and uniform number visible in far, in-cone, high-quality player observations. The velocity and orientation fields still fade. |
| `server::see_no_fade_all` | Bypasses those far-object fades, including the relevant motion and orientation fields. Close and low-quality observation paths keep their existing behavior. |
| `server::see_noise_free` | Selects `ExactObservation` ahead of Gaussian see. It bypasses distance quantization. Bearing and motion fields still use their normal rounding. |

To select Gaussian see, set `gaussian_see : on` in the Helios player
configuration. To suppress normal teammate payloads, set
`server::say_msg_size=0`. The hearing-capacity settings control delivery
limits.

## Optional command-priority control

For synchronous treatment-versus-stock trials, `randomize_player_commands=true`
buffers complete player action batches and shuffles their execution with an
independent per-cycle generator. It defaults to false. Configure
`command_order_seed=-1` for an automatically selected, recorded game seed.
`command_order_log=true` writes a CSV under `game_log_dir` even when ordinary
logging is off. Missing `done`, malformed input or overflow aborts the match
without a scored result. The underlying physics is unchanged, but this is
a different scheduling condition from the standard server.

[Exp4 implementation, settings and verification](exp4/reports/06-server-command-order.md).
Client v19 parameter mirrors preserve the resolved order and simulator seeds
in player metadata.

## Planned comparisons

Exp1 supplies B0. Exp2 changes self-position estimation while keeping the
standard visual configuration. [Exp3](exp3/README.md) combines V1, F1 and S0
with quantization enabled. Treat the individual follow-up rows as proposed experiments
until a completed run and report are linked from the [index](README.md).

| ID | Change from B0 | Question |
|---|---|---|
| B0 | Standard quantized observations, default view and fade, teammate messages enabled | How large are the baseline errors? |
| N1 | `see_noise_free=true` | How much error comes from distance quantization? |
| G1 | `gaussian_see : on` | How do errors change under Gaussian distance noise? |
| V1 | `full_view=true` | How much does seeing more of the field more often help? |
| F1 | `see_no_fade_unum=true` | How much does persistent player identity help matching and tracking? |
| F2 | `see_no_fade_all=true` | How much do persistent identity, velocity and orientation fields help? |
| S0 | `say_msg_size=0` | How does disabling teammate payloads affect the estimates? |
| U0 | `see_noise_free=true full_view=true see_no_fade_all=true` | What error remains when distance quantization, visual gaps and far-field fades are reduced together? |
| GV | `full_view=true` with Gaussian see | How does Gaussian noise behave with frequent, wide observations? |

Change one factor at a time before running combined configurations.
V1 changes both coverage and observation frequency, so separating those
effects requires independent controls. N1 isolates distance quantization
while leaving other rounded fields available for analysis.

## Run and measure

- For scored runs, enable `server::fullstate_l=true` and `server::fullstate_r=true`.
  Use Helios `--fullstate reference` so the server positions are available for
  scoring while decisions and model inputs still use observations.
- Use a separate port set and game ID per parallel match, and the same ID
  for both teams in that match. Use a separate log directory per
  configuration. Baseline Exp5 logs have schema 2; the optional Exp5-S1 live integration
  writes schema 4 with separate raw/decision positions. A complete log has
  `final: true` with `status: "ok"` in `<game_id>/<side>_<unum>/meta.json`.
- Score `PlayOn` with valid fullstate. Give each eligible sample equal
  weight. Group by true distance, visual age, estimate age and visibility.
  Join on cycle and stopped counter, and score next-cycle predictions only
  across consecutive `PlayOn` cycles. Keep train, validation and test
  separated by whole game.
- For self, use position, velocity, looking-direction and raw visual-fix
  errors from `agent_cycle`. For ball and players, separate raw sensor,
  localized visual and fused track errors from `objects`. Check ambiguous
  matches before interpreting raw player error. Compare means and tail
  errors such as p90 on equivalent samples.
- A decoded teammate report can be scored against a fresh visual estimate
  or a prior-cycle local estimate for the same target. That comparison
  describes message quality. A claim about fusion needs before-and-after
  instrumentation or a controlled run such as B0 versus S0.

The original matrix aimed for about 100 games per configuration. Use a short
smoke run to check settings, valid truth and finalized logs before starting
a batch.

## Exp2 handoff

The deployed model replaces only self position during decision-making in
quantized `PlayOn`. It keeps librcsc velocity and orientation. Visual ball
and player localization has already run when the correction reaches the
decision model. Moving the correction earlier would change the object
localization inputs and would be a separate experiment.

The 100-game live run reduced paired self-position error by 27.2%. It
showed little change in aggregate ball and player position errors and no
clear match-result effect. The active team always played on the left.
Keep that side assignment in mind when comparing scores across batches.
See the [live results](exp2/reports/steps/07-live-100-games.md) and
[comparison report](exp2/reports/comparison.html).


## Action arrival order in live model tests

Synchronous mode does not make all commands simultaneous. Some commands
are handled immediately on arrival; an intentional foul can mark a victim's
command done before its kick is handled. The [F4 60-game audit](exp4/reports/05-live-f4-60-games.md)
records slower treatment decisions and many more same-cycle foul-suppressed
kicks, despite lower perception error. Measure raw send timing, sense-body
acknowledgements and charge expiry, and use a time-matched control when
separating model benefit from computation cost. Do not silently change
server physics to erase this effect. Fullstate supplies evaluation groups
only; preserve causal observation-only inputs.

## Exp4 closeout (2026-11-15)

The [ball-stability closeout](exp4/reports/09-closeout-ball-stability.md)
records observer left 9's cycles 784–802 and the saved stage-by-stage evidence.
Quantized 360° vision still omits distant ball-motion fields when
see_no_fade_all=false. The current ball model can provide an inaccurate
valid velocity during those omissions and does not eliminate position
jumps. Retain Step26 corrected player positions as the strongest accuracy
candidate; ball correction remains experimental. Player velocity/body match
benefit is not isolated. The then-available randomized batch contained 35
finalized games, not the planned 200. No runtime change was made at closeout.

## Exp3/Exp4 archive and next scope

Use the [combined handoff](exp3-exp4/README.md) for numerical gains,
game-result limits, source-branch locations and the proposed visual-gap
experiment. Librcsc/Helios model implementations are preserved on
denoise-exp3-exp4; their source checkouts were returned to denoise.
Retrieve logging selectively and rebuild before new collection. The next
focus is accurate visible measurements with finite views and prediction
while targets are unseen. Existing see_noise_free is distance-only, so
verify the observation model before calling that experiment noiseless.

## Exp5 baseline prepared

[Exp5](exp5/README.md) adds an opt-in see_no_quantization mode and exact-mode
client localization, preserving finite FOV/cadence. Its tested pilot has
zero logged raw/fresh localization quantization error; error grows while
targets are unseen. Both baseline teams use no neural models. The first
100-game collection uses adaptive views, no fading/player payloads and
reference-only fullstate. The native schema-2 logger remains sufficient.
Analyze before selecting a learned gap predictor.

## Exp5 200-game live baseline (2026-11-15)

The [full visual audit](exp5/reports/02-live-200/report.html) covers 200 finalized games, 4,400 observers and 21,470,966 PlayOn decisions. All 819,660,850 checked raw/fix scalar readings were zero at stored precision; hearing logs contained no payloads and fullstate was reference-only. Both teams are baseline, with no neural correction.

Position-count-1 errors: self 1.8406 cm, ball 19.1122 cm, teammate 24.5088 cm, opponent 24.0665 cm. At count 5: ball 302.6543 cm, teammate 112.0816 cm, opponent 116.4721 cm. Means weight valid samples equally.

Do not equate receiving see with a self-position fix, a count-0 player track with confirmed identity, or a fresh ball position with fresh velocity. Anonymous near-field updates can retain the wrong uniform number; an exact position-only ball sighting can retain a badly outdated velocity. Player velocity/body fields are not logged once invalid (default counters >=5 / >=2), so unavailable old-field errors cannot be reconstructed from the saved WM columns. The report includes concrete cases, counter-conditioned view/distance plots, validity and error tails.

Use live_audit.py for current WM counters and verify_run.py for finalized metadata/hearing checks. Original analyze.py measures separate causal baselines using time since the last identified sighting and resets across PlayOn boundaries; do not mix that age definition with WM counters. Preserve confirmed fresh measurements and field/identity masks in the next learned predictor; use whole-game splits and no truth inputs.

## Exp5-S1: position prediction comparison

[Exp5-S1](exp5/exp5-s1/README.md) builds observation-only 32-cycle histories for all 200 games/4,400 observers. Whole-game split is 160 train, ten selection, ten confidence calibration and 20 test. Three 64-unit GRUs predict current position only: deterministic, full-covariance Gaussian, three-component mixture. All use identical features and sampled endpoints; true states are labels only.

On 28,180,368 paired held-out target-cycles, stock error 129.43 cm became 57.22 / 65.47 / 73.68 cm. The strongest point predictor reduced the mean by 55.79%, improving 66.30% and worsening 27.87% of rows (1 cm tolerance). Constant velocity alone scored 113.52 cm. All three improved pooled per-game means in all 20 games; this is one seed and is not a match-performance result.

Do not apply the raw predictor universally: the single model worsened 77.31% of already-accurate (<10 cm stock error) rows, and its worst added error was 16.56 m. Current anonymous observations can be accurate despite no named sighting; preserve this distinction. A 25 cm capped/gated policy scored 117.44 cm with bounded added position error. Gaussian/mixture calibrated 90% moment ellipses covered about 89.65%/89.63%; mixture density/NLL was better, but point error was worse. Next candidates are uncertainty around the accurate point predictor, observation-preservation guards and replicated training seeds, before live integration. Full metrics, exact cases, code, TensorBoard logs and frozen splits are linked from the step.

## Exp5 closeout and cache-v2 qualification

[Final report](exp5/reports/final/report.html) · [SoccerWindow one-game guide](exp5/debug/README.md).

The frozen visual-gap model reduced point error substantially, but both old200-game live batches used an overlay refresh that duplicated derived player containers and could rerun identity inference. Cache-v2 fixes those refreshes and resets current kickable pointers before rebuilding. Earlier live scores are historical measurements, not a clean test of the intended predictor. The corrected build first passed short correctness/display checks; the subsequent corrected-cache benchmark is recorded below. New metadata: player_gap_integration=cache-v2.


### Corrected-cache preserve follow-up · 2026-11-18

[New 200-game report](exp5/exp5-s1/reports/live-preserve-cache-v2-200/report.html):
all 4,400 records identify cache-v2, left apply/preserve versus right off,
635–684 goals and 78/22/100 W/D/L. Below 50 m on eligible stale-player rows,
mean error improved 142.32→53.78 cm, but rare severe errors remain and no gameplay
gain is established. The new cache-v2 zero batch is pending. The user removed
old live logs and reused the preserve directory name; analysis must validate
source fingerprints and use a new dated cache, not the old preserve-200 cache.
Native/PyTorch replay and reconstructed truth-position RCG examples are included
in the new report. Do not treat reconstructed clips as original action logs.


## Corrected-cache zero review and half policy · 2026-11-18

[Complete comparison](exp5/exp5-s1/reports/live-count-cache-v2-200/report.html) · [Half launch/checks](exp5/exp5-s1/reports/half-count-policy.md).

Zero completed **678–702 goals, 78/35/87 W/D/L**, versus **635–684, 78/22/100**
for preserve. Both have 200 games /4,400 finalized cache-v2 observers. Below
50 m, eligible stale-target error means are **53.99 /53.78 cm**, respectively;
added error >25 cm remains in **3.79% /3.77%** of target-cycles. Zero was correctly
applied; no gameplay gain over stock is established. Its observed +0.125 goals
difference/game versus preserve has independent-game bootstrap 95% interval
[-0.365,+0.610], including zero. Seeds/trajectories are different.

The requested **half** policy is now available: max(1,floor(original raw
posCount/2)), on applied overlays only. Preserve remains default. Native/live
correctness checks exist; no half gameplay benchmark has completed.


## Exp5 closeout · all three corrected-cache policies completed

[Short all-step closeout](exp5/reports/final/report.html) ·
[600-game detailed comparison](exp5/exp5-s1/reports/live-count-cache-v2-200/report.html).

Half completed 200 games: 656–696 goals, 86/28/86 W/D/L, 286–286 points.
Its eligible stale-player mean error below50m changed142.87→53.70cm, versus
53.78cm preserve and53.99cm zero. Count/routing checks and sampled native/Python
output parity passed. Same frozen bundle, no runtime code/model changes for
this analysis. No policy has established a gameplay advantage: half balances
points, zero has a less negative goal margin, and independent-game uncertainty
does not isolate a policy effect.

Retain observation-only prediction, raw-input isolation, age/identity masks,
cache-v2 invariants and visual tools. Do not equate lower mean error with safer
decisions, selector score with calibrated confidence, or an inferred position
with a true sighting. The compact closeout covers all stages; the original
detailed cache/debug handoff is archived under reports/final/integration-handoff.
