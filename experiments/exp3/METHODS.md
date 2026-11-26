# Exp3 · Player quantization · Methods and archive

[Experiment summary](README.md)

[Combined Exp3–Exp4 results and handoff](../exp3-exp4/README.md)

Exp3 uses 200 games with frequent observations and persistent player identity
to measure the error that remains in fresh, quantized player observations.
The [HTML report](reports/report.html) compares the 200 games with the 100
standard-configuration games in [Exp1](../exp1/README.md).

[Step summary and paired gains](reports/step-summary.md) · [Research roadmap](ROADMAP.md)

## Main results

- A high-quality 360° observation and reference fullstate were available at
  all **21,539,323** Exp3 PlayOn decisions. All **452,325,783** on-field player
  observations had an explicit identity, range and bearing. No teammate
  payloads were logged.
- Mean world-model player position error went from **1.595 m** in Exp1 to
  **0.800 m** in Exp3 across all available tracks. The combined configuration
  improved tracking coverage. Fresh raw error at 10–20 m was essentially
  the same in both runs, **39.60 cm** and **39.54 cm**.
- Exp3's raw relative position error averaged **78.1 cm**. Substituting
  true range alone left **13.2 cm**. Substituting true bearing alone left
  **75.5 cm**. These truth-assisted diagnostics prioritize range correction
  for the first filter.
- Motion fields were available for **55.7%** of Exp3 observations overall
  and for none at 40 m or more. On matched fresh Exp3 samples, the visual
  global estimate and the final world-model track had the same mean error.

## Steps

1. Run the same Helios baseline on both sides with different team names.
   Use synchronized mode, reference fullstate and quantized see. Set
   `server::full_view=true`, `server::see_no_fade_unum=true` and
   `server::say_msg_size=0`. Leave `see_no_fade_all` and
   `see_noise_free` disabled. Keep Gaussian see off in the clients.
2. Verify both teams have finalized player logs and that each PlayOn
   decision receives its visual observation. Count distance, bearing,
   identity and motion-field availability before measuring error.
3. Compare Exp1 and Exp3 within the same true-distance bins. Score raw
   relative positions, visual estimates and final tracks separately.
   Use fullstate only as an evaluation reference.
4. Use the range-versus-bearing diagnostic to prioritize the first
   position filter. Hold out entire games when evaluating it.

## Why start with a quantization-aware filter?

The [server distance calculation](../../src/visualsenderplayer.cpp) rounds on
a log grid, uses the observation focus, and rounds the resulting distance to
0.1 m. Protocol v19 rounds bearing to the nearest degree. Repeated readings
from unchanged geometry carry the same constraint. Motion of the observer
or target can reveal where the target lies within the quantization interval.

Start with a constant-velocity prior and a small set of candidate target
positions and velocities. Forward-quantize each candidate using the
observer's estimated pose and focus. Penalize candidates whose simulated
measurements disagree with the observed bins. Use the velocity fields when
present and retain their availability masks. Score the filter against a
plain constant-velocity tracker and the fresh librcsc visual estimate on
held-out games. Check range-binned relative and global position error.

## Temporal range experiment

See [Step 02](reports/02-temporal-range.md) for the filter and GRU comparison.
The sequence builder uses one observer per side per game and keeps all 21
targets per observer. Train, validation, and test are split by whole game.
The first recurrent model predicts a bounded correction along the line of
sight using an eight-cycle history.

From the server checkout, in the same Python environment as Exp2:

```sh
dataset=../logs/exp3-sequences
runs=../logs/exp3-runs
python3 experiments/exp3/code/build_sequences.py ../logs/exp3-logs "$dataset" --workers 24
mkdir -p "$runs"
python3 experiments/exp3/code/train_range.py "$dataset" "$runs/gru32-quantized-18" --anchor quantized --epochs 18
python3 experiments/exp3/code/train_range.py "$dataset" "$runs/gru32-blend-18" --anchor blend --epochs 18
python3 experiments/exp3/code/train_range.py "$dataset" "$runs/gru64-blend-18" --anchor blend --hidden 64 --epochs 18
# This checkpoint had the lowest validation mean in the recorded runs.
best_checkpoint="$runs/gru64-blend-18/best.pt"
python3 experiments/exp3/code/train_range.py "$dataset" "$runs/test-final.json" \
  --evaluate "$best_checkpoint" --stride 1 --threads 8
```

Choose new output paths for a repeat. Fullstate coordinates are loaded as
targets for offline evaluation and training. The baseline filters and GRU
features use only observed data and the observer's librcsc pose estimates.

## Reachability and context

The selected context GRU reduced held-out relative mean from **50.61 to
48.65 cm** versus Step 02. In the normal-view replay, only
**0.93%** of fresh identified observations had the required
history.


| Step | Change |
|---|---|
| [03 · Reachable intersection](reports/03-reachable-intersection.md) | Intersect the current observation sector with an expanded prior region. |
| [04 · Learned motion](reports/04-learned-motion.md) | Test a target-history GRU inside the region, then add a past-only displacement head. |
| [05 · Ball context](reports/05-ball-context.md) | Add the ball's estimated state from the preceding observations. |
| [06 · Nearby players](reports/06-nearby-players.md) | Add two nearby players on each team relative to the target. |
| [07 · Uniform-number embedding](reports/07-unum-embedding.md) | Add a categorical identity feature, ablate the motion head and score the selected model on test. |
| [08 · Normal-view replay](reports/08-normal-view.md) | Check how often the eight-consecutive-observation model can run in standard games. |

From the server checkout with the same environment used for Step 02:

```sh
dataset=../logs/exp3-sequences
sidecars=../logs/exp3-intersections
runs=../logs/exp3-context-runs
baseline=../logs/exp3-runs/gru64-blend-18/best.pt
python3 experiments/exp3/code/build_intersections.py "$dataset" ../logs/exp3-logs "$sidecars" --workers 26
mkdir -p "$runs"
python3 experiments/exp3/code/train_context.py "$dataset" "$sidecars" "$baseline" \
  "$runs/geometry-validation.json" --evaluate
for mode in region motion ball players unum unum_no_motion; do
  python3 experiments/exp3/code/train_context.py "$dataset" "$sidecars" "$baseline" \
    "$runs/$mode" --mode "$mode"
done
python3 experiments/exp3/code/train_context.py "$dataset" "$sidecars" "$baseline" \
  "$runs/validation-all.json" --evaluate "$runs"/{region,motion,ball,players,unum,unum_no_motion}/best.pt
# Winner by relative validation mean in the recorded runs.
winner=unum_no_motion
python3 experiments/exp3/code/train_context.py "$dataset" "$sidecars" "$baseline" \
  "$runs/test-final.json" --split test --stride 1 --threads 8 --evaluate "$runs/$winner/best.pt"
```

To reproduce the normal-view coverage and eligible-only replay:

```sh
normal=$(mktemp -d)
python3 experiments/exp3/code/check_normal_view.py ../logs/exp1 "$normal/coverage.json" \
  --dataset-output "$normal/sequences"
python3 experiments/exp3/code/build_intersections.py "$normal/sequences" ../logs/exp1 \
  "$normal/sidecars" --workers 10
python3 experiments/exp3/code/train_context.py "$normal/sequences" "$normal/sidecars" "$baseline" \
  "$normal/eligible.json" --split test --stride 1 --evaluate "$runs/$winner/best.pt"
```

## 360° quantization follow-up

[Steps 26–28](reports/step-summary.md) compare the two current-position readouts, then test whether a
one-cycle motion prior improves the validation-selected readout. They use
the same 160/20/20 game split and the same held-out position endpoints as
Step 19. The forecast evaluation uses the extra preceding observation when
available and otherwise falls back to the position-only model.

```sh
code=experiments/exp3/code
dataset=../logs/exp3-sequences
sidecars=../logs/exp3-intersections
runs=../logs/exp3-context-runs
base=../logs/exp3-runs/gru64-blend-18/best.pt
python3 "$code/train_context.py" "$dataset" "$sidecars" "$base" "$runs/step26-polar" --mode masked_set_polar
python3 "$code/train_context.py" "$dataset" "$sidecars" "$base" "$runs/step26-validation.json" \
  --evaluate "$runs/masked_set/best.pt" "$runs/step26-polar/best.pt"
python3 "$code/train_context.py" "$dataset" "$sidecars" "$base" "$runs/step26-test.json" \
  --evaluate "$runs/masked_set/best.pt" "$runs/step26-polar/best.pt" --split test --stride 1
for components in 1 3; do
  python3 "$code/compare_360_forecast.py" "$dataset" "$sidecars" "$runs/step27-28" --train "$components"
done
# Step 26 was selected by mean position error on validation.
python3 "$code/compare_360_forecast.py" "$dataset" "$sidecars" "$runs/step27-28" \
  --tracker "$runs/step26-polar/best.pt"
```

### Step 26 live test

[Step 29](reports/29-live-integration.md) records the implemented flags,
Python/C++ parity checks and smoke games.

Use the frozen Step 26 `masked_set_polar` checkpoint from
`../logs/exp3-context-runs/step26-polar/best.pt`. Reproduce its Python
outputs against a small set of raw logged windows before enabling apply
mode. Keep the first live comparison on the Exp3 quantized 360°
configuration, with identity preserved, hearing disabled and fullstate
used as a logging reference. This measures the effect of the same
quantization change in closed-loop play.

In librcsc, use the active player localization and track-update path
after observer pose estimation. Keep causal histories keyed by side
and uniform number and use stock localization until eight consecutive
identified PlayOn observations are available. Reset on gaps and
PlayOn transitions. Correct the measured player position before
decision caches are built. In Helios, own weight loading and expose
off, shadow and apply modes through `start.sh`. Shadow mode logs the
candidate alongside stock localization. Check Python/C++ parity in
both team coordinate frames.

Run a short shadow smoke test before the scored games. For the 100-game
comparison, use identical team builds and settings with the model enabled
for one team at a time. Run 50 with apply on the left and 50 on the
right. Pair seeds when the match runner can repeat them. Log the stock
position, model position, final world-model position, target identity,
eligibility and fallback reason, inference time, and fullstate reference.
Score relative and global position mean and p90 on paired samples,
including distance bins and all-cycle coverage. Report goals and
win/draw/loss by side separately from the position metrics.

### Decision-routing comparison

The [200-game F1 and 100-game F2 report](reports/31-live-results.md)
records the paired position gains and the actual match comparisons.

[Step 30](reports/30-decision-routing.md) compares the original Step 26
apply behavior with corrected position inputs to interception, pass and
dribble safety. Both sides use the same weights in apply mode. Add
`--player-position-routing corrected` to the treatment side only.

## Normal-view tracking

The [research roadmap](ROADMAP.md) records all 17 ideas and links each
completed step. Steps 09–13 reuse the same 100 normal-view Exp1 games,
with one observer per side per game and an 80/10/10 split by whole game.
The test split is held out from training, fallback selection, and gating.
The guarded tracker reduced paired test error from **167.71 to 165.60 cm**.
See [Step 13](reports/13-close-range-gate.md) for the comparison.
[Step 14](reports/14-directional-reachability.md) reached **165.50 cm**
with a directional update on fresh measurements. Step 13 remained
the validation choice for cycles without a measurement.
[Step 22](reports/22-teammate-fusion.md) reached **158.60 cm** with
full-precision teammate observations delayed one cycle.
[Step 24](reports/24-teammate-packet.md) reached **164.36 cm** under a
one-target packet model. These are offline replays. The packet result
is the priority for a live communication experiment.

Run from the server checkout in the Exp2 Python environment. Choose new
output directories to repeat a run.

```sh
events=../logs/exp3-normal-events
known=../logs/exp3-normal-known
associated=../logs/exp3-normal-associated
runs=../logs/exp3-normal-runs
python3 experiments/exp3/code/build_normal_events.py ../logs/exp1 "$events" --workers 24
python3 experiments/exp3/code/replay_normal.py "$events" "$known"
python3 experiments/exp3/code/replay_normal.py "$events" "$associated" --associate
for mode in gap_blend gap_quantized single; do
  python3 experiments/exp3/code/train_normal.py "$events" "$known" "$associated" "$runs" --train "$mode"
done
python3 experiments/exp3/code/train_normal.py "$events" "$known" "$associated" "$runs" --evaluate --threads 8
```

The event builder reads player measurements from the raw `see` messages.
It stores fullstate separately for training labels and evaluation. The
librcsc comparison uses tracks explicitly matched by uniform number.
The association audit counts anonymous observations with a unique match
when all true candidate states are forward-quantized. This audit uses
truth after assignment.

To repeat the directional reachability comparison using the same
replays and checkpoints:

```sh
kinematics=../logs/exp3-normal-kinematics
python3 experiments/exp3/code/build_normal_kinematics.py ../logs/exp1 "$events" "$kinematics"
python3 experiments/exp3/code/compare_directional.py ../logs/exp1 "$events" "$known" \
  "$associated" "$kinematics" "$runs" "$runs/directional.json"
```

To repeat [Step 15](reports/15-velocity-bins.md) with the same kinematics
and checkpoints:

```sh
python3 experiments/exp3/code/compare_velocity_bins.py ../logs/exp1 "$events" \
  "$associated" "$kinematics" "$runs" "$runs/velocity-bins.json"
```

### Follow-up replays

Use the same normal-view artifacts above for Steps 16–25. The
[step summary](reports/step-summary.md) lists each immediate baseline.

```sh
code=experiments/exp3/code
for readout in cartesian polar; do
  python3 "$code/compare_polar.py" "$events" "$known" "$associated" "$runs" "$runs/step16" --train "$readout"
done
python3 "$code/compare_polar.py" "$events" "$known" "$associated" "$runs" "$runs/step16"

for item in pose_uncertainty motion_paths; do
  python3 "$code/compare_${item}.py" ../logs/exp1 "$events" "$associated" "$kinematics" "$runs" "$runs/$item.json"
done
python3 "$code/compare_focus_geometry.py" "$known" "$events" "$runs/focus-geometry.json"
python3 "$code/compare_teammate_fusion.py" ../logs/exp1 "$events" "$associated" "$runs" "$runs/teammate-fusion.json"
python3 "$code/compare_teammate_fusion.py" ../logs/exp1 "$events" "$associated" "$runs" "$runs/teammate-packet.json" --packet

expanded=../logs/exp3-normal-expanded
python3 "$code/expand_normal_observers.py" ../logs/exp1 "$events" "$known" "$expanded"
python3 "$code/train_normal.py" "$expanded/events" "$expanded/known" "$associated" "$runs/expanded-matched" --train gap_quantized
python3 "$code/train_normal.py" "$expanded/events" "$expanded/known" "$associated" "$runs/expanded-long" --train gap_quantized --batches 512
python3 "$code/train_normal.py" "$events" "$known" "$associated" "$runs/original-long" --train gap_quantized --batches 512
python3 "$code/compare_polar.py" "$events" "$known" "$associated" "$runs" "$runs/step16" --additional-cartesian \
  "$runs/expanded-matched/gap_quantized.pt" "$runs/expanded-long/gap_quantized.pt" "$runs/original-long/gap_quantized.pt"
python3 "$code/compare_directional.py" ../logs/exp1 "$events" "$known" "$associated" "$kinematics" \
  "$runs" "$runs/individual-running-speeds.json" --individual-types
```

For Steps 19–20, use the 360° context commands above with the new modes
`masked_set`, `masked_set_no_unum` and `masked_set_role`. Use
`unum_no_motion` as the Step 07 control.


## Rebuild the report

Run from the server checkout with NumPy, pandas and Matplotlib installed.

```sh
python3 -m experiments.exp3.reporting.analyze_players \
  ../logs/exp1 experiments/exp3/reports/exp1-metrics.json
python3 -m experiments.exp3.reporting.analyze_players \
  ../logs/exp3-logs experiments/exp3/reports/exp3-metrics.json
python3 -m experiments.exp3.reporting.render_report \
  experiments/exp3/reports/exp1-metrics.json \
  experiments/exp3/reports/exp3-metrics.json \
  experiments/exp3/reports/report.html
```

The analyzer pools player samples with equal weight. All-track estimates can
include stale and heuristically matched tracks. Raw-error comparisons use
explicit uniform-number matches. Exp1 can have very few or zero identified
observations at long range, even when Exp3 has many in the same bin. The
comparison measures the combined configuration across different games.
