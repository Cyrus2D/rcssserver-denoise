# Exp5 · Predict players during visual gaps · Methods and archive

[Experiment summary](README.md)

**Complete: strong position-error reduction, no established gameplay gain.**
Exact visual measurements with adaptive finite views isolate the problem of
unseen motion. We collected 200 baseline games, compared/refined position
predictors, confirmed frozen policies on ten independent baseline games, and
completed 600 corrected-cache live games across three decision-age policies.

[Short closeout: all steps, achievements and lessons](reports/final/report.html) ·
[600-game live comparison and audit](exp5-s1/reports/live-count-cache-v2-200/report.html) ·
[Full model/refinement study](exp5-s1/reports/report.html) ·
[200-game baseline report](reports/02-live-200/report.html) ·
[SoccerWindow commands](debug/README.md).

## Final live results · cache-v2

| Count policy | Original→used error cm | Added error >25 cm | Goals L–R | W / D / L |
|---|---:|---:|---:|---:|
| preserve | 142.32→53.78 | 3.77% | 635–684 | 78/22/100 |
| zero | 144.00→53.99 | 3.79% | 678–702 | 78/35/87 |
| half | 142.87→53.70 | 3.80% | 656–696 | 86/28/86 |

Each policy has 200 games. Errors cover PlayOn, valid identified stale WM,
named-sighting age 1–30 and true observer distance below 50 m, with equal weight
per target-cycle. These are different trajectories, not paired games.
All three reduce mean error by about 62%; none establishes a team-strength
advantage. Half balances points/W–L; zero has the least negative goal difference.
The same weights/selector are used throughout.

## What we completed

1. **Isolated and audited the baseline:** exact visible kinematics, adaptive
   60°/120°/180° views, no hearing payloads/fading. Fullstate is labels only.
2. **Compared three causal models:** point /Gaussian uncertainty /three-position
   mixture. On the same held-out 28.18M rows, WM129.43 cm became57.22 /65.47 /73.68 cm.
3. **Refined accuracy and safety:** capacity/training/data controls, uncertainty
   gates, model-versus-WM selectors, readable trees, harm-aware residual GRU128,
   correction caps/blends and independent new-game confirmation.
4. **Integrated and inspected live:** client-owned model inference, decision-only
   overlays, preserved raw inputs, numerical checks and colored per-observer
   SoccerWindow logging. Fixed an important duplicate-cache integration defect.
5. **Tested decision age:** preserve, zero and half over 200 games each after
   the cache-v2 fix. Position accuracy remains similar; gameplay benefit is unproven.

The final frozen practical model predicts current player x,y only from51 causal
features ×32 cycles plus categorical uniform-number input. It leaves self,
ball, player velocity and body unchanged. Prediction is opt-in/off by default.
[Step-by-step model record](exp5-s1/README.md) ·
[Half-count launch/verification](exp5-s1/reports/half-count-policy.md).

## Historical results and important caveat

The earlier400 live preserve/zero games used a defective cache refresh;
their scores cannot cleanly evaluate the intended policy. Retain their
[archived comparison](exp5-s1/reports/live-zero-200/report.html) and
[detailed cache/debug handoff](reports/final/integration-handoff.html).
The new600 cache-v2 games above are separate, fully reviewed batches.
[Corrected preserve failure clips](exp5-s1/reports/live-preserve-cache-v2-200/report.html).

## Baseline collection (200 games)

Use the rebuilt normal executable directory:

`/home/nader/workspace/denoise/helios-base-denoise/build/bin`

Use both teams with the same adaptive policy. Preserve the runner's unique
ports and GAME_ID per match. Add these to the **server** launch:

```sh
-server::synch_mode=true \
-server::full_view=false \
-server::see_no_quantization=true \
-server::see_noise_free=false \
-server::see_no_fade_unum=true \
-server::see_no_fade_all=true \
-server::say_msg_size=0 \
-server::fullstate_l=true \
-server::fullstate_r=true \
-server::randomize_player_commands=true \
-server::command_order_seed=-1 \
-server::command_order_log=true
```

The same settings are in server-baseline.conf. The plain server::key=value
syntax also works. Existing see_noise_free is distance-only; it is not a
substitute for see_no_quantization.

For either team, adapt your existing worker launch as follows:

```sh
BASE_PATH=/home/nader/workspace/denoise/helios-base-denoise/build/bin
LOG_PATH=/home/nader/workspace/denoise/logs/exp5-logs

cd "$BASE_PATH"
./start.sh -p "$PORT" -P "$OLCOACH_PORT" -t "$TEAM" \
  --fullstate reference \
  --view-policy adaptive \
  --denoise-log-dir "$LOG_PATH" \
  --denoise-game-id "$GAME_ID" \
  --no-denoise-raw --no-denoise-landmarks \
  >"$RESULT" 2>&1
```

Here TEAM is lt or rt. Raw visual values needed for player prediction are
already in objects.csv.gz. The two --no-denoise-* flags save space by
omitting full packet traces and landmark tables; omit those flags for a
small diagnostic run when full protocol/landmark logs are desired.
All players still write agent_cycle, objects, hear, extra_tracks and
metadata, including per-cycle view, observation ages, commands and truth.
Do not pass old player-position/Exp4 model flags.

No fixed view is imposed in this first baseline. Optional --view-policy
narrow, normal or wide permits separate controlled cohorts later. It fixes
width after other view/neck decisions; native 1/2/3-cycle cadence remains.
Do not mix view-policy cohorts under the same report without separating them.

## Analyze completed logs

From /home/nader/workspace/denoise:

```sh
python3 rcssserver-denoise/experiments/exp5/analyze.py \
  logs/exp5-logs logs/exp5/reports/baseline-200-motion-baselines --workers 4
```

The script selects only games with 22 finalized status-ok observer logs.
It rejects fullstate-as-input, 360° mode, enabled hearing payloads or the
wrong sensor mode. It also checks that raw kinematic quantization errors
are zero to the stored precision. Incomplete games are listed and skipped.

Output report.html shows the age-conditioned baseline errors. metrics.json
preserves the exact game list, sample counts, coverage, view/distance
subgroups and methods. Every eligible target-cycle is weighted equally.
Current view width alone is descriptive; gap age is the primary comparison.

The causal baselines are stock WM, last observed position, constant velocity,
and velocity with server-default decay. See the pilot report for validity,
anonymous sightings and cohort exclusions. The p90 histogram resolution is
1 cm for position; tiny sensor errors should be checked through their
exact mean/max rather than the histogram.


## Detailed live-data report

Use the extended audit for self, ball, teammates and opponents, with
position counts 0, 1, 2, 3, 4, 5 and >5. Velocity/body are shown both
against their own counters and against position count. View-width
heatmaps, visibility/age distributions, distance groups and large-error
rates separate sensor precision from missing-information errors.

```sh
cd /home/nader/workspace/denoise
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 rcssserver-denoise/experiments/exp5/live_audit.py logs/exp5-logs logs/exp5/reports/baseline-200 --workers 8 --expected-games 200
```

The result is report.html, embedded SVG charts, report.md, tables.csv and
metrics.json with the frozen game list and sample counts. Finalized
per-game caches allow later runs to reuse completed work. Optional
--prepare-only creates caches during collection without publishing a
partial report. Sensor-audit failures remain visible in the report;
inspect them before training.

Current named tracks with anonymous updates remain in the state-error
means: wrong retained identity is a real live WM error. Raw sensor
exactness uses only unambiguous current identified observations. A see
update need not yield a new self-position fix, and body/velocity counters
can represent inferred fields rather than direct measurements.

The original analyze.py remains available for paired causal player-motion
baselines (WM, hold, constant velocity, default decay); it uses a stricter
last-identified-sighting age within uninterrupted PlayOn blocks. That age
is not interchangeable with the WM counters in live_audit.py.

Check actual hearing logs as well as metadata:

```sh
python3 rcssserver-denoise/experiments/exp5/verify_run.py logs/exp5-logs logs/exp5/reports/baseline-200/run-audit.json --expected-games 200
```

Run that before live_audit.py to include its verification in the HTML.

Regression checks:

```sh
python3 rcssserver-denoise/experiments/exp5/test_live_audit.py
python3 rcssserver-denoise/experiments/exp5/test_analyze.py
```

## What stays uncertain

The server's actuator/physics randomness and other players' unknown future
commands remain. They are prediction uncertainty, not measurement
quantization. Vision still has its native near-field anonymous observations.
Self pose can drift between visual fixes. All of these are logged rather
than replaced with hidden fullstate inputs.

Use whole-game train/validation/test splits in the later dataset. A model
should update from available observations, retain missing-field masks and
age, and be scored on unseen-player cycles. Fresh exact positions should
not be denoised again. First inspect the 200-game baseline before choosing
architecture or training a model.

## Reproduce the checks

```sh
python3 rcssserver-denoise/experiments/exp5/test_analyze.py
python3 rcssserver-denoise/experiments/exp5/run_smoke.py another-pilot
python3 rcssserver-denoise/experiments/exp5/verify_logs.py logs/exp5/smoke/another-pilot
```

Use a fresh pilot name and --port when needed. Analysis requires numpy,
pandas and matplotlib (requirements.txt); they are available in the current
system Python. The training environment is not needed yet.

## Optional Exp5-S1 live trial

The baseline collection above remains model-off. An opt-in live implementation
is now built in both sibling repositories. See the [complete launcher commands,
counter policy and verified checks](exp5-s1/reports/live-integration.md).
Start with left-only apply and preserved real observation age; compare zero
decision age in a separately labelled batch. The same Exp5 server preset applies.

## Live preserve-count result (200 games)

[Live report](exp5-s1/reports/live-preserve-200/report.html): below50m, position error141.55→52.62cm, but goals628–705 and75/30/95 W/D/L. Integration checks passed; current-point accuracy alone did not improve gameplay. A separate zero-count trial is the next comparison.
