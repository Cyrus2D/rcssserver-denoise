# Exp6 · Gaussian visual noise · Methods and archive

[Experiment summary](README.md)

**Baseline complete.** Both teams used Gaussian in 200 games with the Exp3
360° observation setup. All 4,400 player logs finalized, and the 200 match
scores agree with AutoTest2D.

Gaussian lowered mean raw player range error from **75.52 to 59.64 cm**.
The stock localization step added a **+158.81 cm** mean radial bias, and
player WM position error rose from **80.04 to 170.60 cm**. Replaying the
stock movable distance table reproduced 409,563 logged localized ranges
in the sampled streams. The report also covers bearing, motion, ball,
landmark and self errors.

[Full Exp6–Exp3 comparison and charts](reports/report.html) ·
[Pooled metrics and scores](reports/metrics.json) ·
[Exp3 quantized baseline](../exp3/README.md)

## Step 1 · Gaussian range localization

Replaying the 200 original games with the same observer pose and reported
range reduced the global visual position error from **170.60 to 67.48 cm**
for players and from **158.15 to 62.47 cm** for the ball. The rebuilt librcsc
uses reported ranges after Gaussian acknowledgment. Short live controls
confirmed reported ranges in Gaussian and the table lookup in quantized mode.
Use the same server preset and team flag with the rebuilt library. Select
a new log directory for any new run. The player metadata now records
`gaussian_range_localization` for the active localizer.

[Step 1 report, replay and live verification](reports/range-replay/report.html)

## Step 2 · Corrected Gaussian live batch and denoising

AutoTest2D exp6logs3 contains **100 finalized games** with the revised library.
The full-batch mean player WM error was **66.38 cm** and mean signed range
error after localization was within 0.01 cm of zero.

First temporal filters were selected on 20 games and evaluated on 40 held-out
games. A measured-range gate and 25 cm cap were then confirmed on another 40
games. Below 50 m, the gated filters changed player error **55.58→47.20 cm**
and ball error **51.98→43.31 cm**. Close targets under 3 m kept the current
visual position throughout the confirmation sample. These are offline
position metrics on the recorded trajectories.

[100-game corrected baseline](reports/fixed-live/report.html) ·
[Temporal filters and independent confirmation](reports/temporal/report.html)

## Step 3 · Residual GRUs

The corrected Gaussian collection grew to 300 finalized games (6,600
observer logs). The additional 200 games supplied 160 training, 20 validation
and 20 new test games. The earlier frozen 40-game test stayed independent of
model fitting and selection.

Two small causal GRUs learned position corrections from eight observations
and a Kalman estimate. On the new test below 50 m, the guarded GRU reached
**43.48 cm** for players and **40.95 cm** for the ball. With the same gate and
cap, Kalman scored **47.28 cm** and **43.22 cm**, respectively. The earlier
40-game benchmark showed similar gains. Weights are under
`logs/exp6-analysis/learning-300/gru40`.

[Training and held-out results](reports/learning/report.html)

## Step 4 · Live left-team player GRU

In 200 live games, the 25 cm capped player GRU reduced the left team's mean
position error under 50 m from **55.67 cm to 43.47 cm**. The left
won 92, drew 31 and lost 77, scoring 845 to 793. Both teams used the corrected
Gaussian localizer and the right used standard player positions.

[Results, launch and verification](reports/live-gru.md)

## Build and launch

Use the `denoise` branches of all three repositories. Build librcsc and install
it to `/home/nader/local` before building Helios against that prefix. Helios
uses Eigen 3.3+ for native inference.
Helios accepts `--gaussian-see` for all players in a team. Schema 2 logs already
contain the raw observations, observer and focus geometry, field-presence masks,
localized/WM estimates, age counters and same-cycle truth. Librcsc also records
the current server's exact-observation flag and command-order settings in metadata.

Use the server binary at
`/home/nader/workspace/denoise/rcssserver-denoise/build/rcssserver`.
Pass the [server preset](server-baseline.conf) using
`include=/home/nader/workspace/denoise/rcssserver-denoise/experiments/exp6/server-baseline.conf`
or replace the observation settings in the runner with the equivalent flags:

```bash
-server::synch_mode=true -server::fullstate_l=true -server::fullstate_r=true
-server::full_view=true -server::see_no_fade_unum=true -server::see_no_fade_all=false
-server::see_no_quantization=false -server::see_noise_free=false -server::say_msg_size=0
-server::randomize_player_commands=false -server::command_order_log=false
-server::dist_noise_rate=0.0125 -server::focus_dist_noise_rate=0.0125
-server::land_dist_noise_rate=0.00125 -server::land_focus_dist_noise_rate=0.00125
```

In AutoTest2D's `match()`, replace the Exp5 observation-settings line with:

```bash
OPTIONS="$OPTIONS include=/home/nader/workspace/denoise/rcssserver-denoise/experiments/exp6/server-baseline.conf"
```

Keep the runner's ports, kickoff, match length and game IDs. The preset retains
Exp3's uniform-number visibility and the usual distance-dependent availability
of motion and body fields. Both exact-observation switches are false so the
server honors the player Gaussian command.

Use this command for both the `lt` and `rt` workers in AutoTest2D:

```bash
BASE_PATH=/home/nader/workspace/denoise/helios-base-denoise/build/bin
LOG_PATH=/home/nader/workspace/denoise/logs/exp6-logs
mkdir -p "$LOG_PATH"
cd "$BASE_PATH" && ./start.sh -h "$HOST" -p "$PORT" -P "$OLCOACH_PORT" -t "$TEAM" \
  --gaussian-see --fullstate reference \
  --denoise-log-dir "$LOG_PATH" --denoise-game-id "$GAME_ID" \
  --no-denoise-raw >"$RESULT" 2>&1
```

Keep landmark logging on for self-localization and landmark noise. The parsed
measurements needed for the comparison and model dataset are in
`objects.csv.gz`, `landmarks.csv.gz` and `agent_cycle.csv.gz`. Omit
`--no-denoise-raw` when inspecting the protocol in a pilot. Use the same log
path and game ID for both sides of each match. Use the rebuilt server on
AutoTest2D's path. For example, with a free starting port and 20 workers:

```bash
cd /home/nader/workspace/denoise/AutoTest2D
PATH="/home/nader/workspace/denoise/rcssserver-denoise/build:$PATH" \
  ./test.sh -l lt -r rt -p 16000 -t 20 -ro 10 -n exp6-logs
```

AutoTest2D stores results under `out/exp6-logs`. The agent CSV logs go to
`/home/nader/workspace/denoise/logs/exp6-logs`.

## Collection pilot

Two short matches on 2026-11-21 used the same server preset and rebuilt
binaries. Each finalized all 22 player logs. The quantized control omitted
the Gaussian flag on both teams.

| Pilot check in PlayOn unless noted | Gaussian | Quantized |
|---|---:|---:|
| Player acknowledgments of Gaussian mode | 22 / 22 | 0 / 22 |
| Decisions with 360° see and same-cycle truth | 968 / 968 | 1,496 / 1,496 |
| Named on-field player rows with raw range and focus truth | 20,328 / 20,328 | 31,416 / 31,416 |
| Named player rows with WM position count 0 | 20,328 | 31,416 |
| Named raw ranges off the 0.1 m grid | 20,312 | 0 |
| Named observations with both motion fields | 14,130 | 21,067 |
| Labeled landmark ranges | 53,240 | 82,280 |
| Logged player hearing payloads | 0 | 0 |

The pilot checks the collection configuration and available fields. The
200-game batch supplies the sample for comparing noise and training.

## Baseline measurements

1. Confirm 22 finalized players per game, Gaussian on both sides, reference-only
   fullstate, same-cycle `see` in PlayOn and no teammate payloads. Check raw
   identity/field coverage and the distribution of WM counters.
2. Compare signed and absolute distance error, bearing and motion fields,
   localization, and WM against Exp3. Use common true-distance bands and
   matching field-availability rules, with equal weight per eligible sample.
   For Gaussian, also check signed distance residuals against
   `sigma = distance * distance_rate + focus_distance * focus_rate`, using
   the rates from metadata. Keep landmarks separate from movable objects.
3. The archived Exp3 report and metrics support comparison on shared measures.
   If a comparison needs raw Exp3 rows that are unavailable, collect a
   quantized control with the same binaries and preset, omitting `--gaussian-see`.
4. Use whole games for the subsequent training and evaluation split. Compare
   transferred models with Gaussian baselines. Quantization-bin-specific
   features and likelihoods need a Gaussian counterpart, even when the
   surrounding temporal model can be reused.
