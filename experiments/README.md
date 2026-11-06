# Denoise Experiments

## Goal
Measure how much each sensor factor adds to **self**, **ball** and **player** observation noise
in RoboCup 2D, so we can decide where to focus denoising work.
Method: run many games (~100 per configuration) over the matrix below, log every agent with
ground truth, and compare the precomputed noise columns (`n_* = observed - truth`).

## What exists (status 2026-11-05)

| Repo | Change |
|---|---|
| rcssserver-denoise | 4 new server params (below); `ExactObservation` for noise-free see; fade guards in the v18+ see serializer |
| librcsc-denoise | Client support for the new params (360° view, see every cycle, noise-free localization); `DenoiseLogger` per-agent dataset (CSV.gz, schema v2) |
| helios-base-denoise | Behaviour/action labels (`pre:*`, `role:*`, `plan:*`, `exec:*`); view logic skipped under `full_view`; `start.sh` denoise options; `player.conf` `version : 19` |

Dataset columns: `../../librcsc-denoise/.github/instructions/denoise-dataset.instructions.md`.

## New server params (all bool, default `false`, protocol v19)

| Param | Effect |
|---|---|
| `server::full_view` | In `synch_mode`: 360° view cone and a `see` every cycle. The client keeps its width setting and adapts. |
| `server::see_no_fade_unum` | Team + unum always shown (no distance fade); vel/body/head still fade. |
| `server::see_no_fade_all` | Everything always shown (team, unum, vel, body, head; pointto if pointing). |
| `server::see_noise_free` | No distance noise (exact distance). Direction stays int; dist_chg/dir_chg keep normal rounding. |

Notes:
- Fade flags affect far objects in high-quality view (v18+). Close objects (<=3 m outside the cone) and low quality are unchanged.
- Say off = `server::say_msg_size=0` (`hear_max` does not limit anything).
- Gaussian see is chosen by the **client**: put `gaussian_see : on` in helios `src/player.conf` (no start.sh flag).
- `see_noise_free` wins over Gaussian (it is selected first).

## Required settings for every run

Server:
```sh
rcssserver server::synch_mode=true server::auto_mode=true \
  server::fullstate_l=true server::fullstate_r=true \
  server::game_log_dir=<out>/rcg server::text_log_dir=<out>/rcl \
  server::team_l_start=<left start.sh> server::team_r_start=<right start.sh> \
  [matrix params ...]
```
Client (helios, each team):
```sh
./start.sh -p 6000 -t <team> --fullstate reference \
  --denoise-log-dir <out>/dataset --denoise-game-id <same id for both teams>
```
- `--fullstate reference` keeps the truth world model separate; decisions still use the noisy model.
- Output: `<out>/dataset/<game_id>/<side>_<unum>/*.csv.gz` + `meta.json` (`final: true` only after a clean close).
- Use a different port set per parallel game; run each config about 100 times.

## Experiment matrix (first round)

Baseline = default server (quantized see, normal view, fade on, say on).
Change one factor at a time first; combine later.

| ID | Noise | View | Fade | Say | Server / client changes | Question |
|---|---|---|---|---|---|---|
| B0 | quantize | normal | on | on | - | Baseline noise level |
| N1 | none | normal | on | on | `see_noise_free=true` | How much is pure distance noise? |
| G1 | gaussian | normal | on | on | `gaussian_see : on` (player.conf) | Gaussian vs quantize |
| V1 | quantize | full 360 | on | on | `full_view=true` | How much comes from not seeing (age / out of view)? |
| F1 | quantize | normal | unum only | on | `see_no_fade_unum=true` | Cost of unknown unum (matching / ID) |
| F2 | quantize | normal | none | on | `see_no_fade_all=true` | Cost of missing vel/body/head |
| S0 | quantize | normal | on | off | `say_msg_size=0` | How much does communication help? |
| U0 | none | full 360 | none | on | `see_noise_free=true full_view=true see_no_fade_all=true` | Lower bound (almost perfect info) |
| GV | gaussian | full 360 | on | on | `full_view=true` + `gaussian_see : on` | Gaussian without view limits |

## What to measure (per config, split by self / ball / players)
- Self: `n_est_pos_d`, `n_est_vel_d`, `n_est_face`, `n_rawfix_pos_d` (table `agent_cycle`).
- Ball/players: `n_dist`, `n_dist_rel`, `n_dir` (raw see), `n_ls_pos_d` (one see), `n_trk_pos_d`, `n_trk_vel_d` (fused) (table `objects`).
- Bin by true distance (`true_dist`), by age (`trk_pos_count`) and by `seen` / `in_cone`.
- Factor effect = metric(config) - metric(B0); report mean, median and p90.
- Use only rows with `truth_valid=1`; filter weak matches with `match_ambiguous` / `match_err_deg`.

## Open items
- Should the no-fade flags also cover close (<=3 m) objects?
- Batch runner script for the matrix (ports, ids, output dirs).
- Optional `--gaussian-see` option in `start.sh`.
- Not built or run yet: build on Linux and do a short smoke game first (check `meta.json` `final: true`).

## Generate a play-on report

Install NumPy, pandas, and Matplotlib in your Python environment if needed.
Run this from the server checkout:

~~~sh
python3 experiments/report_denoise.py /home/nader/workspace/denoise/logs/exp1 --workers 24
~~~

The report defaults to `experiments/reports/<log-directory-name>/`. Use
`--output PATH` to choose a different directory. Pass one game directory
for a quicker preview. Each agent must have finalized schema-v2 logs
with fullstate separate from the decision world model.

- `report.html` includes plots and expandable tables for self, ball,
  teammates, and opponents. It covers true distance, view width,
  visibility, visual age, fused-estimate age, and recorded teammate reports.
  The visual-gap summary includes all distances and targets strictly under
  50 m from the observer, based on server ground truth.
- `metrics.csv` contains the pooled means, approximate medians and p90s,
  signed bias for scalar residuals, and metric coverage by group.
- `run.json` records the source, run settings, and coverage checks.

Every eligible sample contributes equally to its metric. Object rows
are joined to the agent's `play_on` decision at the same cycle and stopped
counter. One-step predictions are scored against fullstate at exactly the
next cycle within uninterrupted `play_on`. Each prediction is paired
with a persistence baseline using the same rows. The hearing breakdown
describes association under the current communication policy. Compare
separate `B0` and `S0` reports to measure the effect of disabling it.

The hearing section compares each decoded teammate position with a local
reference for the same target and cycle. For a seen target it uses the
fresh visual position. For an unseen target it uses the previous
ball forecast or the previous known player position. It groups the
results by message type, visual age and target distance. A gain or
loss larger than 5 cm counts as better or worse. These comparisons
evaluate the reported positions against those references. Actual
fusion gains require before-and-after logging or a controlled run.
