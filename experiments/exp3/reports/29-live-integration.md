# Exp3 Step 29 — Step 26 live integration and smoke tests

The Helios player loads the frozen Step 26 weights. librcsc maintains
the feature history, computes the range interval and reachability ray,
and applies the candidate after stock player matching and before
decision caches are built. Apply mode changes the tracked player
position. Stock localization records, seen positions and velocity
estimation remain available. The context snapshot is taken before
applying this cycle's corrections. The rest of the closed-loop world
model can respond to earlier corrections.

The feature runs in high-quality quantized 360° PlayOn with uniform
numbers preserved, normal motion-field fading and hearing disabled.
Each target needs eight consecutive observations. Gaps and play-mode
changes restart the history. The fullstate reference goes to the
dataset logger.

## Smoke tests

| Run | Observed behavior |
|---|---|
| Shadow on both sides | 77,616 candidates, zero applied. Stock positions retained. |
| Apply on left, off on right | 39,501 candidates applied on the left. Off side produced no candidates. |
| Off on left, apply on right | 22,638 candidates applied on the right. Off side produced no candidates. |
| Normal view with the model enabled | No candidates or corrections on either side. |

The C++ network matched Python within **0.007 mm** on 400 identical
recorded inputs from both sides. End-to-end comparison on 7,056 shadow
rows from two live observers gave p99 differences of **0.14 mm** and
**0.46 mm**. The largest individual difference was **1.32 cm**.
Per-target feature and inference time was about **0.27 ms at p95**
in the short shadow match. Throughput with 100 simultaneous matches
has yet to be measured.

## Running the scored games

Use the existing Exp3 server and match-runner settings. Keep
`server::full_view=true`, `server::see_no_fade_unum=true`,
`server::see_no_fade_all=false`, `server::see_noise_free=false`,
`server::say_msg_size=0`, and reference fullstate for both sides.

```sh
MODEL=/home/nader/workspace/denoise/logs/exp3-context-runs/step26-polar/player-position.weights
# Add to the treatment team's existing start.sh arguments.
--player-position-denoise apply --player-position-model "$MODEL"
```

Use `shadow` in place of `apply` for a log-only run. The control team
defaults to `off`. Use the same `--denoise-log-dir`, unique per-match
`--denoise-game-id` and `--fullstate reference` on both sides. Run 50
games with treatment on the left and 50 with treatment on the right.

The candidate appears in `objects.csv.gz` as `denoise_pos_x/y`.
Compare it with stock `ls_pos_x/y`, final `trk_pos_x/y`, and the
reference. Use `denoise_age`, `denoise_applied`, and `denoise_us`
to measure coverage and timing.

For a new shadow run, compare the logged predictions with Python:

```sh
python3 experiments/exp3/code/verify_step26_shadow.py \
  ../logs/exp3-context-runs/step26-polar/best.pt \
  "$LOG_PATH/$GAME_ID/l_2" "$LOG_PATH/$GAME_ID/r_2"
```

[Offline Step 26 results](26-bounded-bearing.md)
