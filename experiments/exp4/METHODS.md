# Exp4 · Player and ball state · Methods and archive

[Experiment summary](README.md)

[Combined Exp3–Exp4 results and handoff](../exp3-exp4/README.md)

Exp4 reuses the original 200 quantized 360° Exp3 games. It measures
current player velocity and body direction, current ball position and
velocity, and the ball's next-cycle position and velocity. The
[Exp3 live report](../exp3/reports/31-live-results.md) explains why
these estimates matter for reach, kickability and safety checks.

## Research phase closed · Ball stability audit

Player-position denoising remains the best-supported accuracy improvement.
Combined ball correction improves average error but can still miss kick
direction changes and retain quantization-driven position jumps. Observer
left 9's cycles 784–802 confirm these limits. No runtime changes were made.
The randomized batch currently has 35 finalized games; its match benefit
is not established.

[Closeout report and plots](reports/09-closeout-ball-stability.html) ·
[Evidence and methods](reports/09-closeout-ball-stability.md)

## Visual inspection · Stock versus used state

A separate debug build can show stock, applied, blocked model predictions
and reference truth in Soccer Window. Focus selects a ball/player pair;
overview adds position markers. It saves DCL replay data. Normal batch
launches remain unchanged.

[Launch and color guide](reports/08-debug-overlay.html) ·
[Implementation and verification](reports/08-debug-overlay.md)

## Random-priority batch · First 15 games

All 330 agent logs finalized correctly. Left-first command priority is 50.22%.
The previous foul/kick asymmetry is absent: new-charge kicks are blocked
49.3% on the left and 50.0% on the right. Sampled model gains and close-range
guards pass. Continue the 200-game batch.

[Initial live check](reports/07-random-order-first15.html) ·
[Methods and snapshot](reports/07-random-order-first15.md)

## Next batch · Random command priority

The server can now collect a complete synchronous cycle and randomly order
player action batches, removing the identified arrival-speed advantage.
The feature defaults to off. The build, 64-seed UDP regression, timeout tests
and full-team checks passed.

[Settings and test report](reports/06-server-command-order.md) ·
[Visual report](reports/06-server-command-order.html)

## F4 first 60 games

Live noise gains persist: ball velocity 32.49 → 24.64 cm/cycle and player
position 78.42 → 52.22 cm. Scores remain close to a tie. The audit finds
a serious foul/kick arrival-order disadvantage and more missed physical
ball-control threats. Prioritize timing and decision uncertainty.

[60-game report](reports/05-live-f4-60-games.html) · [Methods and scripts](reports/05-live-f4-60-games.md)

## F4 integration repair

The live history-feedback and ball-path issues have been repaired. Two
short apply matches, testing both sides, reduced ball velocity error
31.66 → 22.67 cm/cycle; close-range guards retained accurate stock values.
Four wiring smoke matches passed. The next full batch will measure outcomes.

[Fix and launch instructions](reports/04-live-f4-fix.md) · [Visual report](reports/04-live-f4-fix.html)

## Live F3 audit

The 155-game combined treatment did **not** establish a match advantage.
Player position improved 78.40 → 52.12 cm, while ball velocity worsened
31.84 → 32.80 cm/cycle in every game. Close-range velocity regressions,
feedback into estimation and inconsistent ball paths in interception
need attention before another combined trial.

[Visual report](reports/03-live-f3-audit.html) · [Markdown and reproduction commands](reports/03-live-f3-audit.md)

## First results

- **Player velocity and body.** The gated context policy reduced mean
  velocity error from 9.59 to 7.24 cm/cycle and body error from 21.97°
  to 14.74°. It preserves observed body direction.
- **Ball current state.** Quantization bounds reduced mean position
  error from 68.75 to 63.98 cm.
- **Ball forecast.** Separate controlled and free-ball heads reduced
  next velocity error from 40.59 to 33.97 cm/cycle.

[Full report and subgroup results](reports/01-state-models.md) · [Combined live test and flags](reports/02-live-integration.md)

## Data and method

- Use the original 160 training, 20 validation and 20 test games.
  Each game contributes the seeded Exp3 observer from each side.
  Keep uninterrupted eight-cycle PlayOn windows. Forecasts also
  require a consecutive next cycle. Weight each scored row equally.
- The target's history includes stock position, velocity, body estimate,
  estimate ages and validity, raw range and bearing, available raw
  motion fields, and observer pose and motion. The model predicts
  residuals around the stock estimate. Body direction is a circular
  output. Missing stock velocity falls back to zero. Missing stock
  body falls back to the estimated movement direction, or 0° if
  velocity is unavailable. These fallbacks define the stock comparison
  on rows where the stock field is missing.
- The context model adds masked, pooled histories of the two closest
  players from each team. Slots carry relative position and velocity,
  body direction, age and validity. Ball context can include a
  categorical identity for the nearest estimated player in kicking
  range. The shared model is compared with the same architecture
  with that identity masked. The two-head model chooses a controlled
  or free-ball head using the estimated control cue.
- The ball baseline extrapolates its stock position by its current
  velocity and applies server decay for next-cycle velocity. The ball
  quantization variant constrains the current position readout to the
  estimated range interval and a half-degree bearing correction. The
  range interval is computed along the raw bearing and retained after
  rotation. The next-cycle predictions remain residuals around the
  coasting baseline.
- The estimated control cue is proximity within the server's nominal
  kickable radius or the observer's own kickability flag. It describes
  a possible kicker. Fullstate supplies labels and evaluation groups,
  including true geometric kicking range and whether next velocity
  differs from coasting by more than 0.2 m/cycle. That last group can
  include kicks, tackles or collisions.

## Reproduce

Run from the server checkout in the same Python environment as Exp3.
Create a new dataset directory for a repeat.

```sh
code=experiments/exp4/code
dataset=../logs/exp4/dataset
runs=../logs/exp4/runs

python3 "$code/build_dataset.py" ../logs/exp3-logs ../logs/exp3-sequences/manifest.json "$dataset"
PYTHONPATH=experiments/exp3/code python3 "$code/build_ball_bounds.py" ../logs/exp3-logs "$dataset"

for variant in player_history player_context ball_history ball_context ball_experts ball_quantized ball_context_no_owner; do
  python3 "$code/train_state.py" "$dataset" "$runs" --train "$variant"
done
python3 "$code/train_state.py" "$dataset" "$runs" --evaluate player
python3 "$code/train_state.py" "$dataset" "$runs" --evaluate ball --include-quantized --include-owner-ablation
```

The models use a 64-unit GRU, AdamW, seed 42, eight epochs and 36,864
sampled windows per epoch. Checkpoints and the body gate are selected
on validation. Model weights and the dataset live under
`/home/nader/workspace/denoise/logs/exp4`.


To export the frozen live candidates:

```sh
python3 experiments/exp4/code/export_models.py ../logs/exp4/runs ../logs/exp4/weights
```
