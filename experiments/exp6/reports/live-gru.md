# Exp6 · Left-team player denoising test

The local AutoTest2D `start_team` sets `EXP6_PLAYER_MODEL` to the epoch-7
weights and passes it to `lt`. Both sides use the Exp6 Gaussian range localizer and 360° preset.
The right team's decisions use the standard Gaussian world model.

The GRU uses eight current and previous visual measurements and a
position-only Kalman estimate. During PlayOn, it can move a freshly observed,
identified teammate or opponent by up to 25 cm. Targets with observed range
under 5 m keep their tracked position. Gaps and game-mode changes reset history.
The tracker keeps its original position, position count, velocity and body.
Ball and self estimates are outside this model. Fullstate is used by the logger
to evaluate the correction.

## Run 200 games in AutoTest2D

```bash
cd /home/nader/workspace/denoise/AutoTest2D
PATH="/home/nader/workspace/denoise/rcssserver-denoise/build:$PATH" \
  ./test.sh -l lt -r rt -p 16000 -t 20 -ro 10 -n exp6-logs4
```

The launcher passes `--gaussian-player-model FILE` to the left players only.
Both teams use `--gaussian-see --fullstate reference`. Player logs go to
`/home/nader/workspace/denoise/logs/exp6-logs4`, as configured by `LOG_PATH`
in `start_team`. Match results go to
`/home/nader/workspace/denoise/AutoTest2D/out/exp6-logs4`.

In `objects.csv.gz`, select `trk_decision_pos_applied=1` to pair the stock
position (`trk_stock_pos_x/y`) and the corrected position (`trk_pos_x/y`)
against the same truth. In `meta.json`, check `gaussian_range_localization`
on both teams and `player_position_corrector` on the left.

## Launch verification

A short mixed-side game finalized all 22 observers. Across the 11 left
observers, 48,763 fresh player positions received a correction and 902
observations under 5 m kept their original position. The right logged zero
corrections. Applied positions retained position count 0. The maximum
stock-to-corrected shift in the rounded CSV was 25.013 cm.

Native inference agreed with PyTorch within 0.000001 m on 4,096 held-out
training-format inputs. Replaying the 48,763 live corrections in PyTorch
agreed within 0.014 cm including CSV rounding. The full-batch results are below.

## Full batch - 200 matches

The left team's mean player position error below 50 m fell from
**55.67 to 43.47 cm (21.9% lower)** on
197,921,999 paired PlayOn observations. The mean fell in all 200 games.

### Match results

| Run | Games | Left W-D-L | Goals L:R | Points L:R | Left wins excluding draws |
|---|---:|---:|---:|---:|---:|
| Both teams standard Gaussian (exp6-logs3) | 300 | 119-37-144 | 1255:1341 | 394:469 | 45.25% |
| Left model, right standard Gaussian (exp6-logs4) | 200 | 92-31-77 | 845:793 | 307:262 | 54.44% |

All 200 AutoTest2D results agreed with all 22 observer logs per game. The GRU
run had +0.26 goals per game for the left (approximate 95% CI -0.12 to +0.64).
Its win rate excluding draws was 54.44% (95% Wilson CI 46.9% to 61.8%,
two-sided exact p=0.281 against 50%). The new batch favors the model side.
The two batches used separate games, and the model always played on the left.

### Paired positions in PlayOn

The paired comparison uses the same observer, target, cycle and fullstate
truth for both estimates. Every eligible row has the same weight. Rows
without a correction count equally in both columns. The table groups by
true distance from observer to target. The 5 m application gate uses
observed range.

| Distance or situation | Stock (cm) | Decision (cm) | Better by >1 cm | Worse by >1 cm | Worse by >10 cm |
|---|---:|---:|---:|---:|---:|
| Under 5 m | 8.57 | 8.43 | 1.6% | 0.3% | 0.0% |
| 5-10 m | 17.81 | 11.37 | 68.0% | 22.9% | 3.8% |
| 10-20 m | 33.28 | 22.08 | 76.5% | 18.9% | 5.9% |
| 20-35 m | 59.28 | 45.47 | 80.2% | 17.2% | 8.4% |
| 35-50 m | 90.36 | 76.97 | 78.5% | 19.7% | 12.2% |
| 50 m and over | 144.76 | 129.89 | 81.6% | 17.0% | 11.4% |
| All under 50 m | 55.67 | 43.47 | 76.4% | 18.3% | 8.1% |
| Teammates under 50 m | 55.69 | 43.32 | 77.4% | 18.4% | 8.1% |
| Opponents under 50 m | 55.64 | 43.61 | 75.5% | 18.3% | 8.1% |
| Opponents within 2 m of ball and under 20 m from observer | 27.24 | 18.61 | 66.9% | 20.3% | 5.7% |

### Checks

- All 4,400 observer streams finalized with acknowledged Gaussian localization.
  The left applied 221,031,850 corrections. The right applied zero.
- Every corrected row was a fresh named player in PlayOn with an observed
  range of at least 5 m. The maximum shift was 25.014 cm,
  consistent with the 25 cm cap at CSV precision.
- Right-side standard position error under 50 m was 55.64 cm. Left-side stock
  error on the same left observations was 55.67 cm. Ball and self were outside
  the correction and had mean left position errors of 61.14 cm and 3.74 cm.

[Full metrics, error counts and verified scores](live-200-metrics.json)

### Reproduce the audit

```bash
cd /home/nader/workspace/denoise
python3 rcssserver-denoise/experiments/exp6/reporting/analyze_live_gru.py \
  logs/exp6-logs4 AutoTest2D/out/exp6-logs4/result.d logs/exp6-logs3 \
  logs/exp6-analysis/live-200 --workers 32
```
