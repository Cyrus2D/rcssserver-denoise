# Step 07 — Self localization in 100 live games

Helios ran with the Step 05 GRU in `apply` mode on the left and with
standard librcsc on the right. All 2,200 player logs finalized. The
model log was present for all 1,100 left-side players and absent on the
right. The run used quantized observations and reference-only fullstate.

## Did the model improve the left team's self position?

Yes. On the **5,015,279 PlayOn player-cycles where the model was applied**,
its mean error was **3.145 cm**, compared with **4.317 cm** for the
sensor-based librcsc estimate recorded for the same player and cycle. The
reduction was **27.2%**, and the game-level mean improved in all 100
games. Resampling games gave a 95% interval of **1.165 to 1.179 cm** for
the mean reduction. Each player-cycle has equal weight in the estimate.

For each position, error is the Euclidean distance from that position
to the server's fullstate position at the same cycle. Both estimates
in each pair came from the same active game and share its action history.

| Age of last visual position fix | librcsc mean | Model mean | librcsc p90 | Model p90 |
|---|---:|---:|---:|---:|
| All scored ages | 4.317 cm | **3.145 cm** | 8.657 cm | **6.260 cm** |
| 0 cycles, fresh fix | 4.460 cm | **2.996 cm** | 9.167 cm | **5.880 cm** |
| 1 cycle | 4.169 cm | **3.330 cm** | 8.024 cm | **6.619 cm** |
| 2 cycles | 3.918 cm | **3.463 cm** | 7.987 cm | **7.325 cm** |

The largest gain was on fresh fixes, at **1.46 cm** on average. After
two cycles without a fresh fix it was **0.46 cm**. The table omits the
age 3+ row because it had one paired observation.

The model supplied the position for **94.7%** of left-team PlayOn
decisions. Across all PlayOn cycles, including warmup and fallback, the
position actually used averaged **3.204 cm** error on the left versus
**4.315 cm** on the right. The paired table above controls for player and
cycle. The team comparison includes differences in observations and
actions.

## Did the match results improve?

| Match outcome | Left, model | Right, control |
|---|---:|---:|
| Wins | 43 | 42 |
| Draws | 15 | 15 |
| Goals | 317 | 337 |
| Points | 144 | 141 |

The mean left-minus-right difference was **−0.20 goals** per game (95%
interval **−0.69 to +0.29**) and **+0.03 points** per game (95% interval
**−0.51 to +0.57**). These are percentile bootstrap intervals from 10,000
resamples of whole games. The run establishes a live improvement in self
localization. The match results leave room for benefit or harm, and the
model always played on the left, so this batch cannot separate a model
effect on match results from a side effect.

## Reproduce

```sh
cd /home/nader/workspace/denoise
python3 rcssserver-denoise/experiments/exp2/reporting/summarize_live.py \
  logs/exp2-logs --workers 12 --json logs/exp2/live-100-summary.json
```
