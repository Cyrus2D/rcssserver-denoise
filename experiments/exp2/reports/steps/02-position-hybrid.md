# Exp2 Step 02 — Position-only hybrid

**Completed November 6, 2026.** This step tested whether the learned
position gain survives when the tracker keeps librcsc’s velocity and
orientation.

## What changed

The hybrid uses the same epoch 16 checkpoint from Step 01. On each
cycle it predicts its own position and uses the GRU’s position gate
when a fresh visual fix is available. It then takes the velocity,
body angle, and neck angle from librcsc’s estimate for that cycle.
The next cycle starts from the hybrid position and those retained
motion components. If a velocity or orientation pair is unavailable,
the tracker uses its own value for that pair.

The evaluation used full contiguous `play_on` sequences from the
10 validation games, with one seed and five warm-up steps. All four
estimators were scored against the same fullstate samples. This step
used the existing dataset and checkpoint.

## Validation result

| Position error | librcsc mean / p90 (cm) | Full GRU mean / p90 (cm) | Hybrid mean / p90 (cm) |
|---|---:|---:|---:|
| Overall | 4.30 / 8.63 | 3.83 / 7.93 | **3.80 / 7.88** |
| Fresh position fix | 4.45 / 9.14 | 3.44 / 6.93 | **3.43 / 6.91** |
| One cycle after a fix | **4.15 / 8.00** | 4.24 / 8.72 | 4.22 / 8.70 |
| Two cycles after a fix | **3.88 / 7.89** | 4.84 / 11.24 | 4.70 / 10.80 |

Across 1,010,339 paired cycles, the hybrid reduced mean position error
by **11.6%** versus librcsc. Its velocity and looking-direction scores
matched librcsc’s exactly, as expected from retaining those estimates.
The fresh-position improvement survived. Position prediction at age 2
still had **0.82 cm** more mean error than librcsc.

## Decision for the next step

Keep the hybrid as the position-fusion baseline. The next test should
target position propagation between fixes using the player physics
known to the agent at that cycle. Compare age 1 and 2 and large turns
against this step. Test individual flag inputs separately for the
fresh-position update.

## Reproduce

From `experiments/exp2` in the training environment:

~~~sh
python -B code/train.py --evaluate /home/nader/workspace/denoise/logs/exp2/runs/gru48/best.pt \
  --split validation --hybrid --run-dir /home/nader/workspace/denoise/logs/exp2/runs/gru48/hybrid-recheck
~~~

The original results and TensorBoard events are under
`/home/nader/workspace/denoise/logs/exp2/runs/gru48/evaluation-validation-hybrid`.
