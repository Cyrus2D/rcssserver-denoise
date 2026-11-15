# Exp4 Step 01 — Player motion, body direction and ball state

The best player policy uses the context GRU for velocity and missing body
direction. It keeps stock body direction when the server sends it.
For the ball, the quantization-constrained model gave the lowest current
position error and the best overall validation score. Separate
controlled and free-ball heads gave the best next-cycle forecast.

## Player state

The table compares the same **3,902,451** test observations. Velocity
is mean Euclidean error in cm/cycle. Body is mean wrapped angle error.

| Method | Velocity mean | Velocity p90 | Body mean | Body p90 |
|---|---:|---:|---:|---:|
| Stock librcsc with missing-field fallbacks | 9.59 | 22.92 | 21.97° | 86.19° |
| History GRU | **7.18** | 14.04 | 19.07° | 63.50° |
| Nearby-player context GRU | 7.24 | 14.05 | 16.51° | 49.55° |
| **Context GRU with validation-selected body gate** | 7.24 | 14.05 | **14.74°** | **49.54°** |

When the raw motion fields were missing, the gated policy improved
velocity from **13.21 to 9.06 cm/cycle**. When raw body direction was
missing, context improved body error from **55.39° to 36.99°**.
The history-only model reached 42.96° in that same missing-body group.
When body direction was reported, stock already scored **0.33°**,
and applying the ungated context model raised it to **3.25°**.
Validation selected the stock observation in that group. Nearby-player
context helped body inference, while velocity was comparable with or
without it.

## Ball state and one-cycle forecast

All methods score the same **184,383** test rows. Position values are
global mean errors in cm. Velocity values are mean errors in cm/cycle.

| Method | Current position | Current velocity | Next position | Next velocity |
|---|---:|---:|---:|---:|
| Stock and physics coasting | 68.75 | 27.63 | 87.02 | 40.59 |
| History GRU | 68.12 | 22.32 | 85.00 | 35.26 |
| Context GRU, identity masked | 68.29 | 22.07 | 84.76 | 35.03 |
| Context GRU with estimated kicker identity | 65.68 | 21.59 | 81.46 | 34.60 |
| **Controlled and free-ball heads** | 65.94 | **21.50** | **80.81** | **33.97** |
| **Quantization-constrained current position** | **63.98** | 21.66 | 81.93 | 34.56 |

The quantization model reduced current position p90 from **147.75 to
136.64 cm**. It had the lowest combined validation error. The matched
identity ablation improved current position from **68.29 to 65.68 cm**
and next position from **84.76 to 81.46 cm**. This measures the usefulness
of the estimated nearest kicker as an input. The estimated control cue
had **64.4% precision** and **75.9% recall** against geometric kicking
range. Neither label identifies who actually performed an action.

| Forecast cohort | Rows | Stock next velocity | Shared context next velocity | Two-head next velocity |
|---|---:|---:|---:|---:|
| Estimated controlled | 64,269 | 57.27 | 49.52 | **48.27** |
| Estimated free | 120,114 | 31.66 | 26.61 | **26.31** |
| Estimated controlled, actual next impulse | 25,713 | 99.44 | 93.13 | **85.48** |
| Estimated free, actual next impulse | 8,028 | 102.51 | **99.54** | 100.83 |

The two heads were especially helpful when the cue indicated control
and the ball's motion then departed from coasting. Across all rows
the model identified **10,648 of 33,741** next-cycle velocity impulses
at the 0.2 m/cycle threshold. That is **31.6% recall** and **60.7% precision**.
The threshold detects a change in velocity beyond decay. It is a proxy
for an impulse and does not establish a kick, tackle or collision label.

## Implications for the next replay

For players, carry the body gate into a paired replay and check
kickability and reach predictions before applying new state estimates
during games. For the ball, compare the quantization model for current
state with the two-head model for one-cycle forecasts. Evaluate the
player-to-ball distance and control decisions with both estimates
on the same observation. This first run uses stock player context
and the original 360° data. Normal-view transfer and live match effects
remain unmeasured. Each architecture used one training seed.

[Player metrics](player-metrics.json) · [Ball metrics](ball-metrics.json) · [Inputs and commands](../README.md)
