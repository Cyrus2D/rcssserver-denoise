# Exp3 Step 14 — Directional reachability

The validation-selected policy reduced all-cycle test error from
**165.60 cm to 165.50 cm** compared with Step 13. Its directional update
runs only when a new visual measurement is available. Step 13 beat
every standalone propagation variant in the cohort 3–12 cycles
since the last explicit observation. Keep Step 13 as the reference for the next stale-motion
experiment.

## What we tested

We replayed the normal-view Exp1 games from Step 13. We selected a
method for each observation age on 10 validation games and evaluated
that policy on the separate 10-game test split. Each comparison uses
the same target-cycle rows with both fullstate and an exact-uniform-number
librcsc track. Each row has equal weight.

The recursive filter advances the previous estimate using the prior cycle's
librcsc track velocity when available and at most six cycles old. It
otherwise uses the replay velocity. On a new visual observation, the
filter scores a grid of candidates inside the reported distance and
bearing cell and averages them using motion likelihood as the weight.
One variant spreads uncertainty equally in all directions. The other
spreads it farther along the velocity than across it, with the same
total variance. On cycles without a measurement, both variants use
their propagated position.

We also tested a one-cycle prediction from librcsc's previous position
and velocity, plus three ways of carrying forward the last fresh
Step 13 correction on top of the current librcsc track. The correction
was decayed over 2 cycles, decayed over 5 cycles, or carried unchanged
while the replay track was active.

The velocity cap uses the fastest type advertised by the server. The
player type in these object logs comes from fullstate, so it was excluded
from the replay. This step tests a soft directional prior with a shared
speed cap. Individual player-type limits and hard reachable-set
intersection remain untested in this comparison.

## Test results

Errors are in centimeters. Relative position error is the Euclidean
distance between the estimate and fullstate, each expressed in its
own observer frame. The p90 is the error at or below which 90% of
the rows fall.

| Method | All-cycle mean | All-cycle p90 | Fresh explicit mean | 3–12 cycles since explicit observation mean |
|---|---:|---:|---:|---:|
| librcsc | 167.71 | 329.36 | 50.16 | 134.80 |
| Step 13 | 165.60 | 327.94 | 44.31 | 133.55 |
| Previous librcsc position + velocity | 168.21 | 328.30 | 44.31 | 139.59 |
| Isotropic motion filter | 180.30 | 383.95 | 44.36 | 177.85 |
| Directional motion filter | 180.34 | 384.11 | 44.24 | 178.11 |
| Carry correction, 2-cycle decay | 166.58 | 329.93 | 44.31 | 134.28 |
| Carry correction, 5-cycle decay | 166.73 | 330.42 | 44.31 | 134.45 |
| Carry full correction | 167.38 | 331.75 | 44.31 | 135.89 |
| **Directional on fresh updates, Step 13 otherwise** | **165.50** | **327.94** | **44.24** | **133.28** |

The table pairs **2,000,986** target-cycles overall, including **282,032**
fresh explicit observations and **422,588** rows that were 3–12 cycles
past the last explicit observation. Validation uses track age after
anonymous association. The table uses age since the last explicit
observation to stay comparable with Step 13. This is why the selected
policy can change the 3–12 cycle average even though it uses Step 13 on
every cycle without a new measurement. Some of those rows have a new
measurement recovered by association.

The selected policy improved mean error in **9 of 10** games. A paired
bootstrap over games gave a 95% interval of **0.03 to 0.16 cm** for the
all-cycle gain over Step 13. Overall global position error changed from
**155.76 to 155.66 cm**. On fresh explicitly identified observations,
mean error improved by **0.07 cm** and p90 increased from **91.92 to
93.11 cm**.

### Fresh measurements by distance

| True distance | Paired observations | librcsc mean (cm) | Step 13 mean (cm) | Directional mean (cm) |
|---|---:|---:|---:|---:|
| Under 10 m | 37,873 | 18.73 | 18.92 | **17.33** |
| 10 to under 25 m | 170,340 | 44.58 | **39.00** | 39.01 |
| 25 to under 40 m | 73,819 | 79.16 | **69.59** | 70.11 |

Under 10 m, the directional update lowered the mean by **1.59 cm**
and p90 from **36.45 to 33.36 cm**. The close-range mean also improved
by **1.54 cm** on validation. This breakdown uses true distance for
evaluation only. A distance gate based on what the agent can estimate
would need its own validation and a fresh test split. There were no
paired fresh explicit observations at 40 m or more.

## What we learned

The directional ellipse lowered the fresh mean by **0.12 cm** relative
to its isotropic control. The two filters both accumulated large
errors when propagating without measurements, reaching about **178 cm**
mean at 3–12 cycles since the last explicit observation. Carrying the
previous correction with the fastest decay was closer to Step 13 at
**134.28 cm**, still above Step 13's **133.55 cm**. The fresh-cycle
gain is small, and the rise in fresh p90 matters for a live rollout.
These results are offline.

The next test is to model the quantized `DistChng` and `DirChng`
readings as measurements of target velocity, accounting for observer
motion. Compare whether that produces a better velocity estimate on the
last fresh cycle and a better position estimate several cycles later.

[Step 13 · Close-range and observation-age gates](13-close-range-gate.md) · [Validation and test metrics](directional-metrics.json)
