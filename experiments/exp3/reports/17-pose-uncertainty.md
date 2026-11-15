# Exp3 Step 17 — Observer-pose uncertainty

We used a fresh-observation likelihood over distance and bearing candidates.
It centers the prior at the previous Step 13 position advanced using the
previous librcsc velocity. The control uses a fixed range-dependent
covariance. Two variants add the logged observer position and heading error
estimates to the covariance at half or full scale. The heading term uses
a first-order rotation of each candidate. We treated the error fields as
uncertainty scales and capped them at 3 m per axis and 15°. Their calibration
as standard deviations was not established.

| Paired test cohort | Rows | Step 13 mean (cm) | Fixed covariance | Validation-selected covariance |
|---|---:|---:|---:|---:|
| All | 2,000,986 | 165.604 | 165.363 | 165.363 |
| Supported fresh | 280,531 | 46.420 | 44.705 | 44.701 |
| Supported fresh under 10 m reported range | 32,979 | 15.871 | 14.382 | 14.382 |
| Supported fresh 10–25 m | 165,953 | 40.100 | 39.198 | 39.196 |
| Supported fresh 25 m and up | 81,599 | 71.621 | 68.160 | 68.152 |

The fixed covariance won under 10 m on validation. The full pose terms won
the two longer bins. The pose terms account for **0.004 cm** of the mean gain
on supported fresh rows. The underlying likelihood accounts for nearly all
of it. On supported fresh rows, p90 increased from **97.23 to 99.40 cm**.
Defer more covariance tuning until a calibrated pose model is available.

[Paired metrics](pose-uncertainty-metrics.json)
