# Exp3 Step 27 — A one-cycle displacement forecast

We froze the validation-selected Step 26 current-position model. The
forecast GRU reads the eight target feature rows through cycle t and predicts
the mean and two axis scales of displacement from t to t+1 in the observer's
cycle-t looking frame. Fullstate displacement supplies the training label.
The forecast starts at Step 26's estimated position at t. The constant
velocity control uses localized velocity when available, otherwise the
existing visual constant-velocity estimate. Its uncertainty is fixed.

At t+1 we score the forecast before incorporating the new sighting. We
also use the forecast density to reweight a grid within the approximate
quantization cell of that sighting. The update applies the change in the
grid centroid to Step 26's current estimate. This keeps the comparison
centered on the frozen tracker, although the two estimates can use
overlapping history. The validation sweep compared strengths 0.25, 0.5
and 1.0 for each forecast. Missing history falls back to Step 26. The
forecast comparison needs nine uninterrupted observations across the two
windows and a valid reference for both cycles.

| Method | Forecast before next sighting, global mean / p90 (cm) | After next sighting, relative mean / p90 (cm) |
|---|---:|---:|
| Step 26 position model | — | 48.16 / 109.83 |
| Constant-velocity prior | 67.42 / 134.50 | 48.89 / 111.25 |
| Learned single-region prior | 61.49 / 128.16 | 48.65 / 110.52 |

The first column of results scores the 3,872,043 forecast-eligible test
rows. The post-sighting column scores all 3,902,451 paired test rows
with the Step 26 fallback. The learned forecast reduced the pre-sighting
mean by **5.93 cm** versus constant velocity. Its 80% marginal
displacement intervals covered **81.4%** and **82.4%** of true movements.
Both priors worsened the position estimate after the new sighting, even
at the weakest positive weight. Validation selected Step 26 alone.

[Step 26](26-bounded-bearing.md) · [Step 28](28-mixture-forecast.md) · [Paired metrics](forecast-360-metrics.json)
