# Exp3 Step 26 — Bounded range and bearing

The Step 19 masked-context GRU predicts range along the ray selected by the
geometry filter. Step 26 adds a small bearing head. It retains Step 19's
range head, radial interval, input features, context, seed and 18-epoch
training budget. The additional head can move the ray by up to half a
degree, with the result constrained to the raw one-degree bearing bin.
The radial interval was computed on the original ray and reused after
rotation. The model therefore approximates a two-dimensional cell.

| Paired cohort | Rows | librcsc mean (cm) | Step 19 mean / p90 (cm) | Step 26 mean / p90 (cm) |
|---|---:|---:|---:|---:|
| 360° validation | 162,099 | 76.56 | 51.05 / 115.34 | 50.78 / 116.29 |
| 360° test | 3,902,451 | 72.79 | 48.48 / 109.08 | 48.16 / 109.83 |
| Normal-view eligible test | 5,252 | 33.14 | 24.97 / 54.62 | 24.50 / 54.58 |

Mean relative error improved in **19 of 20** 360° test games. A paired
game bootstrap gave a **0.09 to 0.48 cm** interval for the 0.32 cm mean
gain. Global mean moved from **50.77 to 50.52 cm**. The largest relative
mean gain was in the 10–20 m cohort, **27.92 to 27.10 cm**. Above 30 m,
the new readout gave little or no mean gain and raised p90. At 60 m or
more, relative p90 rose from **215.91 to 224.25 cm**. Stock librcsc
scored **338.76 cm** p90 in that cohort.

We selected the polar readout by validation mean before testing the
forecast variants. The normal-view comparison reused the Step 08
eligible-only cohort. It covers **0.93%** of fresh explicitly identified
observations in that sample. It measures transfer on uninterrupted
history, with no estimate of the benefit across all normal-view cycles.

[Validation](step26-validation.json) · [360° test](step26-test.json) · [Normal-view replay](step26-normal-view.json)
