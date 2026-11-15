# Exp3 Step 11 — One-observation fallback

Validation selected the **midpoint** fallback when the current
observation is the only fresh target observation in the eight-cycle window.

The empirical median learns a range correction per reported distance bin
from training games. The small MLP learns a correction bounded by the
range interval. It receives current range, bearing sine and cosine, bin
width and motion-field availability. Both keep the observed bearing.

| Method | Scored rows | Relative mean (cm) | Relative p90 (cm) | Global mean (cm) |
|---|---:|---:|---:|---:|
| librcsc | 43,116 | 62.88 | 127.65 | 64.50 |
| Bin midpoint | 43,116 | 62.88 | 127.65 | 64.50 |
| Empirical bin median | 43,116 | 63.07 | 128.10 | 64.68 |
| Single-observation MLP | 43,116 | 63.09 | 128.17 | 64.71 |
| Gap-aware quantized GRU | 43,116 | 63.01 | 126.39 | 64.56 |

The midpoint gave the simplest fallback and matched the fresh librcsc
estimate on this cohort to the precision shown. The empirical correction
and MLP brought no meaningful gain in this cold-start setting.

[Step 10 · Quantized likelihood](10-quantized-likelihood.md) · [Step 12 · Identity recovery](12-identity-recovery.md)
