# Exp3 Step 10 — Quantized likelihood

Using the quantized update lowered the gap-aware GRU's mean from
**116.61 to 104.72 cm** on the
same held-out observations. The plain filter also improved substantially
with the quantized update.

The update evaluates 21 candidate ranges on the observed bearing. Each
candidate is run through the v19 distance quantizer using the estimated
focus. It weights compatible candidates with a Gaussian motion prior whose
spread grows with observation age and range. The implementation samples
the radial interval and uses a fallback if no grid point matches. It
does not yet model the full two-dimensional likelihood or pose covariance.

| Method | Scored rows | Relative mean (cm) | Relative p90 (cm) | Global mean (cm) |
|---|---:|---:|---:|---:|
| librcsc | 1,113,529 | 92.48 | 182.26 | 85.48 |
| Simple blend | 1,113,529 | 167.14 | 382.29 | 161.78 |
| Quantized filter | 1,113,529 | 135.38 | 330.76 | 129.38 |
| Gap GRU with blend anchor | 1,113,529 | 116.61 | 264.23 | 110.19 |
| Gap GRU with quantized anchor | 1,113,529 | 104.72 | 239.58 | 98.16 |

On fresh identified observations, the quantized GRU reached
**44.91 cm** versus
**50.16 cm** for librcsc. After three or more cycles
without an explicitly identified observation, its error was
**174.85 cm** versus
**134.80 cm**. Keep the quantized model as a candidate
and guard against long gaps.

[Step 09 · Gap-aware history](09-gap-aware-history.md) · [Step 11 · One observation](11-single-observation.md)
