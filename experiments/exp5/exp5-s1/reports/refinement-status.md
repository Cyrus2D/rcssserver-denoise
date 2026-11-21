# Exp5-S1 refinement: progress and findings

All means use natural per-target-cycle weights. True distance<50 m is an evaluation label; runtime distance uses unmodified WM geometry.

| Stage | Status |
|---|---|
| Distance<50 m rescoring | Complete |
| GRU128 + readable trees / simple rules | Complete |
| Controlled retraining, residual candidate and seed repeats | 8 models scored |
| Fractional, fixed-cap and age-dependent-cap policies | Complete |
| Full original test-cohort evaluation | Complete |
| New-game confirmation with frozen policies | Complete |

## Full original-cohort results

| Policy | All mean cm | Below50m mean cm | All >25 cm added error % | Below50m >25 cm added error % |
|---|---:|---:|---:|---:|
| Mean focused | 46.73 | 50.61 | 7.65 | 8.09 |
| Correction capped at25cm | 116.27 | 127.90 | 0.00 | 0.00 |
| Correction capped at50cm | 107.44 | 118.41 | 5.52 | 5.77 |
| Harm penalized | 49.43 | 53.46 | 3.53 | 3.76 |
| Mean focused, estimated <50 m | 46.73 | 50.61 | 7.65 | 8.09 |
| Harm penalized, estimated <50 m | 49.43 | 53.46 | 3.53 | 3.76 |
| Validation >25 cm-harm budget1% | 63.51 | 68.52 | 0.77 | 0.83 |
| Validation >25 cm-harm budget3% | 51.52 | 55.73 | 3.00 | 3.22 |

Independent confirmation complete: 10 new games, 14,184,868 samples. No new-game result was used to fit, choose thresholds or select another policy.

The practical objective designated before new-game scoring is near50_risk. Alternatives explicitly trade mean error, harm frequency and a hard cap on correction distance. Fresh games use the same baseline teams, not corrected-agent feedback.

[Open the visual report](report.html#refinement).
