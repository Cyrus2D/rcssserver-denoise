# Exp3 results by step

All errors are mean observer-relative position error in centimeters. Positive
gain means lower error. Each row compares methods on the same observations.
Gains use the unrounded means when available. Different cohorts have
different baseline errors, so gains across cohorts
cannot be added. The roadmap uses separate idea numbers.

| Step | Change | Immediate baseline | Before → after (cm) | Gain (cm) | Cohort |
|---|---|---|---:|---:|---|
| 01 | Diagnose raw quantization | Raw measurement | 78.10 → 13.20 | +64.90 | 360° diagnostic with true range supplied |
| [02](02-temporal-range.md) | Train temporal range GRU | Fresh librcsc | 72.79 → 50.61 | +22.18 | 360° test |
| [03](03-reachable-intersection.md) | Intersect reachable and observed regions | Bounded blend | 71.480 → 71.459 | +0.021 | 360° validation |
| [04a](04-learned-motion.md) | Train GRU inside the region | Step 02 GRU | 53.43 → 53.12 | +0.31 | 360° validation |
| [04b](04-learned-motion.md) | Add displacement head | Step 04a region GRU | 53.12 → 53.28 | −0.16 | 360° validation |
| [05](05-ball-context.md) | Add ball context | Step 04b motion GRU | 53.28 → 52.52 | +0.76 | 360° validation |
| [06](06-nearby-players.md) | Add nearby players | Step 05 ball model | 52.52 → 51.83 | +0.69 | 360° validation |
| [07a](07-unum-embedding.md) | Add uniform-number embedding | Step 06 context model | 51.83 → 51.54 | +0.29 | 360° validation |
| [07b](07-unum-embedding.md) | Remove displacement head | Step 07a model | 51.54 → 51.21 | +0.33 | 360° validation |
| [07 final](07-unum-embedding.md) | Evaluate selected context model | Step 02 GRU | 50.61 → 48.65 | +1.96 | 360° test |
| [08](08-normal-view.md) | Replay Step 07 in standard games | Fresh librcsc | 33.14 → 24.89 | +8.25 | Normal view, only 0.93% of fresh identified rows eligible |
| [09](09-gap-aware-history.md) | Learn with missing observations | Simple blend | 167.14 → 116.61 | +50.53 | Normal-view active tracks, librcsc 92.48 |
| [10](10-quantized-likelihood.md) | Add quantized range likelihood | Step 09 GRU | 116.61 → 104.72 | +11.89 | Same active tracks, librcsc 92.48 |
| [11](11-single-observation.md) | Use midpoint for cold start | Step 10 GRU | 63.01 → 62.88 | +0.13 | Normal-view cold starts only |
| [12](12-identity-recovery.md) | Recover anonymous observations | Known-only stack with Step 11 fallback | 174.52 → 171.04 | +3.48 | Normal-view all paired rows |
| [13](13-close-range-gate.md) | Protect vulnerable tracks with librcsc fallback | Step 12 ungated stack | 171.04 → 165.60 | +5.44 | Same all-row cohort, librcsc 167.71 |
| [14](14-directional-reachability.md) | Apply directional filter on fresh updates | Step 13 guarded stack | 165.60 → 165.50 | +0.10 | Same all-row cohort |
| [15](15-velocity-bins.md) | Invert the server’s quantized motion fields | Step 13 guarded stack | 165.60 → 165.58 | +0.02 | Same all-row cohort |
| [16](16-polar-readout.md) | Change GRU output to range and bearing | Step 13 Cartesian GRU | 52.13 → 50.88 | +1.24 | Normal-view fresh test rows. Cartesian won all-row validation |
| [17a](17-pose-uncertainty.md) | Reweight quantization-compatible position candidates | Step 13 guarded stack | 46.420 → 44.705 | +1.715 | Supported fresh test rows |
| [17b](17-pose-uncertainty.md) | Add logged pose uncertainty to the likelihood | Step 17a fixed covariance | 44.705 → 44.701 | +0.004 | Same supported fresh rows |
| [18a](18-motion-paths.md) | Mix stop, coast and straight paths | Step 13 | 107.28 → 132.04 | −24.76 | Supported stale test rows |
| [18b](18-motion-paths.md) | Learn when to use an alternative motion path | Step 13 | 197.16 → 195.85 | +1.31 | Supported test rows aged 6–12 |
| [18c](18-motion-paths.md) | Fit the previous sightings, or interpolate with the next sighting | Step 13 | 107.28 → 129.65 causal, 93.96 → 68.09 hindsight | −22.36 causal, +25.87 hindsight | Different paired stale cohorts. Hindsight is diagnostic |
| [19](19-masked-player-context.md) | Pool neighbors with a masked shared encoder | Step 07 ordered neighbors | 48.65 → 48.48 | +0.17 | 360° test |
| [20a](20-role-and-uniform.md) | Remove uniform number | Step 19 with uniform number | 51.05 → 51.48 | −0.42 | 360° validation |
| [20b](20-role-and-uniform.md) | Add a position and ball-distance role proxy | Step 20a no identity | 51.48 → 51.66 | −0.19 | Same validation cohort |
| [21](21-focus-geometry.md) | Simulate a second quantized reading at a 20 m focus offset | Repeated zero-focus reading | 48.54 → 11.77 | +36.78 | Fixed-geometry radial diagnostic |
| [22](22-teammate-fusion.md) | Fuse full-precision teammate observations after one cycle | Step 13 | 165.60 → 158.60 | +7.00 | All normal-view paired rows, ideal delivery |
| [23a](23-expanded-observers.md) | Double training observers at original budget | Step 13 | 165.60 → 165.74 | −0.14 | All normal-view paired rows |
| [23b](23-expanded-observers.md) | Double training windows with original observers | Step 13 | 165.60 → 165.40 | +0.21 | Same cohort |
| [23c](23-expanded-observers.md) | Double observers at the larger budget | Step 23b | 165.396 → 165.360 | +0.036 | Same cohort |
| [24](24-teammate-packet.md) | Restrict fusion to one quantized target per cycle | Step 13 | 165.60 → 164.36 | +1.25 | Same cohort, estimated hearing range and assumed delivery |
| [25](25-player-type-speeds.md) | Use the type's running speed in the directional prior | Step 14 shared-cap directional filter | 157.5277 → 157.5277 | +0.0000 | Same paired teammate rows, validation retained shared cap |
| [26](26-bounded-bearing.md) | Learn a bounded bearing correction as well as range | Step 19 range-only GRU | 48.48 → 48.16 | +0.32 | Same 360° test rows. Relative p90 rose 0.75 cm |
| [27](27-one-cycle-forecast.md) | Learn one-cycle displacement and reweight the next observation | Constant velocity before sighting, Step 26 afterward | 67.42 → 61.49 forecast, 48.16 → 48.65 after sighting | +5.93 forecast, −0.49 after sighting | 360° test, global forecast and relative corrected error |
| [28](28-mixture-forecast.md) | Predict three weighted displacement regions | Step 27 single region | 61.49 → 62.08 forecast, 48.65 → 48.64 after sighting | −0.60 forecast, +0.01 after sighting | Same 360° cohorts. Step 26 alone remains best |
| [29](29-live-integration.md) | Integrate Step 26 into librcsc and Helios | Frozen Python Step 26 | — | — | Live smoke and parity checks. Scored games pending |
| [30](30-decision-routing.md) | Feed fresh corrections into interception and safety checks | Step 29 legacy apply | — | — | Side-swapped smoke tests. Scored comparison pending |
| [31](31-live-results.md) | Audit live Step 26 with legacy and corrected routing | Stock position on the same applied rows | 78.66 → 52.33 F1, 79.06 → 52.75 F2 | +26.33 F1, +26.31 F2 | Separate live cohorts. Both faced stock |

Step 01 uses fullstate to measure how much error comes from range and does
not represent a deployable gain. Step 14's strongest gain was on fresh
explicit observations under 10 m, from 18.92 to 17.33 cm. Step 13 remains
the baseline for stale-motion work.

[Research roadmap](../ROADMAP.md)
