# Exp3 Step 12 — Anonymous identity recovery

Allowing anonymous observations to update a known track lowered the
ungated stack's relative mean from **174.52 to
171.04 cm** on the same held-out librcsc
cohort. The stack still needed a fallback to beat librcsc overall.

The matcher reads raw `see` events. It removes slots already taken by
explicit observations, gates candidates by team if visible, projected
reachability and consistency with the current measurement sector, then
rejects assignments without a sufficient margin over competing tracks
or observations. Fullstate does not enter the matcher.

| Check | Held-out result |
|---|---:|
| Accepted anonymous observations | 82,248 |
| Accepted observations with a unique truth-based audit match | 82,231 |
| Correct within that auditable subset | 81,596 (99.23%) |
| Active target-cycle slots with explicit observations only | 1,113,539 |
| Active target-cycle slots after identity recovery | 1,194,428 |

The audit forward-quantizes each true candidate state and scores a match
only when one candidate uniquely agrees with the received measurement.
This measures precision on accepted, auditable events. It does not measure
recall over all anonymous observations.

| Method | Scored rows | Relative mean (cm) | Relative p90 (cm) | Global mean (cm) |
|---|---:|---:|---:|---:|
| librcsc | 2,000,986 | 167.71 | 329.36 | 157.85 |
| Known-only stack | 2,000,986 | 174.52 | 354.12 | 164.90 |
| Stack with identity recovery | 2,000,986 | 171.04 | 345.99 | 161.47 |

[Step 11 · One observation](11-single-observation.md) · [Step 13 · Confidence gate](13-close-range-gate.md)
