# Exp3 Step 18 — Motion paths, learned trust and hindsight

The stale-motion experiment covers three roadmap ideas. It uses previous
Step 13 estimates as anchors and starts only with at least two recent
observations in the same uninterrupted PlayOn segment. Each row has equal
weight.

| Comparison on test | Rows | Immediate baseline mean (cm) | Candidate mean (cm) |
|---|---:|---:|---:|
| A · Stop, coast and straight-motion mixture | 744,338 | Step 13, 107.28 | 132.04 |
| B · Learned gate on ages 6–12 | 106,761 | Step 13, 197.16 | 195.85 |
| B · Validation-selected policy, all rows | 2,000,986 | Step 13, 165.60 | 165.53 |
| C · Causal recent trajectory | 744,338 | Step 13, 107.28 | 129.65 |
| C · Interpolate when the next observation arrives | 597,216 | Step 13, 93.96 | 68.09 |

The mixture keeps a stop path, a decaying-velocity path, and a straight
path from the last two or three sightings. It weights them by observed
speed, velocity agreement and time between sightings. The straight path
alone was slightly stronger. We defer more hand-weighted path variants.

The ridge gate predicts each candidate's error difference relative to
Step 13 from causal age, estimated range, available velocity and motion
agreement. Five validation games fit the gate and five separate games
chose the policy. The selected policy uses the gate only at ages 6–12.
There the test p90 also improved from **394.56 to 391.31 cm**.

The hindsight estimate interpolates between actual sightings on either
side of a gap. It uses the next observation, so its **25.87 cm** reduction
is a retrospective diagnostic for the eligible gaps. It provides a useful
target for assessing how much new information arrives at the next sighting.

[Paired metrics](motion-paths-metrics.json)
