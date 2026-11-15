# Exp3 Step 15 — Quantized velocity readings

We tested whether motion fields from the last explicit observation can
improve player position during the next observation gap. The reference
was the guarded Step 13 stack. We evaluated the same 10 validation and
10 test games, with equal weight per target-cycle.

The server reports radial velocity divided by true range and rounded
to 0.02, then multiplies by reported range. It also rounds angular rate
to 0.1 degrees per cycle in the default configuration. We inverted the
two bins over the estimated range interval and added the observer's
estimated velocity. We used the last fresh Step 13 position as the
anchor, a shared public speed limit and an assumed 0.4 coasting decay.
A valid previous-cycle librcsc velocity was required for every tested
variant so the comparison uses identical support. Anonymous observations
can end a gap. They did not seed a velocity in this experiment.

| Cycles since update | Eligible test rows | Step 13 mean (cm) | Coasting on previous librcsc velocity | Coasting on bin midpoint | Coasting on clipped prior | 50% blend with Step 13 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 199,716 | 64.83 | 70.02 | 67.83 | 68.38 | **64.63** |
| 2 | 94,730 | **78.34** | 100.21 | 97.01 | 97.92 | 82.40 |
| 3–5 | 91,744 | **104.82** | 163.96 | 160.39 | 161.50 | 121.54 |
| 6–12 | 51,323 | **206.96** | 365.11 | 361.64 | 363.00 | 267.17 |

Validation selected the 50% blend for age one and Step 13 for all later
ages. Across all 2,000,986 paired test rows, mean error moved from
**165.60 to 165.58 cm** and p90 stayed at **327.94 cm**. On supported
age-one rows, p90 fell from **127.05 to 125.53 cm**. The midpoint beat
previous librcsc velocity under the same coasting assumption, but the
coasting model still trailed the tracker. Defer further fixed-decay
tuning. If we revisit the motion fields, use them as measurements in a
model that can represent changes in player acceleration.

[Step 14](14-directional-reachability.md) · [Paired metrics](velocity-bins-metrics.json)
