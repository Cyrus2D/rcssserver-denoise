# Exp3 Step 16 — Bounded polar readout

We trained two 48-unit GRUs with the same quantized anchor, input history,
12 epochs, optimizer, sampling budget and Step 13 guards. The Cartesian
model predicts observer-frame x/y corrections. The polar model predicts a
bounded range correction and a bounded bearing correction around the anchor.
The bounds grow with range and observation age.

| Paired test cohort | Rows | Cartesian mean (cm) | Polar mean (cm) |
|---|---:|---:|---:|
| All | 2,000,986 | 165.60 | 165.43 |
| Fresh after identity recovery | 364,275 | 52.13 | 50.88 |
| Age 1–2 | 505,127 | 84.50 | 84.38 |
| Age 3–12 | 324,999 | 150.54 | 151.06 |

The Cartesian rerun reproduced Step 13. It won the all-cycle validation
comparison at **161.07 vs 161.11 cm**. The polar test gain is exploratory.
The fresh subset also improved on validation, from 52.40 to 51.32 cm.
Prioritize the readout for a fresh-only comparison against Steps 14 and 17
on a new test split.

[Paired metrics](polar-metrics.json)
