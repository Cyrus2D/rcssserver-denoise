# Exp3 Step 04 — Learned motion inside the region

The region GRU slightly improved on Step 02. Adding the learned motion head
alone increased mean validation error from **53.12 to 53.28 cm**.

The region GRU receives the eight-cycle target history and predicts a radial
correction inside the intersection's selected ray. The motion variant adds
a head that uses the seven preceding observations to predict displacement
from the preceding filtered position. Its estimate supplies a radial hint
to the current-cycle readout. We add a small training loss on the
displacement prior using the current fullstate position.

All rows use the same stride-24 validation endpoints from 20 games (162,099 paired observations overall). Values are relative position mean / p90 in cm.

| Cohort | Step 02 GRU | Region GRU | Region + motion |
|---|---:|---:|---:|
| all | 53.43 / 121.58 | 53.12 / 120.83 | 53.28 / 121.16 |
| 10–20 | 28.05 / 56.31 | 27.94 / 56.61 | 28.24 / 56.49 |
| 30–40 | 66.03 / 136.50 | 65.60 / 135.14 | 65.68 / 135.95 |
| 40–60 | 84.22 / 180.07 | 83.82 / 178.53 | 84.16 / 180.22 |
| 60+ | 112.91 / 251.60 | 112.62 / 248.73 | 111.21 / 251.41 |
| motion_missing | 77.05 / 162.25 | 76.61 / 160.89 | 76.63 / 161.92 |

The explicit motion head offered no standalone gain in this run. The next
step keeps that architecture fixed to test whether context helps the motion
and position estimates.

[Step 03 · Reachable intersection](03-reachable-intersection.md) · [Step 05 · Ball context](05-ball-context.md)
