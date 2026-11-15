# Exp3 Step 06 — Nearby players

Adding nearby players on top of the ball lowered mean validation error from
**52.52 to 51.83 cm** and p90 from **118.52 to 116.64 cm**.

Each prior observation contains slots for the two closest estimated players
on the target's team and the two closest on the opposing team. The slots
exclude the target and the observing agent. A slot holds relative position,
relative velocity, track age and availability masks. A player's identity
must be known by uniform number before its track contributes. An audit of the
41,120,268 available player-context positions found no heuristic identity
matches in the Exp3 input cohort.

All rows use the same stride-24 validation endpoints from 20 games (162,099 paired observations overall). Values are relative position mean / p90 in cm.

| Cohort | Motion + ball | Motion + ball + players |
|---|---:|---:|
| all | 52.52 / 118.52 | 51.83 / 116.64 |
| 10–20 | 28.19 / 56.20 | 27.99 / 55.70 |
| 30–40 | 64.06 / 131.99 | 63.27 / 130.84 |
| 40–60 | 83.53 / 178.86 | 82.17 / 176.22 |
| 60+ | 108.47 / 235.51 | 103.82 / 221.66 |
| motion_missing | 75.27 / 158.91 | 74.05 / 155.50 |

[Step 05 · Ball context](05-ball-context.md) · [Step 07 · Uniform-number embedding](07-unum-embedding.md)
