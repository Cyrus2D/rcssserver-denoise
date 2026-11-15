# Exp3 Step 05 — Ball context

Adding the ball lowered mean validation error from **53.28 to 52.52 cm**
and lowered p90 from **121.16 to 118.52 cm** relative to the same model
without context.

The input uses the ball's WorldModel position relative to the target, the
relative velocity when both estimates are available, the ball track age,
and availability flags. Vectors are rotated into the observer's looking
frame. Context comes from prior observations. The GRU still predicts a
position on the current observation's selected ray.

All rows use the same stride-24 validation endpoints from 20 games (162,099 paired observations overall). Values are relative position mean / p90 in cm.

| Cohort | Motion only | Motion + ball |
|---|---:|---:|
| all | 53.28 / 121.16 | 52.52 / 118.52 |
| 10–20 | 28.24 / 56.49 | 28.19 / 56.20 |
| 30–40 | 65.68 / 135.95 | 64.06 / 131.99 |
| 40–60 | 84.16 / 180.22 | 83.53 / 178.86 |
| 60+ | 111.21 / 251.41 | 108.47 / 235.51 |
| motion_missing | 76.63 / 161.92 | 75.27 / 158.91 |

[Step 04 · Learned motion](04-learned-motion.md) · [Step 06 · Nearby players](06-nearby-players.md)
