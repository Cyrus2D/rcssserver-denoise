# Exp3 Step 24 — One quantized teammate target per cycle

Step 22 established the value of a second viewpoint with generous delivery.
This stricter replay keeps the same single sender and one-cycle delay.
It selects at most one identified target per send cycle by cycling through
IDs, applies the four-character librcsc OnePlayerMessage's 0.63 m grid and field
limits, and filters by the configured hearing range using the agents'
estimated positions. It assumes delivery if those checks pass. Actual
message competition, packet loss and interference with existing calls are
outside this replay.

| Paired test cohort | Rows | Step 13 mean (cm) | Packet fusion mean (cm) |
|---|---:|---:|---:|
| All | 2,000,986 | 165.60 | 164.36 |
| Packet available | 50,383 | 117.70 | 68.17 |
| Receiver fresh | 13,369 | 48.40 | 40.54 |
| Receiver age 1–2 | 16,713 | 85.66 | 69.42 |
| Receiver age 3 or more | 20,301 | 189.71 | 85.33 |

The all-row gain was **1.25 cm** and p90 improved from **327.94 to
325.61 cm**. Step 22's full-precision replay reached **158.60 cm** on the
same overall cohort. That difference measures the combined restrictions
and lower coverage. The packet version is the candidate for a live
communication test. The next comparison should preserve the existing
team's message budget and measure which packets actually arrive.

[Paired metrics](teammate-packet-metrics.json)
