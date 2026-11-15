# Exp3 Step 23 — More training observers and sampled windows

We added a second uniform number on each side for each of the same 80
training games. That increased the number of observer streams from 160
to 320. The 10 validation and 10 test games retained their original
observers. We ran both observer counts at two sampling budgets with the
same Cartesian GRU, seed, optimizer, 12 epochs and Step 13 guards.

| Training observers | Sampled windows per epoch | Full validation mean (cm) | Full test mean (cm) | Test age 3–12 mean (cm) |
|---:|---:|---:|---:|---:|
| 160, Step 13 | 49,152 | 161.070 | 165.604 | 150.538 |
| 320 | 49,152 | 161.210 | 165.742 | 150.837 |
| 160 | 98,304 | 160.893 | 165.396 | 149.498 |
| 320 | 98,304 | 160.874 | 165.360 | 149.686 |

The 320-observer run with more sampled windows won all-row validation
and reduced mean test error by **0.24 cm** against Step 13. At the larger
budget, doubling observers accounts for **0.036 cm** of the all-row test
difference relative to the matched 160-observer run. At the original
budget, extra observers made the result worse. The longer original run
had the lowest test mean in ages 3–12. The learning curves were collected
with one seed, so defer further observer expansion until there is stronger
evidence that data diversity helps beyond simply taking more optimizer steps.

[Paired metrics](expanded-observers-metrics.json)
