# Exp5-S1: three position prediction approaches

[Open the visual report](report.html).

28,180,368 held-out paired target-cycles in 20 games. Every sample has equal weight.

| Model | Stock cm | Full prediction cm | Gain | Improved >1 cm | Worsened >1 cm | Max added error cm |
|---|---:|---:|---:|---:|---:|---:|
| Single position | 129.43 | 57.22 | 55.79% | 66.30% | 27.87% | 1656.18 |
| Position + uncertainty | 129.43 | 65.47 | 49.42% | 65.09% | 32.28% | 1597.00 |
| Three possible positions | 129.43 | 73.68 | 43.07% | 63.67% | 33.22% | 2141.03 |

The original comparison is followed by completed confidence gates (section5.1), controlled training/capacity/data experiments (section9), and observation-only WM/model selectors (section10).

These follow-ups reuse previously inspected test games and are exploratory. See the HTML report for full denominators, harm thresholds, validation protocol and limitations.

[Verified selector research](selector-research.md).

## Latest refinement — New independent10 games

| Policy | All mean cm | Under50m mean cm | Under50m made >25cm worse % |
|---|---:|---:|---:|
| mean | 45.55 | 49.37 | 7.69 |
| cap25 | 110.07 | 121.08 | 0.00 |
| cap50 | 101.53 | 111.90 | 5.40 |
| risk | 48.34 | 52.30 | 3.64 |
| near50_mean | 45.55 | 49.37 | 7.69 |
| near50_risk | 48.34 | 52.30 | 3.64 |
| harm1 | 62.05 | 66.98 | 0.79 |
| harm3 | 50.08 | 54.19 | 3.06 |
