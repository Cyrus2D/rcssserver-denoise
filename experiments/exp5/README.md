# Exp5 · Predict players during visual gaps

[All experiments](../README.md) · [Closeout](reports/final/report.html) · [600-game live comparison](exp5-s1/reports/live-count-cache-v2-200/report.html) · [Model study](exp5-s1/reports/report.html)

## Results at a glance

The practical position predictor reduced paired live stale-player error by
about **62% below 50 m**. Each row is a 200-game batch after correcting the
decision caches. The model was active on the left.

| Decision age for corrected positions | Stock → used mean | Added >25 cm | Goals L:R | Left W–D–L |
|---|---:|---:|---:|---:|
| Preserve original count | 142.32 → 53.78 cm | 3.77% | 635–684 | 78–22–100 |
| Set count to zero | 144.00 → 53.99 cm | 3.79% | 678–702 | 78–35–87 |
| Halve count, minimum 1 | 142.87 → 53.70 cm | 3.80% | 656–696 | 86–28–86 |

Half finished **286–286 on points**. Zero had the smallest goal deficit.
No policy established better gameplay. The earlier 400 pre-fix games had
duplicated player caches and are outside this comparison.

## Setup

Collect 200 baseline games with adaptive 60°/120°/180° vision and exact
visible measurements (`see_no_quantization=true`). Preserve visible identity
and motion fields and disable teammate payloads. The finite cone still
creates unseen cycles. Only the logged reference has server truth.

## Steps and reports

1. Verify sensor exactness and measure the cost of gaps. [200-game baseline](reports/02-live-200/report.html)
2. Compare a single position, position with uncertainty and a three-position
   mixture. The single position had the best mean in the initial comparison.
   [Model comparison](exp5-s1/README.md)
3. Refine the point predictor and learn when to keep the standard world
   model. The frozen practical policy scored **134.98 → 52.30 cm** below
   50 m in ten new baseline games. [Refinement study](exp5-s1/reports/report.html#refinement)
4. Verify native inference and cache behavior. Run the three age policies
   and review selected scenes. [Cache and model handoff](reports/final/integration-handoff.html) · [SoccerWindow guide](debug/README.md)

## Reproduce

[Server flags, launch commands and audit scripts](METHODS.md) · [S1 training and integration](exp5-s1/README.md)
