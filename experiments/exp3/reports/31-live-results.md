# Exp3 Step 31 — Live position accuracy and match results

Step 26 removed about **one third of the mean relative player-position
error** during live 360° play. Both scored batches used quantized 360°
vision, preserved uniform numbers, disabled hearing, and used fullstate
for reference logging. The observed match margins were small relative
to the variation across games.

| Batch | Actual left vs right | Games | Goals | Left W–D–L | Goal margin per game, 95% game-bootstrap interval |
|---|---|---:|---:|---:|---:|
| Original Exp3 | Stock vs stock | 200 | 795–796 | 81–38–81 | −0.005 |
| F1 | Step 26 legacy apply vs stock | 200 | 816–807 | 85–34–81 | +0.045 [−0.33, +0.43] |
| F2 | Step 26 corrected routing vs stock | 100 | 386–359 | 43–17–40 | +0.27 [−0.24, +0.80] |

F2's right-side launcher omitted the apply flag. Its metadata said
`legacy`, which is also the default routing value when the model is off.
One observer per side per game confirmed no right-side model candidates.
F1 and F2 therefore give separate model-versus-stock comparisons. The
increase of **0.225 goals per game** from F1 to F2 has a 95% unpaired
bootstrap interval of **−0.405 to +0.850**. Both treatment teams played
on the left. The intended corrected-versus-legacy comparison with
both teams applying the same model remains open.

## Position results on matched observations

| Batch | Applied PlayOn rows | Relative mean stock → model | Relative p90 stock → model | Global mean stock → model |
|---|---:|---:|---:|---:|
| F1, all left observers | 212,995,020 | 78.66 → 52.33 cm | 166.8 → 119.8 cm | 80.55 → 54.93 cm |
| F2, one left observer per game | 9,663,045 | 79.06 → 52.75 cm | 167.55 → 120.93 cm | 80.98 → 55.37 cm |

Each row compares the same target-cycle against fullstate. The stock
position comes from the treatment observer's original localization,
so this measures the instantaneous correction during that team's play.

In F1, **32.6%** of individual corrections increased relative error,
**3.73%** increased it by more than 50 cm, and **0.58%** by more than
1 m. Within 5 m, the mean improved from **8.46 to 7.38 cm** and 0.91%
of corrections worsened by more than 10 cm. At 60 m or farther, the
mean improved from **188.04 to 107.86 cm**, while 12.19% worsened by
more than 50 cm. For targets within 2 m of the ball and 10 m of the
observer, mean error improved from **17.72 to 13.85 cm**.

## Why measure velocity, body and ball next?

A diagnostic that held the estimated ball position fixed found that
corrected player positions reduced false kickability alarms from
13,871 to 8,828 and increased missed opportunities from 6,589 to
10,314 over 1.96 million sampled observations. This used physical
range against fullstate and did not reproduce the WorldModel's buffers.
The finding motivates scoring ball and player state together around the
control boundary. Step 30 gets the corrected positions to the reach and
pass planners. Those planners still use stock velocity and body direction.

[Step 29 · Live integration](29-live-integration.md) · [Step 30 · Decision routing](30-decision-routing.md) · [Exp4 · Player and ball state](../../exp4/README.md)
