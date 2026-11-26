# Exp3 · Player quantization

[All experiments](../README.md) · [Baseline report](reports/report.html) · [Step summary](reports/step-summary.md) · [Live results](reports/31-live-results.md) · [Exp3–Exp4 archive](../exp3-exp4/README.md)

## Results at a glance

| Measurement | Result |
|---|---|
| Fresh, raw position at 10–20 m | **39.60 cm** in Exp1 and **39.54 cm** in Exp3. The combined sensor setup mainly changed which tracks were available and how fresh they were. |
| Step26, held-out 360° games | Relative position error **72.79 → 48.16 cm** on 20 games. |
| F1 live, all left observers | Relative position error **78.66 → 52.33 cm**. Left goals **816–807** over 200 games. |
| F2 live, corrected decision routing | Relative position error **79.06 → 52.75 cm**, sampled at one left observer per game. Left goals **386–359** and **43–17–40 W–D–L** over 100 games. |

F2 averaged +0.27 goals per game, with a 95% game-bootstrap interval of
**−0.24 to +0.80**. The player-position effect is well measured. The match
advantage remains uncertain.

## Setup

Collect 200 stock games with quantized 360° vision every cycle, persistent
uniform numbers and hearing payloads off. Keep distance-dependent motion/body
field omission. In live tests only the left applies Step26. Its output is
a bounded range/bearing correction using eight cycles and nearby context.

## Steps and reports

1. Compare wide-coverage quantized tracks with the [Exp1 baseline](reports/report.html).
2. Test temporal filters, reachability and contextual GRUs. [Step-by-step gains](reports/step-summary.md)
3. Select the [bounded polar correction](reports/26-bounded-bearing.md). Test
   [mixtures and forecasts](reports/28-mixture-forecast.md) against it.
4. Compare [F1 and F2 live](reports/31-live-results.md). The normal-view study
   found limited availability of the 360° model's required history.
   [Normal-view results](reports/08-normal-view.md)

## Reproduce

[Methods, training commands and report regeneration](METHODS.md) · [Research roadmap](ROADMAP.md)
