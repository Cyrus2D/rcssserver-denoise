# Exp6 · Gaussian visual noise

[All experiments](../README.md) · [Live 200-game report](reports/live-gru.md) · [Verified metrics and scores](reports/live-200-metrics.json)

## Results at a glance

The capped GRU reduced paired live player position error under 50 m from
**55.67 to 43.47 cm (21.9%)**. The mean fell in all 200 games. **8.1%** of
eligible rows under 50 m gained more than 10 cm of error.

The left team scored **845–793 goals** and went **92–31–77 W–D–L**. Its win
rate excluding draws was **54.44%** (95% Wilson interval **46.9–61.8%**).
The 300-game stock-vs-stock Gaussian batch had **45.25%** for the left.
The runs used different games with the model always on the left, so the
match-strength effect remains uncertain.

## Setup

Use Exp3's 360° synchronized setup with persistent uniform numbers and
standard motion-field fading. Disable teammate payloads. Both teams request
Gaussian see.
The left applies the player-only model in the live test. The model combines
eight visual observations with a Kalman estimate, keeps standard positions
below 5 m observed range, and limits each correction to 25 cm.

## Steps and reports

| Step | What we found |
|---|---|
| [1 · Compare sensors](reports/report.html) | Raw range error was **59.64 cm** under Gaussian versus **75.52 cm** under quantization. The old localizer treated Gaussian range like a quantized reading. |
| [2 · Fix range localization](reports/range-replay/report.html) | Replaying the same observations with reported Gaussian range reduced global visual player error **170.60 → 67.48 cm**. Then verify in [corrected live baseline games](reports/fixed-live/report.html). |
| [3 · Train and compare](reports/learning/report.html) | On 20 new held-out games under 50 m, visual position was **55.61 cm**, capped Kalman **47.28 cm** and capped GRU **43.48 cm**. The ball model was assessed offline. |
| [4 · Use the player GRU live](reports/live-gru.md) | All 4,400 observer logs finalized. All match scores matched the server. The right applied zero corrections. |

## Reproduce

[Build, server presets and collection checks](METHODS.md) · [Training and model export](training/) · [Live launcher and audit](reports/live-gru.md)
