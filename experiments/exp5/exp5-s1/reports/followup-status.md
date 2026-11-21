# Exp5-S1 completed follow-ups

These earlier follow-ups are complete. The later [refinement](report.html#refinement) also completes GRU128 routing, harm-aware training, seed repeats and new-game confirmation. No live code was changed.
The original 20 test games have already been inspected; results below are
exploratory reuse. Every paired row has equal weight.

| Stage | What was tested | Finding |
|---|---|---|
| 1: confidence |27 policies: Gaussian/mixture probability within 25/50 cm,50/80/90/95% thresholds, mixture dominance with two readouts | High confidence reduces harmful replacements but does not identify relative advantage over WM. Gaussian 80%/25 cm gate:125.67 cm overall mean;2.19% made >25 cm worse. |
| 2: training/capacity/data | Longer 64-unit training, width 128, depth 2,40 versus 160 source training games at fixed update budget | Selection winner GRU128:51.46 cm on all 28,180,368 test rows versus 57.22 cm original. More updates also help. Source diversity alone had little effect under this budget. |
| 3: WM/model selection | Eight selector families; independent fit/tune validation games; one-million-row matched candidate comparison | Severity-weighted classifier wins both validation objectives. Full-cohort mean policy:56.02 cm,14.88% worsened; risk policy:59.47 cm,5.28% worsened, versus 32.90% for always using original point model. |
| 4: multiple experts | Choose WM or one of the three frozen original models | Full-cohort mean policy 56.36 cm; risk policy 58.53 cm. Does not dominate the simpler selector on all metrics. |
| Research | Ten verified primary/official sources, eight ranked approaches | Expected relative gain, cost-sensitive deferral and local competence are useful formulations; marginal confidence is not superiority over WM. |

## What remained after this earlier stage

1. Freeze the improved 128-unit position model and fit/tune a selector around it.
   No measured result currently combines these improvements.
2. Repeat promising recipes with more seeds and genuinely new games.
3. Inspect rare severe regressions and near-ball cases; choose any targeted
   sampling from training/validation only, preserving natural evaluation weights.
4. After confirmation, consider a separate shadow/live experiment with coherent
   state integration and timing checks. Offline position gains are not win-rate
   evidence.

[Main visual report](report.html) · [Research](selector-research.md) ·
[Capacity learning curves](scaling-study.md)
