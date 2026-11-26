# Exp3–Exp4 · Completed research summary

**Best-supported result:** recurrent player-position denoising removed about
one third of live mean position error. Player velocity/body estimates also
improved numerically. Ball models improved averages after integration
repairs, but remained unreliable around kicks and quantized position jumps.
**A dependable game-result advantage was not established.**

[All experiments](../README.md) · [Combined report](report.html) · [Markdown](report.md) · [Structured results](results.json)

| Topic | Where to find it |
|---|---|
| What worked, noise reductions and match results | [Combined report](report.md) |
| Full Exp3 research steps, scripts and reports | [Exp3](../exp3/README.md) |
| Full Exp4 models, live integration and audits | [Exp4](../exp4/README.md) |
| Observer 9 ball instability, cycles 784–802 | [Closeout case study](../exp4/reports/09-closeout-ball-stability.html) |

## Code preserved for future work

| Repository | Archive branch | Recorded commit |
|---|---|---|
| librcsc-denoise | denoise-exp3-exp4 | b0894ef |
| helios-base-denoise | denoise-exp3-exp4 | ba8fe96 |
| rcssserver-denoise | denoise | Server changes remain together |

The lib/base working trees were returned to `denoise`; archive branches
matched their local origin-tracking refs when checked. Detailed experiment
directories remain in place for reproducibility. Raw data and weights stay
outside Git under the workspace's `logs/`.

## Next research direction

Study player-position prediction during visual gaps with 60°/120°/180°
views. Start from `denoise`; retrieve only required logging/instrumentation
from the archived branches. First verify truly noise-free visible
measurements: current `see_noise_free` removes distance quantization but
leaves other rounding. Score every eligible gap age, not only uninterrupted
fresh observations. No next-experiment runtime change or dataset run is
part of this archive step.

Checking out a source branch does not replace existing executables.
Rebuild librcsc and Helios before collecting the next experiment.
