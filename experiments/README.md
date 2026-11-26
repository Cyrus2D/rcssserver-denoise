# RoboCup 2D denoising experiments

We tested how much Helios can improve what it knows about itself, the ball and
other players. The experiments isolate three causes of error: noisy visual
measurements, time without a sighting and missing fields such as velocity.
We measure accuracy during active play (PlayOn) and separately check whether
the same methods improve match results.

[Open the 15-slide presentation](denoise-presentation.html) · [Download PDF](denoise-presentation.pdf) · [Server changes and research guide](RESEARCH.md)

## Results at a glance

| Experiment | What we tested | Main result | Match result |
|---|---|---|---|
| [Exp1 · Baseline](exp1/README.md) | 100 games with normal quantized vision and hearing | Self position averaged **4.31 cm** error. Below 50 m, opponent error was **78.74 cm** fresh and **140.70 cm** at visual age 3–5. | Stock vs stock baseline |
| [Exp2 · Self localization](exp2/README.md) | Recurrent self-position tracker in normal vision | **4.317 → 3.145 cm** on paired live samples (**−27.2%**). | 100 games, **43–15–42 W–D–L**. No clear advantage. |
| [Exp3 · Player quantization](exp3/README.md) | Quantized 360° every cycle, known identities, no teammate payloads | Step26 reduced live relative player error **79.06 → 52.75 cm** in F2's sampled observers. | F2, 100 games, **386–359 goals**. Advantage uncertain. |
| [Exp4 · Player and ball state](exp4/README.md) | Player velocity/body and ball state on the same quantized setup | F4 reduced live player velocity error **10.43 → 7.57 cm/cycle** and ball position error **75.17 → 69.92 cm**. Ball motion could still be wrong after a kick. | F3, **671–671 goals** in 155 games. Repaired F4, **222–225** in the first 60. |
| [Exp5 · Visual gaps](exp5/README.md) | Exact visible inputs, adaptive 60°/120°/180° vision. Predict stale player positions. | Live half-count policy **142.87 → 53.70 cm** below 50 m. | **600** corrected-cache games across three policies. Half tied on points, **656–696 goals**. |
| [Exp6 · Gaussian noise](exp6/README.md) | Fix Gaussian range localization, then denoise fresh players with a capped GRU | Live player error below 50 m **55.67 → 43.47 cm** (**−21.9%**), with a lower mean in all 200 games. | **845–793 goals**, **92–31–77 W–D–L**. Favorable result, with match-strength uncertainty. |

**How to read the numbers.** Paired error compares the two estimates for the
same observer, object and cycle against reference fullstate. Match results
come from separate simulated games. Each accuracy result has its own sensor
setup, distance filter and target, so the values cannot be ranked across
experiments as if they used the same test. Fullstate provides labels and
evaluation truth. The models use information available to the agent.

## Start here

- [Research guide](RESEARCH.md) documents server controls, logging, measurement
  and reproduction rules across server, librcsc and Helios.
- [Exp3–Exp4 archive](exp3-exp4/README.md) records the quantized-model handoff,
  integration repairs and the ball stability investigation.
- Each experiment README links to its main reports, scripts and reproduction
  instructions. Raw logs, generated training data and weights live under
  the workspace's sibling directory `logs/`. The committed reports and
  metrics preserve results from batches whose raw logs were retired.
