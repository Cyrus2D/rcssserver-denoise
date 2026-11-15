# Exp3 player tracking roadmap

Start with the [held-out Exp3 result](reports/07-unum-embedding.md) and
[normal-view replay](reports/08-normal-view.md). The next comparison uses
normal-view Exp1 logs to measure both position error and coverage.

| Idea | Experiment | Record |
|---|---|---|
| 01 | Gap-aware history with elapsed cycles and availability masks | [Step 09](reports/09-gap-aware-history.md). Keep the gap-aware input. The simple motion anchor lagged librcsc. |
| 02 | Forward-quantized likelihood for candidate states | [Step 10](reports/10-quantized-likelihood.md). Keep the quantized update behind a confidence gate. |
| 03 | One-observation fallback with midpoint, empirical bin median and MLP | [Step 11](reports/11-single-observation.md). Keep the midpoint. The learned fallback gave no useful gain. |
| 04 | Reachability-gated recovery of anonymous player identities | [Step 12](reports/12-identity-recovery.md). Keep confident matches. Audit precision and position error separately. |
| 05 | Validation-selected close-range protection | [Step 13](reports/13-close-range-gate.md). Keep the validation-selected distance and age fallback. |
| 06 | Directional, player-type-aware reachable regions | [Step 14](reports/14-directional-reachability.md). The shared-cap directional update helped most on fresh observations under 10 m. Retain Step 13 during gaps. [Step 25](reports/25-player-type-speeds.md) tested notified teammate types. Validation kept the shared cap. |
| 07 | Forward-model `DistChng` and `DirChng` as additional measurements | [Step 15](reports/15-velocity-bins.md). Small one-cycle gain. Defer additional fixed-decay variants. |
| 08 | Learn bounded range and bearing or observer-relative Cartesian corrections | [Step 16](reports/16-polar-readout.md). Polar improved fresh estimates. Cartesian won all-cycle validation. |
| 09 | Carry observer-pose uncertainty into target projection | [Step 17](reports/17-pose-uncertainty.md). The likelihood helped. Logged pose uncertainty added little. |
| 10 | Maintain multiple motion hypotheses | [Step 18A](reports/18-motion-paths.md). Stop/coast/straight mixture lagged Step 13. Defer further hand weighting. |
| 11 | Learn how much to trust the motion prior | [Step 18B](reports/18-motion-paths.md). Learned trust helped ages 6–12. |
| 12 | Fit a short trajectory jointly and use hindsight only for an offline diagnostic | [Step 18C](reports/18-motion-paths.md). Causal fit lagged Step 13. Hindsight improved gap reconstruction. |
| 13 | Encode nearby players as a masked set | [Step 19](reports/19-masked-player-context.md). Small held-out improvement under 360°. |
| 14 | Separate uniform-number embeddings from role and test on other opponents | [Step 20](reports/20-role-and-uniform.md). Identity ablation complete. Cross-opponent test needs different-team games. |
| 15 | Change focus to vary the distance quantization interval | [Step 21](reports/21-focus-geometry.md). Strong fixed-geometry radial signal. Live focus policy needs controlled games. |
| 16 | Fuse time-aligned observations from two teammates | [Step 22](reports/22-teammate-fusion.md) and [Step 24](reports/24-teammate-packet.md). A one-target quantized packet retained a measurable gain. |
| 17 | Expand the number of observers per training game and compare learning curves | [Step 23](reports/23-expanded-observers.md). More sampled windows helped. Extra observers added little at matched budget. |

## 360° quantization follow-up

- [Step 26](reports/26-bounded-bearing.md) selected bounded range and bearing
  by validation mean. It reached 48.16 cm on the 360° test cohort.
- [Step 27](reports/27-one-cycle-forecast.md) improved the forecast before
  the next sighting. Reweighting the next observation raised position error.
- [Step 28](reports/28-mixture-forecast.md) improved displacement likelihood.
  The position-only Step 26 tracker remains the live candidate.

## Follow-up games

The [step summary](reports/step-summary.md) records each immediate baseline
and its paired gain. The offline comparisons reuse test games across
the series. Use new games for a final choice between fresh-observation
updates or for a live team comparison.

- **Communication.** Test the one-target packet policy with the real
  message budget and log which packets arrive. Separate arrivals from
  target selection and distance-based gating.
- **Generalization.** Repeat the uniform-number and role comparison
  against a different opponent.
- **Active focus.** Run a controlled focus policy for a selected target.
  Measure range error and any loss on other targets across consecutive
  observations with actual movement.

Keep the game split fixed. Report the change against its immediate baseline
and the complete tracker against librcsc. Include mean, p90, coverage,
distance and observation age. Keep ground truth out of online features
and association. Select variants and gates on validation before using
test results.

Sources informing these experiments include [CYRUS 2023](https://arxiv.org/abs/2305.19283),
[CYRUS 2024](https://arxiv.org/abs/2406.05623),
[WrightEagle 2006](https://wrighteagle2d.github.io/robocup/2006/2D_TDP_WrightEagle.pdf),
[GRU-D](https://doi.org/10.1038/S41598-018-24271-9), and
[filtering with quantized measurements](https://arxiv.org/abs/2509.07837).
