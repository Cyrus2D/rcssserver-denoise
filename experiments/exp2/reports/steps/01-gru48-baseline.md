# Exp2 Step 01 — GRU baseline for self localization

**Completed November 6, 2026.** This step tested whether a recurrent
tracker could improve the agent’s self position and looking direction
using the normal-mode Exp1 games.

## Data and model

- **Data.** 100 games, 2,200 agent logs, and 10,634,763 labeled
  `play_on` player-cycles. Split by whole game into 80 training,
  10 validation, and 10 test games. The training split supplied
  735,391 candidate windows.
- **Inputs.** Previous actions, current `sense_body`, librcsc’s raw
  visual position and looking-direction fixes, landmark summaries,
  and the age and availability of each fix. Fullstate supplied the
  target at the same cycle. It was stored separately from the inputs.
- **Model.** A physics-based tracker and a 48-unit GRU with 13,490
  parameters. The GRU learned how much to trust each fresh visual fix.
  Between fixes the tracker propagated its own state with the server’s
  base player parameters. This version used landmark summaries.
- **Training.** PyTorch 2.7.1 on CPU, seed 42, 20 epochs, batch size
  128, and 60,000 sampled 11-step windows per epoch. Each window was
  seeded from the preceding librcsc state. The loss scored the final
  six steps after five warm-up steps. AdamW started at 0.001 with
  weight decay 0.0001 and gradient clipping at 1.
- **Evaluation.** After one seed and five warm-up steps, each tracker
  carried its own state through the remainder of each contiguous
  sequence. The references were logged librcsc and the same motion
  tracker with fixed visual weights. The combined validation position
  and looking-direction score selected **epoch 16**.

## Result on held-out games

The three trackers were compared on the same 1,015,290 player-cycles.

| Metric | librcsc mean / p90 | Fixed tracker mean / p90 | GRU mean / p90 |
|---|---:|---:|---:|
| Self position (cm) | 4.32 / 8.66 | 5.08 / 10.71 | **3.90 / 8.08** |
| Looking direction (°) | **0.535** / 1.034 | 0.764 / 1.125 | 0.733 / **1.014** |
| Velocity (m/cycle) | **0.00294 / 0.00506** | 0.00363 / 0.00591 | 0.00359 / 0.00581 |

Mean position error fell **9.8%**, while mean looking-direction error
rose **37%**. The overall position gain came from cycles with fresh
position fixes.

| Visual age for the relevant estimate | Position mean, librcsc → GRU | Looking-direction mean, librcsc → GRU |
|---|---:|---:|
| Fresh fix | 4.46 → **3.50 cm** | 0.25 → 0.25° |
| 1 cycle after a fix | **4.19** → 4.33 cm | **0.91** → 1.34° |
| 2 cycles after a fix | **3.90** → 4.95 cm | **1.07** → 1.72° |

The validation results show where the large direction errors occur.
At looking-direction ages 1–2 after commanded turns of at least 30°,
mean direction error was **3.60° for librcsc and 6.91° for the GRU
tracker** across 48,463 paired cycles. Across all validation cycles,
p99 direction error was 6.53° for librcsc and 12.28° for the GRU.
The large errors matter even though the overall direction p90 was
slightly lower for the GRU.

## Decision for the next step

Keep the learned position fusion as a candidate and test a hybrid
that uses librcsc’s looking direction and velocity. Evaluate the hybrid
through full rollouts, because direction and velocity affect the next
position estimate.

Then test motion prediction with the player type known to the agent
at each cycle. The current tracker uses base physics, while 2,000 of
the 2,200 agent logs ended with a non-default type. That makes turn
inertia a concrete hypothesis for the gap. Recheck the large-turn
subset to see how much it explains. Test individual flag inputs
separately as a change to fresh visual fusion.

## Run records

- Dataset index — `/home/nader/workspace/denoise/logs/exp2/manifest.json`
- Training run — `/home/nader/workspace/denoise/logs/exp2/runs/gru48`
- Validation — `validation.json` and TensorBoard events in the run directory
- Selected model — `best.pt` in the run directory
- Held-out evaluation — `evaluation-test/metrics.json` and TensorBoard events
