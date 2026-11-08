# Exp2 Step 05 — Gate headroom and model comparisons

This step tested whether the position-only hybrid could improve with
more training, a wider GRU, circular angle inputs, a direct position
correction, or fewer landmark inputs.

## How much can the current gate improve?

On 595,683 fresh fixes in the validation games, we held the trained
rollout’s motion prediction and fresh visual fix fixed. We projected
the true position onto the line segment between those two estimates.
The projection gives the best position that any scalar trust weight
could produce on that cycle.

| Fresh position estimator | Mean error (cm) | p90 (cm) |
|---|---:|---:|
| Logged librcsc | 4.452 | 9.140 |
| Motion prediction | 4.447 | 8.714 |
| Fresh visual fix | 5.631 | 12.633 |
| Step 04 learned weight | 3.007 | 5.857 |
| Best weight with fullstate | 2.160 | 4.602 |

This is an offline diagnostic. Fullstate chooses the weight, and the
result is evaluated on that cycle without feeding it into later
predictions. The gap is room within the current fusion rule on these
inputs. Recovering that gap depends on whether the available sensors
can tell us which estimate is closer to the truth.

## Controlled training comparisons

All Step 05 variants used the same Exp2 split, player-specific
physics, position-only hybrid, 11-step windows and seed 42. We doubled
the sample budget from 30,000 to **60,000 windows per epoch** for
eight epochs. Each variant changed one thing relative to the
60,000-window control. Checkpoints were selected by overall
validation mean position error.

| Validation variant | Selected epoch | Overall mean (cm) | p90 (cm) | Fresh mean (cm) | Age 1 mean (cm) | Age 2 mean (cm) |
|---|---:|---:|---:|---:|---:|---:|
| Step 04 summary, 30,000 windows | 8 | 3.148 | 6.238 | 3.007 | 3.326 | 3.429 |
| 48-unit control, 60,000 windows | 7 | 3.131 | 6.222 | 2.986 | 3.314 | 3.420 |
| 96-unit GRU | 8 | 3.133 | 6.222 | 2.987 | 3.316 | 3.429 |
| Circular angle inputs | 8 | 3.129 | 6.219 | 2.984 | 3.312 | 3.423 |
| Bounded 2D correction on fresh fixes | 8 | 3.142 | 6.215 | 2.996 | 3.325 | 3.441 |
| No landmark summaries | 8 | 3.137 | 6.229 | 2.994 | 3.316 | 3.425 |

The extra training windows accounted for most of the improvement
over Step 04. The circular variant replaces scalar dash direction,
sensed movement direction and fresh visual looking direction with
masked sine/cosine pairs. It kept the 48-unit GRU and added 432
parameters. The larger GRU and position correction gave no mean-error
gain over the control within this one-seed budget. Dropping the
landmark summaries also gave no gain. The residual variant achieved
the lowest overall p90 by a small margin, with higher mean errors.

## Held-out result

The lowest-mean validation variant, circular angles, and its
48-unit control were scored on the same 10 test games and 1,015,290
cycles. Test results were used to check the validation choice.

| Test position error | librcsc | Step 04 summary | Step 05 control | Step 05 circular |
|---|---:|---:|---:|---:|
| Overall mean (cm) | 4.322 | 3.184 | 3.166 | 3.162 |
| Overall p90 (cm) | 8.662 | 6.326 | 6.306 | 6.299 |
| Fresh mean (cm) | 4.463 | 3.043 | 3.020 | 3.015 |
| Age 1 mean (cm) | 4.186 | 3.367 | 3.354 | 3.351 |
| Age 2 mean (cm) | 3.902 | 3.451 | 3.443 | 3.444 |

The circular model is the candidate to carry forward. It reduced
mean position error by **26.8%** versus librcsc, but its advantage
over the extended control was only **0.004 cm** on test. These runs
used a single seed and the existing games, so the differences among
the learned variants are best treated as small. Orientation and
velocity still come from librcsc.

The gate diagnostic suggests a separate line of work if more offline
accuracy is needed. A quantization-aware visual solver could improve
the input position itself. The present experiments mostly change
how the tracker uses an already computed visual fix.

## Run records

- Gate diagnostic — `/home/nader/workspace/denoise/logs/exp2/runs/step05-gate-oracle.json`
- Training runs and TensorBoard events — `/home/nader/workspace/denoise/logs/exp2/runs/step05-*`
- Candidate checkpoint — `/home/nader/workspace/denoise/logs/exp2/runs/step05-circular/best.pt`
- Held-out results — `evaluation-test/metrics.json` in the control and circular run directories
- Reproduction commands — [training and evaluation commands](../../code/training.md)
