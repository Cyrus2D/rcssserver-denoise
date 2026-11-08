# Exp2 Step 04 — Individual landmark inputs

This step tested whether telling the GRU exactly which flags and goals
were seen improves its position estimate. Both variants used the
position-only hybrid with player-specific motion physics from Step 03.

## Comparison

| Variant | Landmark inputs |
|---|---|
| Summary | The existing counts and summaries of distance, bearing spread and map span |
| Flags | The same summaries, plus fixed slots for 53 flags and two goals |

A visible slot supplies a visibility mask, distance divided by 60 m,
and the sine and cosine of the measured bearing. An unseen slot
supplies zeros to the network. The landmark index uses the raw
observations at the current cycle and librcsc’s landmark IDs. It
contains no fullstate labels. The same frame and per-cycle player
physics apply to both runs.

Both variants used a 48-unit GRU and the same architecture with
19,314 parameters. The summary variant receives a zero landmark
embedding. The flags variant learns a 16-value embedding from the
slots. Both started from the same seed and used the same sampled
windows, eight epochs, 30,000 windows per epoch, batch size 128,
and AdamW at 0.001. The loss scored position after five warm-up
steps in each 11-step window. The checkpoints were selected by
overall validation position error. Epoch 8 was selected for both.

## Results

On validation, overall mean position error was **3.148 cm** with
summaries and **3.147 cm** with flags. Fresh-fix errors were
**3.007 cm** and **3.005 cm**, respectively.

The selected checkpoints were then scored on the same 10 held-out
games and 1,015,290 player-cycles. Step 03 was scored on those same
cycles to show the effect of position-only retraining separately.

| Position error on test | librcsc | Step 03 hybrid | Step 04 summary | Step 04 flags |
|---|---:|---:|---:|---:|
| Overall mean (cm) | 4.322 | 3.230 | 3.184 | 3.183 |
| Overall p90 (cm) | 8.662 | 6.444 | 6.326 | 6.324 |
| Fresh fix mean (cm) | 4.463 | 3.107 | 3.043 | 3.041 |
| One cycle after a fix, mean (cm) | 4.186 | 3.393 | 3.367 | 3.367 |
| Two cycles after a fix, mean (cm) | 3.902 | 3.450 | 3.451 | 3.450 |

The two Step 04 variants were effectively tied. Adding slots
reduced the overall mean by about **0.009 mm** on test. The fresh
fix group had a similarly small difference. The summary variant
reduced overall mean position error by about **26%** versus librcsc
and improved slightly on Step 03.

## Decision

Keep **Step 04 summary** as the working model. This one-seed pilot
found no practical gain from adding slots to the visual trust gate,
while the slot indices required about 4.4 GiB of extra storage. A
different landmark solver or a model that directly corrects the
visual fix would be a separate experiment. The current GRU only
adjusts how much to trust the existing fix.

The next useful check is a small live comparison against librcsc.
All results here replay the recorded actions. Live games can show
whether the more accurate self position changes ball or player
tracking and decisions.

## Run records

- Training and validation — `/home/nader/workspace/denoise/logs/exp2/runs/step04-summary`
  and `/home/nader/workspace/denoise/logs/exp2/runs/step04-flags`
- Held-out metrics and TensorBoard events — `evaluation-test/` inside each run
- Step 03 test comparison — `/home/nader/workspace/denoise/logs/exp2/runs/gru48/evaluation-test-player-physics`
- Build, train and evaluation commands — [training and evaluation commands](../../code/training.md)
