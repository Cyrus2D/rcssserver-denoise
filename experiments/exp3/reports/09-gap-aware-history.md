# Exp3 Step 09 — Gap-aware history

The gap-aware GRU reduced the simple blend's relative mean from
**167.14 to 116.61 cm** on the same
normal-view test endpoints. On that cohort, librcsc achieved
**92.48 cm**.

## What changed

- Parsed actual player observations from raw `see` messages. The baseline
  track comparison uses WorldModel tracks explicitly matched by uniform number.
- Kept one observer per side per game. Split the 100 Exp1 games into
  80 training, 10 validation and 10 test games by whole game.
- Retained an eight-cycle history with an observation mask and elapsed age.
  Missing observations no longer reset the history. The baseline can
  propagate up to 12 cycles after the last identified observation.
- Trained a 48-unit GRU to correct the simple blend using the same 12-epoch
  budget and seed as the quantized variant in Step 10.

| Method | Scored rows | Relative mean (cm) | Relative p90 (cm) | Global mean (cm) |
|---|---:|---:|---:|---:|
| librcsc | 1,113,529 | 92.48 | 182.26 | 85.48 |
| Simple blend | 1,113,529 | 167.14 | 382.29 | 161.78 |
| Gap-aware GRU | 1,113,529 | 116.61 | 264.23 | 110.19 |

The model now produces an estimate for histories with gaps. The simple
motion anchor is still weak, especially for older tracks. Coverage and
error must be measured together. Test measurements use all eligible
endpoints and the same label sources for every method.

[Step 10 · Quantized likelihood](10-quantized-likelihood.md)
