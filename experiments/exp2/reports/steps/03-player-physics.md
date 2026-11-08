# Exp2 Step 03 — Player-specific motion physics

**Completed November 6, 2026.** This step tested whether the hybrid’s
position drift and the full GRU’s large-turn errors came from using
the same motion parameters for every player.

## What changed

The index builder reads the agent’s received `change_player_type`
messages in packet order. For the transition from cycle `t−1` to
`t`, it uses the last own type known before the decision send at
`t−1`. This lets a substitution affect subsequent transitions
without applying its type to earlier cycles. The index has a separate
file per validation game, aligned to the existing Exp2 shards.

The index covered 1,052,293 transitions. Of those, 956,631 used a
non-default type and 59 agents had a midgame type change. As an
independent audit after building the index, all indexed types matched
the preceding cycle’s fullstate player type. The index itself was
built from received messages and the player type catalogue.

The comparison reused the epoch 16 checkpoint. The first variant
changed **turn inertia only**. The second also changed dash power
rate, speed limit and decay. We tested full player physics in both
the standalone GRU and the position-only hybrid. All variants were
scored on the same 10 validation games and 1,010,339 paired cycles.

## Position results

| Variant | Overall mean / p90 (cm) | Fresh mean (cm) | Age 1 mean (cm) | Age 2 mean / p90 (cm) |
|---|---:|---:|---:|---:|
| librcsc | 4.30 / 8.63 | 4.45 | 4.15 | 3.88 / 7.89 |
| Step 02 hybrid | 3.80 / 7.88 | 3.43 | 4.22 | 4.70 / 10.80 |
| GRU with player-specific inertia only | 3.81 / 7.89 | 3.43 | 4.23 | 4.70 / 10.82 |
| GRU with full player physics | 3.20 / 6.38 | 3.08 | 3.36 | 3.43 / 7.29 |
| Hybrid with full player physics | **3.20 / 6.37** | **3.08** | **3.36** | **3.43 / 7.28** |

Full player physics closed the age-2 gap. The hybrid’s age-2 mean
fell from **4.70 cm to 3.43 cm**, below librcsc’s **3.88 cm**. The
fresh-position gain also increased.

## Looking direction after turns

On 48,463 validation cycles at looking-direction ages 1–2 after a
commanded turn of at least 30°, mean error was **6.91°** for the
original GRU, **3.59°** for the inertia-only GRU, and **3.60°** for
librcsc. Inertia alone recovered the large-turn gap on this subset.
Across all cycles, the full-physics GRU averaged 0.534°, and the
hybrid retained librcsc’s 0.544° along with its velocity estimate.

## Decision for the next step

Use the **hybrid with full player physics** as the baseline for the
individual-flag experiment. It improves position on fresh fixes and
at ages 1 and 2, while keeping librcsc’s orientation and velocity.
Compare fixed landmark slots against the current landmark summaries
with the motion model held constant. These Step 03 results are from
offline validation on recorded actions.

## Reproduce

From `experiments/exp2` in the training environment:

~~~sh
python -B code/build_player_physics.py --output /home/nader/workspace/denoise/logs/exp2/player-physics-recheck
python -B code/train.py --evaluate /home/nader/workspace/denoise/logs/exp2/runs/gru48/best.pt \
  --split validation \
  --player-physics /home/nader/workspace/denoise/logs/exp2/player-physics-recheck \
  --run-dir /home/nader/workspace/denoise/logs/exp2/runs/gru48/physics-recheck
~~~

The original index is in `/home/nader/workspace/denoise/logs/exp2/player-physics-validation`.
The metrics and TensorBoard events are in
`/home/nader/workspace/denoise/logs/exp2/runs/gru48/evaluation-validation-player-physics`.
