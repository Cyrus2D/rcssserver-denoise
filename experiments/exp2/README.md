# Exp2 · Self localization

[All experiments](../README.md) · [Live audit](reports/steps/07-live-100-games.md) · [Full report](reports/report.html) · [Exp1 comparison](reports/comparison.html)

## Results at a glance

On 5.02 million paired PlayOn player-cycles, the self-position GRU reduced
mean error from **4.317 to 3.145 cm (27.2%)** and p90 from **8.657 to 6.260 cm**.
The paired mean fell in all 100 games.

The left team recorded **43 wins, 15 draws and 42 losses**, with goals
**317–337** and points **144–141**. These matches leave the gameplay effect
uncertain.

## Setup

Use the Exp1 quantized logs to train a 48-unit GRU with circular angle
features and up to 11 consecutive cycles. In live tests the model supplies
self position on the left. Standard velocity and orientation are retained.

## Steps and reports

| Step | What we tested |
|---|---|
| [01 · GRU baseline](reports/steps/01-gru48-baseline.md) | Built the dataset from Exp1 using an 80/10/10 split by game. Trained a tracker with up to 11 consecutive cycles. |
| [02 · Position-only hybrid](reports/steps/02-position-hybrid.md) | Used the learned position and kept librcsc's velocity and orientation. |
| [03 · Player physics](reports/steps/03-player-physics.md) | Used the player-type parameters known to each agent during play. |
| [04 · Landmark inputs](reports/steps/04-landmark-inputs.md) | Compared individual flags with landmark summaries. Their test errors were nearly identical. |
| [05 · Architecture and training](reports/steps/05-gate-and-model-comparisons.md) | Compared training budget, width, circular angles, and a position residual. Selected the 48-unit model with sine/cosine angles. |
| [06 · Live integration](reports/steps/06-live-self-localization.md) | Added `shadow` and `apply` modes for quantized `PlayOn`, with resets and warmup at play boundaries. |
| [07 · 100 live games](reports/steps/07-live-100-games.md) | Ran the left team in `apply` and the right team as control. |

The [training guide](code/training.md) covers dataset construction, training,
TensorBoard, and the offline comparisons.

## Reproduce

Use the same match launcher and Helios `start.sh` as in Exp1.

```sh
MODEL=/home/nader/workspace/denoise/logs/exp2/runs/step05-circular/self-position-gru.weights
./start.sh -p "$PORT" -P "$OLCOACH_PORT" -t lt \
  --fullstate reference --denoise-log-dir "$LOG_PATH" --denoise-game-id "$GAME_ID" \
  --self-localization apply --self-localization-model "$MODEL"
```

Use `shadow` to record the model position while playing with librcsc. Use the
original Exp1 arguments for the control team. Give both teams the same game
ID and each parallel match a distinct ID.

### Rebuild the HTML reports

Run from the server checkout with NumPy, pandas and Matplotlib installed.
Both reports use left-team observers. Ball and player metrics include
targets from either team.

```sh
baseline=$(mktemp -d)
python3 -m experiments.exp1.reporting.report_denoise ../logs/exp1 \
  --side left --output "$baseline"
python3 -m experiments.exp1.reporting.report_denoise ../logs/exp2-logs \
  --side left --output experiments/exp2/reports
python3 -m experiments.exp2.reporting.summarize_live ../logs/exp2-logs \
  --json ../logs/exp2/live-100-summary.json
python3 -m experiments.exp2.reporting.compare_denoise_reports \
  "$baseline" experiments/exp2/reports ../logs/exp2/live-100-summary.json
```
