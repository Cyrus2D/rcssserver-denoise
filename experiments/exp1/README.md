# Exp1 · Baseline

[All experiments](../README.md) · [Full report](reports/report.html) · [Metrics CSV](reports/metrics.csv)

## Results at a glance

In 100 stock-versus-stock games, self-position error averaged **4.31 cm**
in PlayOn. For targets within 50 m, the world-model error was higher on
tracks last seen 3–5 cycles earlier.

| Target within 50 m | Fresh | Last seen 3–5 cycles ago |
|---|---:|---:|
| Ball | 57.44 cm | 85.74 cm |
| Teammate | 71.75 cm | 115.90 cm |
| Opponent | 78.74 cm | 140.70 cm |

The groups contain different target-cycles. This comparison describes how
errors vary with age in normal play. It motivated the later gap experiment.

## Setup

Normal quantized vision, adaptive view, visual fading and teammate hearing.
Both teams use reference fullstate for measuring current errors and
one-cycle forecasts.

## Steps and reports

1. Run both teams with normal view, visual fading and teammate communication.
   Keep Gaussian see, noise-free see and 360° view disabled.
2. Enable server fullstate for both sides and pass
   `--fullstate reference --denoise-log-dir "$LOG_PATH" --denoise-game-id "$GAME_ID"`
   to each team's `start.sh`. Both teams share the same ID for a match.
3. Score `PlayOn` against fullstate, giving each eligible sample equal weight.
   Match one-cycle predictions to fullstate exactly one cycle later.
   Hearing comparisons describe the recorded messages and local estimates.

## Reproduce

From the server checkout with NumPy, pandas, and Matplotlib installed:

```sh
python3 -m experiments.exp1.reporting.report_denoise ../logs/exp1 --workers 24
```

The default output is [reports/](reports/) for the full Exp1 batch. For a
single-game preview or a left-side comparison, use `--output` to choose a
separate directory. Use `--side left` to restrict the observer cohort.
