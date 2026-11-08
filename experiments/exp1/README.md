# Exp1 · Baseline

We ran 100 games with standard quantized observations to measure the errors
Helios starts with. The average self-position error was **4.31 cm** in
`PlayOn`. The [full report](reports/report.html) covers self localization,
ball and player tracks, hearing, and one-cycle predictions.

## Steps

1. Run both teams with normal view, visual fading and teammate communication.
   Keep Gaussian see, noise-free see and 360° view disabled.
2. Enable server fullstate for both sides and pass
   `--fullstate reference --denoise-log-dir "$LOG_PATH" --denoise-game-id "$GAME_ID"`
   to each team's `start.sh`. Both teams share the same ID for a match.
3. Score `PlayOn` against fullstate, giving each eligible sample equal weight.
   Match one-cycle predictions to fullstate exactly one cycle later.
   Hearing comparisons describe the recorded messages and local estimates.

## Rebuild the report

From the server checkout with NumPy, pandas, and Matplotlib installed:

```sh
python3 -m experiments.exp1.reporting.report_denoise ../logs/exp1 --workers 24
```

The default output is [reports/](reports/) for the full Exp1 batch. For a
single-game preview or a left-side comparison, use `--output` to choose a
separate directory. Use `--side left` to restrict the observer cohort.
