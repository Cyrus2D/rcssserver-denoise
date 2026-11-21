# Exp5: 200-game finite-view baseline

**Complete: 200 games, 4,400 finalized observers, 21,470,966 PlayOn decisions.**
All 819,660,850 checked raw/fix scalar readings had zero error at the saved precision. All 4,400 hearing logs were empty. Fullstate remained reference-only; both teams used adaptive finite views and no neural correction.

[Open the visual report](report.html) (12 embedded visuals) · [Compact position table](report.md) · [All aggregate data](tables.csv) · [Frozen game list and metrics](metrics.json)

## What we learned

- Visible measurements are exact; errors during visual gaps remain substantial.
- Player position error is about 24 cm at position count 1, rising to 112–116 cm at count 5. Ball error rises from 19 cm to 303 cm.
- Count 0 can include a wrong retained player identity after an anonymous near-field update. These live errors remain in the main WM means.
- A fresh ball position can have an old velocity: one saved example has position count 0, velocity count 2, and 4.902423 m/cycle velocity error because motion fields were absent.
- A see update need not provide a new self-position fix. Successful fresh self fixes were exact; self position count 1 averaged 1.8406 cm error.
- Old player velocity/body values become invalid and are omitted by the existing logger: velocity count >=5 and body count >=2. Missing cells cannot be interpreted as zero error.
- Current-width comparisons are descriptive. Compare matching counters and examine how each width changes exposure to stale information.
- Main means weight every valid observer-target-cycle equally. The 11 self-position-count-3 samples are explicitly marked as a small cohort.

## Steps

| Step | What we did | Result |
|---|---|---|
| 0 | Exact/quantized pilot controls and finite-view cadence checks | Sensor mode and client localization verified; see previous pilot report |
| 1 | Collected both baseline teams for 200 adaptive-view games | All 4,400 logs finalized; no missed-decision placeholders in PlayOn |
| 2 | Audited raw sensors, WM state, counters, width/distance and error tails | Exact raw measurements; gap growth and rare identity/missing-field failures isolated |
| Next | Build causal player-gap prediction baselines/datasets, preserving fresh readings and identity/field masks | No model trained or applied in this step |

## Reproduce

From /home/nader/workspace/denoise, follow the live_audit.py and verify_run.py commands in [the experiment README](../../README.md). Generated caches and the working report are under logs/exp5/reports/baseline-200; caches are excluded from this repository snapshot.

The HTML is standalone for its plots. Keep metrics.json, tables.csv, run-audit.json and the saved case CSV beside it for the download links. [Match outcomes](match-results.txt) are baseline versus baseline, not a test of model benefit.
