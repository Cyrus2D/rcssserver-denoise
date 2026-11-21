# Half-count policy: launch and verification

**Ready in /home/nader/workspace/denoise/helios-base-denoise/build/bin.**
Use --player-gap-pos-count half on the apply team only. Preserve remains default.

## Exact behavior

Only while a selected forecast is actually applied:

**effective posCount = max(1, floor(original trackedPosCount / 2))**

| Raw count | Half |
|---|---:|
| 1 | 1 |
| 2 | 1 |
| 3 | 1 |
| 4 | 2 |
| 7 | 3 |
| 29 | 14 |

The division always uses raw tracker age, never the previously halved count.
It never marks a forecast as a count-zero observation. Raw/seen/heard ages,
position tracker, velocity and body are unchanged; model inputs still use raw
columns. Clearing the overlay restores raw position/count. Rejected forecasts,
shadow/off mode and fresh measurements retain their original counts.

This is a decision-age experiment. It can change gates/margins/interception and
view behavior; it does not improve the network itself or bound its errors.

## Left team in the existing runner

Keep the same Exp5 server configuration and the runner's unique ports/GAME_ID.
Use a fresh log directory so this trial is not confused with preserve/zero.

```bash
BASE_PATH=/home/nader/workspace/denoise/helios-base-denoise/build/bin
LOG_PATH=/home/nader/workspace/denoise/logs/exp5-live-half
MODEL_DIR=/home/nader/workspace/denoise/logs/exp5-s1/live-model

cd "$BASE_PATH"
./start.sh -p "$PORT" -P "$OLCOACH_PORT" -t lt \
  --fullstate reference --view-policy adaptive \
  --denoise-log-dir "$LOG_PATH" --denoise-game-id "$GAME_ID" \
  --no-denoise-raw --no-denoise-landmarks \
  --player-gap apply --player-gap-model-dir "$MODEL_DIR" \
  --player-gap-pos-count half >"$RESULT" 2>&1
```

Right team stays off: use the same reference/logging/view flags with -t rt,
without player-gap model arguments. Normal and start-debug.sh launchers both
accept half. dlog prints original>effective counts; metadata records
player_gap_pos_count=half. Fullstate remains reference/labels only.

## Completed correctness checks

- CMake Release librcsc rebuilt/installed to /home/nader/local; normal Helios
  build/bin rebuilt against it. The old bool preserve/zero API remains available.
- Native PlayerObject regression covers ages 1–30, odd/even halving, preservation
  of raw/seen state, clearing and legacy preserve/zero behavior.
- Final isolated short game: 22 finalized observers, left apply/half, right
  shadow/half. 5,848 actual corrections, 39,884 routing
  joins checked; shadow proposals never applied.
- 41,580 native feature rows compared with the original encoder;
  maximum feature difference 0.000003815, within the verifier's
  float/rounding tolerance. 528 output samples checked against
  PyTorch, maximum position difference 0.00006841 m;
  zero selection disagreements. Maximum selector-score difference
  0.003144; this is tolerant numerical parity,
  not bit-for-bit equality or calibrated confidence.
- Python audits now understand half, including expected effective count on
  applied rows. Source/generated shell scripts pass syntax checks.

[Machine-readable verification](half-count-checks.json).
Run evidence: logs/exp5-s1/live-smoke/half-count-final-20261118.
**Completed live follow-up:** 200 cache-v2 half games, 656–696 goals,
86/28/86 W/D/L and286–286 points. Eligible stale-player error below50m changed
142.87→53.70 cm. The short checks above verify implementation; the complete
[three-policy audit](live-count-cache-v2-200/report.html) records the gameplay
result and remaining uncertainty. No team-strength advantage is established.

[Completed cache-v2 zero/preserve comparison](live-count-cache-v2-200/report.html).
