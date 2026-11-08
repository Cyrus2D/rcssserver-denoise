# Step 06 — Live self localization in Helios

The Step 05 circular, 48-unit position model now runs in the Helios player.
Helios loads the exported weights and owns feature extraction, inference,
recurrent state, and logging. The player uses a librcsc pre-decision estimator
hook. In apply mode the returned position is visible to decision caches, action
selection, and command generation. librcsc restores its sensor-based self
position before processing subsequent messages. The model carries its own
position across decisions and takes velocity, body and neck from librcsc.
Reference fullstate stays separate and supplies labels for evaluation.

| Mode | Behavior |
|---|---|
| `off` | Standard Helios. This is the default. |
| `shadow` | Record the librcsc and model positions on the same decision. Helios uses the librcsc position. |
| `apply` | Use the model position when ready. Record the original librcsc position and the candidate. |

The model runs with quantized observations in `PlayOn` on advancing cycles.
Gaussian see and fullstate override are rejected at startup. A noise-free
server bypasses inference at runtime. This checkpoint trained only on
`PlayOn`. The server and client can reposition players at kickoffs and some
restarts, where training did not cover the motion. Outside `PlayOn`, after a
referee mode change, across a cycle gap, or after a known move, the tracker
resets. It seeds from librcsc on a fresh visual position fix, then warms up for
five updates before allowing a correction. Large unexpected jumps, unsupported
two-leg dash commands, invalid reference state, and corrections over 0.5 m also
fall back to librcsc and reset the tracker.

## Run through the existing match launcher

Pass the model arguments directly to the same Helios `start.sh` used to
collect the original games. Keep passing the player and coach ports and the
game ID supplied by the launcher. Run each batch with its own log directory.

```sh
MODEL=/home/nader/workspace/denoise/logs/exp2/runs/step05-circular/self-position-gru.weights

./start.sh -p "$PORT" -P "$OLCOACH_PORT" -t lt \
  --fullstate reference --denoise-log-dir "$LOG_PATH" --denoise-game-id "$GAME_ID" \
  --self-localization shadow --self-localization-model "$MODEL"
```

Change `shadow` to `apply` on the team that should use the model.
Use `--self-localization off` for a baseline team and omit the model file
argument. Give both teams the same game ID, and give each parallel match a
distinct ID. If the launcher can reuse an ID across ports, pass
`"${GAME_ID}-${PORT}"` from both teams. Keep the server's reference fullstate
enabled for the teams you want to score. Add `--no-denoise-raw` to reduce disk
use when protocol replay is unnecessary.

Summarize the batch once its logs are complete.

```sh
cd /home/nader/workspace/denoise
exp=rcssserver-denoise/experiments/exp2
python3 "$exp/reporting/summarize_live.py" "$LOG_PATH"
```

The summary checks that the recorded world-model position matches the
expected source. It reports mean and 90th-percentile error against fullstate
for the position used on all scored `PlayOn` cycles and the paired candidate
vs librcsc errors when the model was ready. In an apply run, the paired
librcsc estimate sees the same action history as the model. The disabled
team supplies the baseline for comparing the positions actually used.

The player log is `self_localization.csv` beside `agent_cycle.csv.gz`. Positions
are in the stadium frame for both teams. The weights can be regenerated with
the existing training environment.

```sh
.venv-exp2/bin/python "$exp/code/export_self_localization.py" \
  logs/exp2/runs/step05-circular/best.pt \
  logs/exp2/runs/step05-circular/self-position-gru.weights
```

## Short validation matches

- Shadow vs shadow recorded 2,738 ready cycles. Mean position error was 4.753 cm
  for librcsc and 3.318 cm for the model on the same cycles.
- Off vs apply recorded 1,857 ready apply cycles. Mean position error on the
  same action history was 4.668 cm for librcsc and 3.378 cm for the model.
  The 90th percentile of the inference step took 52 µs. This timing excludes
  feature extraction and logging.
- Replaying the final shadow logs through the PyTorch checkpoint gave at most
  0.162 mm difference from the live C++ output across 3,068 comparisons on both
  sides. A separate noise-free match bypassed inference on all 2,215 decisions.
- The logs confirmed the five-step warmup after reseeding and that neither
  mode applied the model during stopped play. The short matches verify the
  integration. The full matches measure sustained accuracy.
