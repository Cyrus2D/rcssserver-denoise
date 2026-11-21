# Exp5-S1 live integration

**Ready for a live trial.** The frozen practical policy is implemented in
librcsc and Helios, built and installed. Default operation is off. The default
active counter policy preserves the real observation age.

## What is applied

- GRU128 `wmresidual128_l1`, 32 cycles of 51 causal features and uniform-number
  embedding; boosted-tree `strong_cost` selector; strict score >0.8.
- Output: another player's current x,y. Apply selected predictions on valid
  stale known-player tracks, named-sighting age 1–30, during continuous PlayOn.
  Fresh count-zero positions, ball, self, velocity and body stay stock.
- Only Exp5 unquantized, finite-view, no-hear/no-fade conditions are supported.
  Unsupported settings log a status and fall back to WM. Fullstate is labels
  only. No true-distance gate or true state enters the model.
- Model files/inference/history are in Helios. Librcsc supplies the hook,
  decision overlay, refreshed geometry/interception and paired logging.

The offline independent baseline-game confirmation below 50m was **134.98 →
52.30 cm** mean error; 3.64% added >25cm error and 1.70% added >50cm. Rare severe
regressions remain (up to 11.82m added). Those are offline results, not a live
win-rate forecast. No extra correction cap was added to the selected policy.

## Position count: configurable, without corrupting history

| Option | Position seen by decision code | Effective posCount | Real tracker and visual ages |
|---|---|---|---|
| off | stock | stock | unchanged |
| shadow | stock; prediction logged | stock | unchanged |
| apply + preserve (recommended first) | selected forecast | stock age | unchanged |
| apply + zero | selected forecast | 0 only while applied | unchanged |

Zero is not an actual sighting. It removes some stale-position penalties and
changes age-based action/view/interception gates. Test it in a separate batch;
do not mix its results with preserve. `seenPosCount` is never reset. The next
sensor fusion and model input use the original tracker state in both modes.
Predictions are removed after action/debug output and before future fusion.
Histories reset across stoppages, setplays and cycle discontinuities.

## Run with the existing parallel launcher

Keep the Exp5 server preset (`../server-baseline.conf`): unquantized visual
fields, full_view=false, say_msg_size=0, both no-fade flags true, fullstate for
reference, synchronized games with randomized command ordering. Keep adaptive
view on both teams unless running a separately labelled fixed-view experiment.

```bash
BASE_PATH=/home/nader/workspace/denoise/helios-base-denoise/build/bin
LOG_PATH=/home/nader/workspace/denoise/logs/exp5-live-preserve
MODEL_DIR=/home/nader/workspace/denoise/logs/exp5-s1/live-model

cd "$BASE_PATH"
if [[ "$TEAM" == lt ]]; then
  ./start.sh -p "$PORT" -P "$OLCOACH_PORT" -t lt \
    --fullstate reference --view-policy adaptive \
    --denoise-log-dir "$LOG_PATH" --denoise-game-id "$GAME_ID" \
    --no-denoise-raw --no-denoise-landmarks \
    --player-gap apply --player-gap-model-dir "$MODEL_DIR" \
    --player-gap-pos-count preserve >"$RESULT" 2>&1
elif [[ "$TEAM" == rt ]]; then
  ./start.sh -p "$PORT" -P "$OLCOACH_PORT" -t rt \
    --fullstate reference --view-policy adaptive \
    --denoise-log-dir "$LOG_PATH" --denoise-game-id "$GAME_ID" \
    --no-denoise-raw --no-denoise-landmarks >"$RESULT" 2>&1
fi
```

For the separate zero-count trial change only `preserve` to `zero` and use
`logs/exp5-live-zero`. For shadow change `apply` to `shadow`. The right team
needs no model arguments. Use the runner's existing unique ports/game IDs.
After an initial integration check, run 100 complete games, then preferably
swap treatment sides for another batch to separate side effects.

```bash
cd /home/nader/workspace/denoise
.venv-exp5-s1/bin/python \
  rcssserver-denoise/experiments/exp5/exp5-s1/code/summarize_player_gap_live.py \
  logs/exp5-live-preserve --output logs/exp5-live-preserve/summary-under50.json
```

Each eligible target-cycle gets equal weight. The tool reports paired original,
proposed and actually used errors, added-error tails, age groups and deduplicated
CPU timing. It skips unfinished observers and explicitly counts them. Shadow's
proposed benefit is separate from its unchanged actually-used WM.

## Validation and measured limits

The following are short correctness matches, with two short halves and all
22 logs finalized. They do not compare gameplay effectiveness. Each row uses
its own trajectory, so the preserve and zero error means cannot be compared
as an A/B estimate.

| Short match / side | Paired samples <50m | Applied (all distances) | Stock mean cm | Selected-policy mean cm | Inference median / p95 ms |
|---|---:|---:|---:|---:|---:|
| shadow-verified / l | 39,140 | 0 | 84.43 | 46.24 | 11.90 / 19.55 |
| shadow-verified / r | 38,058 | 0 | 136.12 | 61.73 | 10.86 / 17.73 |
| apply-preserve-verified / l | 36,495 | 23,697 | 178.22 | 58.87 | 11.60 / 21.44 |
| apply-zero-verified / l | 29,595 | 16,661 | 151.42 | 69.33 | 10.03 / 20.54 |

- Native inference vs PyTorch on 256 frozen calibration windows: max output
  position difference 0.00000581m; selector score difference <1e-15; zero routing
  disagreements. The 21-target CPU microbenchmark was about 10.6ms median.
- All 51 live input features were reconstructed from the same raw logged
  observations on both sides and in both age policies. Floating-point tolerances
  are 3e-6 absolute +3e-7 relative; max observed difference was 3.82e-6.
- 1,056 sampled live outputs were also compared with the original PyTorch
  pipeline: max position difference <0.000071m (including four-decimal output
  logging), zero selection disagreements. Boosted-tree scores are sensitive
  to tiny float differences at split thresholds: max score difference 0.0334,
  retained as a diagnostic rather than described as exact score parity.
- Actual `posCount` matched the chosen policy for every applied smoke row.
  Shadow/off never applied; fresh positions were not replaced. Overlay tests
  verify unchanged raw/seen ages, corrected geometry and clearing before the
  next raw update. Unsupported quantized mode never activated in 11 agents.
- CPU live inference medians were about 10–12ms; p95 about 18–22ms, observed
  maximum 34ms on this machine. These are inference times, not complete action
  times or a guarantee under large parallel loads. No CUDA or PyTorch runtime.

Artifacts: `logs/exp5-s1/live-smoke/{shadow-verified,apply-preserve-verified,
apply-zero-verified}` (parity.json and summary.json); frozen model bundle
`logs/exp5-s1/live-model`. Earlier diagnostic smoke directories are not final
correctness evidence. No server runtime changes were needed for integration.

## Rebuild, export and reproduce

Eigen3 headers are required (`sudo apt install libeigen3-dev`). Both CMake and
autotools source lists include the new model files. CMake Release was built
and installed successfully; autotools target lists/dependency flags were
updated but a full autotools build was not run.

```bash
cd /home/nader/workspace/denoise
cmake --build librcsc-denoise/build -j4
cmake --install librcsc-denoise/build
cmake --build helios-base-denoise/build -j4

exp=rcssserver-denoise/experiments/exp5/exp5-s1
.venv-exp5-s1/bin/python "$exp/code/export_player_gap_live.py" \
  logs/exp5-s1/refinement logs/exp5-s1/live-model \
  --dataset logs/exp5-s1/dataset
.venv-exp5-s1/bin/python "$exp/code/run_player_gap_smoke.py" my-shadow-check \
  --left shadow --right shadow --features --port 22600
.venv-exp5-s1/bin/python "$exp/code/verify_player_gap_live.py" \
  logs/exp5-s1/live-smoke/my-shadow-check/player/my-shadow-check \
  --model-dir logs/exp5-s1/live-model
```

Native checks are in `librcsc-denoise/tests/player_prediction_check.cpp` and
`helios-base-denoise/tests/player_gap_model_check.cpp` (compile with the
installed librcsc or model_runtime.cpp/Eigen respectively). The golden checker
requires the exporter's `--dataset` fixture. The Python verifier's model check
uses the original checkpoint/selector; normal live games do not need them.

Bundle ID: `40e7c00f9436dad47d627da9d8280cb42fa7d9c9a85bc5ced75a7f80e663f929`. Logging schema is 4, with raw tracker columns and two
prediction tables. Dataset prepare.py version 2 prefers raw player columns
and leaves ball inputs unchanged; old datasets remain readable. Existing
Exp3/4 model paths/flags are not enabled by this implementation.

## Final duplicate-track regression check

A fourth match (final-duplicate-guard, left apply/zero, right shadow)
passed after ensuring duplicate uniform-number tracks use the same lowest
original age in encoder, application and logger matching. It finalized all
22 observers, applied 10,854 corrections, reconstructed 63,294 feature rows,
and matched another 528 PyTorch outputs with zero routing disagreements.
The maximum position difference remained below 0.000071m. The final binaries
include this guard. Preparation of schema-4 zero-count logs was also tested:
training inputs retain the original player ages, not the effective zeros.
No missing-decision placeholder rows occurred in the three primary checks.
