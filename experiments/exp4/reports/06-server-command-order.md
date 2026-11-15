# Exp4 step 06 · Remove command arrival priority

## Result

The optional server mode is implemented, built and installed at
`/usr/local/bin/rcssserver`. Models and physics equations are unchanged.
The next 200 games can test perception improvements without giving the
faster team's player actions first priority.

The previous 60-game audit found that new foul charges blocked 238 of 248
left kicks, versus 32 of 276 right kicks in that selected foul-overlap group.
Those counts describe that group, not all kicks. The treatment took longer
to decide, and ordinary receive-time execution could penalize its later kick.

## What changes

| Before | New optional mode |
|---|---|
| Gameplay handlers execute as UDP messages are processed. | The server collects each participating player's actions until its `done`. |
| A faster response can kick before a later foul or tackle. | After every participant finishes, the server shuffles the player batches once and executes them. |
| Receive polling consumes the shared random generator. | Receive polling does not consume physics randomness; command ordering uses its own generator. |
| A synchronous timeout can continue with missing commands. | An incomplete batch is discarded and the match exits with failure, without saving a scored result. |

The server preserves each player's command sequence and existing action
limits. It randomizes **player priority**, rather than individual commands.
The mode defaults to off and requires synchronous play.

A cycle's order depends on the configured order seed, simulation cycle,
stopped-cycle count and participating player identities. It does not depend
on receive order or number of polling calls. A new automatically selected
seed is recorded for each game when `command_order_seed=-1`.

## Verification

| Check | Result |
|---|---|
| Build server, rebuild/install client parameter mirror and rebuild Helios | Passed |
| 64 seeds, each with left-first and right-first real UDP delivery, a 2 ms delay and 0 versus 43 extra receive polls | Exact equality of noisy ball physics, neck angles and action counters for all 128 cases |
| Both contested-action priorities exercised | 34 seeds kick-first; 30 seeds foul-first |
| Mode off control | Original arrival-order behavior reproduced |
| Per-player command sequence, post-done actions, incomplete batch, overflow and malformed input | Passed |
| Installed executable: incompatible mode and invalid seed | Rejected with failure |
| Installed executable: action without `done` | Action stayed unapplied; timed-out match exited 1 and saved no score |
| Left models / right stock, 1,200 scheduled cycles | 22 finalized agents; 1,348 full-roster rounds including stopped time; left first in 702 |
| Left stock / right models, 600 scheduled cycles | 22 finalized agents; 738 full-roster rounds including stopped time; left first in 375 |
| Installed server, left models / right stock, 300 scheduled cycles | 22 finalized agents; no action-window warnings or batch aborts |
| Mode off, 300 scheduled cycles | 22 finalized agents; no command-order trace |

The final left-apply smoke log has no field-write mismatches, close-range
guard violations, overwritten observed body angles or applied ball forecasts.
The action-window warnings and batch-abort messages were absent from the
successful full-team matches.

These are wiring and scheduling tests. Their short-match scores do not
measure a win-rate benefit. Random scheduling removes this identified
latency-to-priority path; it does not establish that the remaining decision
uncertainty or nearby post-kick errors are solved.

[Saved verification](server-command-order/verification.json) ·
[Regression output](server-command-order/regression.txt)

## Run the next 200 games

Keep the current quantized 360° experiment settings, left-side model flags
and stock right-side launch. Add these **server** arguments in the game runner:

```sh
-server::synch_mode=true \
-server::randomize_player_commands=true \
-server::command_order_seed=-1 \
-server::command_order_log=true
```

The plain `server::name=value` form also works when invoking the server directly.
Use a new player log directory, for example
`/home/nader/workspace/denoise/logs/exp3-f5-logs`, and a separate game result
directory. Keep unique game IDs and the runner's existing
`game_log_fixed=true` and `game_log_fixed_name=$GAME_ID`.
There is no new `start.sh` option.

The server writes `<game_log_dir>/<GAME_ID>.command-order.csv` even when
ordinary game/text logging is off. Columns record cycle, stopped count,
round, resolved seed, execution order and queued handler counts. Counts
exclude `done` and immediate protocol queries. Received-command RCL order is
still receive order; use the CSV for actual execution priority.

For reproducible controlled probes, set nonnegative `command_order_seed`
and `random_seed`. Normal batches can retain automatic seeds. Both seeds
and the scheduling flags are mirrored into v19 player metadata.

## Implementation notes for future work

- Gameplay actions, including neck, view, attention and say, queue only
  during parsing in an open think window. Physics/internal handler calls
  execute normally. Both generated parsing and the Cygwin fallback reach
  the same guarded handlers; no grammar change was needed.
- Queries, initialization, transport options and coach/trainer commands
  retain immediate semantics. The existing coach/trainer done barrier
  remains. Trainer interventions are outside the tested player-order
  invariant.
- Each queue is bounded at 128 handlers. A malformed command, overflow or
  missing barrier response invalidates the experiment instead of committing
  a partial batch. New connections join at the next collection window.
- Actions after `done` or outside a collection window are dropped. Legacy
  UDP messages have no cycle identifier: an arbitrarily delayed duplicate
  arriving inside a later open window cannot be identified perfectly
  without a protocol extension.
- A match that times out must be rerun to count toward 200 completed games.
  The existing synchronous wait limit is retained.
- This is an opt-in research scheduling rule. Compare both teams under
  this same rule; keep standard-server results as a separate condition.

## Reproduce verification

From `/home/nader/workspace/denoise`:

```sh
exp=rcssserver-denoise/experiments/exp4
.venv-exp2/bin/python "$exp/verification/build_order_test.py" rcssserver-denoise/build /tmp/server-command-order-test
RCSS_CONF_DIR=/tmp/server-order-fixture-conf /tmp/server-command-order-test /tmp/server-order-fixture
.venv-exp2/bin/python "$exp/verification/order_protocol.py" /usr/local/bin/rcssserver logs/exp4/server-order-protocol-repeat.json
.venv-exp2/bin/python "$exp/verification/run_smoke.py" random-order-repeat --half-time 60 --randomize-commands --order-seed 4242 --server /usr/local/bin/rcssserver
```

Use a fresh smoke name and free port (`--port`) for a repeat. Regression
fixtures use test-only access to actual production CMake objects; their
controlled player settings do not change the server defaults.
