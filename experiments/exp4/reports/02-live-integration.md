# Exp4 Step 02 — Combined live state test

**Historical F3 design, superseded by [F4 observation isolation and physical inertia](04-live-f4-fix.md).**

Helios loads three exported checkpoints. librcsc owns the eight-cycle
observation history and rebuilds the feature and nearby-player inputs
before applying this cycle's corrections. The model state uses the
team-normalized field frame. Logged coordinates are in the global
server frame.

| Output used by the live team | Model or policy |
|---|---|
| Player position | Exp3 Step 26 with corrected decision routing |
| Player velocity | Exp4 nearby-player context GRU |
| Player body | Stock when body is reported, otherwise the context GRU |
| Current ball position | Exp4 quantization-constrained model |
| Current ball velocity | Exp4 controlled/free-ball heads |
| Next ball position and velocity | Exp4 controlled/free-ball heads |

The ball forecast seeds the first step of ball inertia queries and
other-player interception paths, followed by ordinary decay. The current
WorldModel uses the current estimates. The specialized self-intercept
solver retains its native extrapolation from current state. Player
reach, pass and dribble checks that choose between tracked and seen
velocity prefer an applied fresh velocity correction. Raw seen
positions and velocities remain available for localization and logs.
No pass, dribble or kickability margins were retuned.

The gate requires uninterrupted high-quality quantized 360° PlayOn,
preserved uniform numbers, ordinary motion-field fading and hearing
disabled. Gaps and mode changes reset history. Ball inference and
the player models use the stock snapshot from this update as inputs.
Earlier live corrections can still affect subsequent WorldModel
updates and team behavior.

## Match runner

Keep the original Exp3 server settings. Put the same Step 26 model
with corrected routing on both teams. Use a fresh dataset directory
and reference fullstate on both sides.

```sh
MODEL=/home/nader/workspace/denoise/logs/exp3-context-runs/step26-polar/player-position.weights
EXP4=/home/nader/workspace/denoise/logs/exp4/weights

# Both teams, in addition to the existing log and match options.
--player-position-denoise apply --player-position-model "$MODEL" --player-position-routing corrected

# Treatment team only.
--exp4-state apply --exp4-model-dir "$EXP4"
```

The Exp4 default is off. Use `--exp4-state shadow` on a diagnostic
run to log Exp4 candidates while retaining the Step 26 decisions.
Swap the treatment side halfway through the scored batch.

The `exp4_*` fields in `objects.csv.gz` include the stock comparison,
candidate, next-cycle forecast and application flags. Player rows
carry velocity and body. The ball row has object ID zero.
`meta.json` records `config.exp4_state`. Score the next-cycle forecast
against the next consecutive PlayOn reference row.

## Verification

C++ inference matched Python on 900 identical model inputs sampled
from both team frames. The largest absolute output difference was
below 0.000008 in the corresponding output units. This check isolates
the network from the live feature builder.

In the first 360° smoke game, the apply side logged 73,689 player-state
updates and 3,509 ball-state updates. Body inference was applied on
31,721 player rows. The shadow side logged the same number of
candidates with zero applied. The current estimates matched the
applied WorldModel fields, and both ball forecast vectors were logged.
These checks establish wiring and logging. A scored match comparison
and throughput with the parallel match runner remain open.

[Offline state results](01-state-models.md)
