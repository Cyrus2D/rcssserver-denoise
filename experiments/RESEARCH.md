# Denoise research guide

Start with [Exp1](exp1/README.md) for the standard quantized baseline and
[Exp2](exp2/README.md) for the live self-position tracker. The goal is to
reduce error in self, ball, and player estimates by separating sensor
rounding, gaps in observations, missing player identity or motion fields,
and teammate reports. The original experiment matrix below records how to
isolate those factors.

## What the three repositories add

| Repository | Denoise code and integration |
|---|---|
| rcssserver-denoise | Protocol v19 experiment settings in [serverparam.cpp](../src/serverparam.cpp). [visualsenderplayer.cpp](../src/visualsenderplayer.cpp) selects the distance model and controls distance-based fading for high-quality observations. |
| librcsc-denoise | Reads the server settings and adjusts observation timing and localization. `DenoiseLogger` writes per-agent schema-v2 logs. The self-position estimator hook runs before decision caches are built. |
| helios-base-denoise | `start.sh` configures dataset logging and the Exp2 model modes. Helios loads the weights and owns inference, recurrent state and model logging. Behavior logs include `pre:`, `role:`, `plan:` and `exec:` labels. |

See the server [noise-model notes](../.github/instructions/noise-model.instructions.md)
and [visual protocol notes](../.github/instructions/visual-observation-detail.instructions.md).
In a workspace with the sibling checkouts, see the librcsc [dataset schema](../../librcsc-denoise/.github/instructions/denoise-dataset.instructions.md)
and [observation and localization notes](../../librcsc-denoise/.github/instructions/observation-parsing-localization.instructions.md).

## Server controls

These four boolean settings default to `false`. Exp1 used those defaults,
standard quantized `see`, normal view control, visual fading and teammate
communication.

| Setting | Effect |
|---|---|
| `server::full_view` | Sets the view cone to 360°. With synchronized visual updates, it also sends a `see` every cycle. The client retains its view-width setting. |
| `server::see_no_fade_unum` | Keeps team and uniform number visible in far, in-cone, high-quality player observations. The velocity and orientation fields still fade. |
| `server::see_no_fade_all` | Bypasses those far-object fades, including the relevant motion and orientation fields. Close and low-quality observation paths keep their existing behavior. |
| `server::see_noise_free` | Selects `ExactObservation` ahead of Gaussian see. It bypasses distance quantization. Bearing and motion fields still use their normal rounding. |

To select Gaussian see, set `gaussian_see : on` in the Helios player
configuration. To suppress normal teammate payloads, set
`server::say_msg_size=0`. The hearing-capacity settings control delivery
limits.

## Planned comparisons

Exp1 supplies B0. Exp2 changes self-position estimation while keeping the
standard visual configuration. Treat the other rows as proposed experiments
until a completed run and report are linked from the [index](README.md).

| ID | Change from B0 | Question |
|---|---|---|
| B0 | Standard quantized observations, default view and fade, teammate messages enabled | How large are the baseline errors? |
| N1 | `see_noise_free=true` | How much error comes from distance quantization? |
| G1 | `gaussian_see : on` | How do errors change under Gaussian distance noise? |
| V1 | `full_view=true` | How much does seeing more of the field more often help? |
| F1 | `see_no_fade_unum=true` | How much does persistent player identity help matching and tracking? |
| F2 | `see_no_fade_all=true` | How much do persistent identity, velocity and orientation fields help? |
| S0 | `say_msg_size=0` | How does disabling teammate payloads affect the estimates? |
| U0 | `see_noise_free=true full_view=true see_no_fade_all=true` | What error remains when distance quantization, visual gaps and far-field fades are reduced together? |
| GV | `full_view=true` with Gaussian see | How does Gaussian noise behave with frequent, wide observations? |

Change one factor at a time before running combined configurations.
V1 changes both coverage and observation frequency, so separating those
effects requires independent controls. N1 isolates distance quantization
while leaving other rounded fields available for analysis.

## Run and measure

- For scored runs, enable `server::fullstate_l=true` and `server::fullstate_r=true`.
  Use Helios `--fullstate reference` so the server positions are available for
  scoring while decisions and model inputs still use observations.
- Use a separate port set and game ID per parallel match, and the same ID
  for both teams in that match. Use a separate log directory per
  configuration. A complete player log has schema version 2 and
  `final: true` with `status: "ok"` in `<game_id>/<side>_<unum>/meta.json`.
- Score `PlayOn` with valid fullstate. Give each eligible sample equal
  weight. Group by true distance, visual age, estimate age and visibility.
  Join on cycle and stopped counter, and score next-cycle predictions only
  across consecutive `PlayOn` cycles. Keep train, validation and test
  separated by whole game.
- For self, use position, velocity, looking-direction and raw visual-fix
  errors from `agent_cycle`. For ball and players, separate raw sensor,
  localized visual and fused track errors from `objects`. Check ambiguous
  matches before interpreting raw player error. Compare means and tail
  errors such as p90 on equivalent samples.
- A decoded teammate report can be scored against a fresh visual estimate
  or a prior-cycle local estimate for the same target. That comparison
  describes message quality. A claim about fusion needs before-and-after
  instrumentation or a controlled run such as B0 versus S0.

The original matrix aimed for about 100 games per configuration. Use a short
smoke run to check settings, valid truth and finalized logs before starting
a batch.

## Exp2 handoff

The deployed model replaces only self position during decision-making in
quantized `PlayOn`. It keeps librcsc velocity and orientation. Visual ball
and player localization has already run when the correction reaches the
decision model. Moving the correction earlier would change the object
localization inputs and would be a separate experiment.

The 100-game live run reduced paired self-position error by 27.2%. It
showed little change in aggregate ball and player position errors and no
clear match-result effect. The active team always played on the left.
Keep that side assignment in mind when comparing scores across batches.
See the [live results](exp2/reports/steps/07-live-100-games.md) and
[comparison report](exp2/reports/comparison.html).
