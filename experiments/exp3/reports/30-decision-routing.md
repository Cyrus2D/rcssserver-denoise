# Exp3 Step 30 — Route corrected positions into player decisions

Step 29 replaced the WorldModel track position. The intercept simulator,
strict pass safety, field reach prediction, self-pass and one dribble
safety path still preferred the stock visual position when the
observation was fresh. Step 30 adds a shared position selector. These
consumers use a fresh applied correction in corrected-routing mode,
and otherwise retain their original position selection.

Corrected routing applies the current sighting's candidate just before
the decision caches and only if the current game mode is PlayOn.
The candidate remains in the observation log if a referee message
changes the mode between sighting and decision. Raw seen position,
velocity and body direction are retained for existing tracking.
Kickability uses the corrected tracked position through the normal
cache update. Safety buffers and model weights are held fixed.

The original apply mode remains the control. Use the same executable
and the same frozen Step 26 weight file on both sides. Append
`--player-position-routing corrected` only to the treatment team's
`start.sh` arguments. Omission is equivalent to
`--player-position-routing legacy`. Both teams still pass
`--player-position-denoise apply --player-position-model "$MODEL"`.
Keep the original Exp3 settings and balance which side receives the
treatment.

## Diagnostics

For a short diagnostic game, pass `--debug --debug_intercept
--debug_pass` on the treatment team. The client debug log contains:

- `EXP3 reach` when a player's reach-cycle estimate changes after
  switching from the stock position. The ball trajectory is held fixed.
  The existing intercept-table summary shows the actual fastest player.
- `EXP3 pass` when the predicted opponent, reach time or rejection
  changes for the same pass candidate. This comparison holds the
  receiver, candidate and safety margins fixed.

Keep these debug options off during the scored games because the
counterfactual calculations add work. The normal dataset retains
`ls_pos_x/y`, `denoise_pos_x/y`, `denoise_applied` and
`trk_pos_x/y` for comparing application and coverage. Each agent's `meta.json` records
`config.player_position_routing` as `legacy` or `corrected`.

## Smoke tests

Both CMake builds passed. Two short quantized 360° games exercised
corrected routing on each side against legacy apply. Debug logs
contained **46,329 changed reach estimates** and **788 pass-rejection
flips** across the two games. These are diagnostic records, including
repeated pass candidates. They establish that corrected positions
reach the decision code. The scored match effect remains open.

In a full-length integration match, corrected routing logged zero
applications on non-PlayOn decisions. Legacy routing logged 42.
The corrected side had no candidate on those non-PlayOn rows, so
this run did not directly exercise a pending correction being dropped.
A final smoke test verified the routing metadata for all 22 players.

[Step 29](29-live-integration.md)
