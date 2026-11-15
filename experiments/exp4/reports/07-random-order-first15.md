# Exp4 step 07 · First 15 games with random command priority

## Conclusion

**Continue the batch.** The new run in `logs/exp3-f4-logs` is configured
correctly. No server scheduling or sampled model-integration fault was found.
This directory name has been reused after deleting the earlier logs; this
snapshot identifies the new games explicitly.

All 15 selected games have scored results and **330 finalized, status-ok
schema-3 player logs**. Five unfinished/unscored game directories were
excluded when freezing the selection. A later scan found 20 scored games
and no scheduling errors in any current server result file; the tables
below remain frozen to the selected 15.

[Selected game IDs](random-order-first15/selection.json) ·
[Configuration and trace checks](random-order-first15/health.json)

## Configuration and action ordering

| Check | Finding |
|---|---|
| Left team | Corrected player positions; Exp4 player and ball state apply |
| Right team | Original estimates; learned state off |
| Forecast | Left shadow only; no learned forecast applied |
| Fullstate | Reference/scoring only on both sides |
| Visual settings | Quantized, 360°, uniform numbers retained; motion/body fading retained |
| Teammate payloads | Disabled, say_msg_size=0 |
| New server mode | Enabled for every agent in all 15 games |
| Order seeds | 15 distinct seeds; consistent between metadata and each game's trace |
| Full-roster rounds after cycle 0 | 102,640 |
| Left player first | 51,544, or 50.22% |
| Server/player error scan | No batch abort, malformed batch, queue overflow, out-of-window action, model load failure or crash marker |

The CSV files were read from
`AutoTest2D/out/exp3-f4-logs/log.d`. Order and command-count lengths,
unique player identities, consecutive round numbers and seed consistency
passed. Startup rounds with an incomplete roster were excluded from the
left-first statistic.

## Does the previous foul/kick bias remain?

Every selected game's 22 player command logs were checked on paired PlayOn
cycles. These rows select kicks with a new same-cycle foul charge.

| Run | Left kicks blocked | Right kicks blocked |
|---|---:|---:|
| Previous F4 60-game audit | 238 / 248 (96.0%) | 32 / 276 (11.6%) |
| New random-priority first 15 games | 33 / 67 (49.3%) | 37 / 74 (50.0%) |

The strong previous asymmetry is absent in this snapshot. These are
conditional foul-overlap groups, not the fraction of all kicks blocked.

There were 36 unacknowledged kicks out of 8,812 on the left (0.41%) and
38 out of 8,286 on the right (0.46%). New charges account for 33 and 37;
the remaining 3 and 1 commands were outside true kicking range.

For the 141 new-charge events with exactly one logged
opponent foul tackler, the recorded execution priority agrees with whether
the kick was acknowledged in 141 events.
Multi-tackler cases, if any, are counted separately in the saved data.

[Kick audit](random-order-first15/kicks.json) ·
[Foul audit](random-order-first15/fouls.json) ·
[Execution-order cross-check](random-order-first15/order-vs-fouls.json)

## Do the models still reduce error?

The noise scan uses **two rotating left observers and one rotating right
observer per game**: 45 observer logs across 15 games. It includes all eligible
PlayOn samples in those logs and weights each sample equally. This is a
health sample, not the complete 330-observer final experiment report.
The comparisons below pair the left observer's stock and actually applied
estimates against the same truth on the same rows.

| Mean error | Stock → applied |
|---|---|
| Player position relative to observer | 78.91 → 52.96 cm |
| Player position in server coordinates | 80.81 → 55.44 cm |
| Player velocity | 10.14 → 7.48 cm/cycle |
| Player body direction | 25.63 → 17.17 degrees |
| Ball position | 76.78 → 71.72 cm |
| Ball velocity | 32.41 → 24.83 cm/cycle |
| Ball velocity within 5 m | 4.24 → 4.24 cm/cycle |

The player-relative position result includes 2,916,438
tracks; current ball results include 138,878
tracks. All 146,840 sampled left PlayOn decisions
used a see update. That does not imply every object's motion fields are
always reported.

All sampled field-write mismatches, close-range guard violations and
overwritten observed body directions were zero. The right observers
recorded zero learned player-position applications. Learned forecasts
remained unapplied.

The mean improvement does not mean every estimate improves: applied player
relative position is worse than its paired stock estimate on 33.0% of
eligible tracks, and ball velocity on 38.2%. The protected ball state within
5 m and player velocity within 10 m remain stock.

[Sampled noise and wiring measurements](random-order-first15/noise-sample.json)

## Scores and next action

The selected games finished **56–63**
in goals, with left wins/draws/losses **5/2/8**. Fifteen games
cannot establish a match-performance benefit.

Keep the current configuration for the planned 200 completed games. Preserve
both the player logs and server command-order CSVs. No implementation change
is indicated by these checks.
