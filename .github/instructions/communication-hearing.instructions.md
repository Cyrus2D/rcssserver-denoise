---
applyTo: "src/{audio,player,serializerplayerstd}*"
---

# Communication and Hearing (say / hear / attentionto)

## TL;DR

How a `(say ...)` from one agent becomes a `(hear ...)` for others. Hearing carries **direction (integer degrees, truncated) but never distance**, and messages are never noisy.
- `say` longer than `say_msg_size` (10) is silently dropped; hearing range is `audio_cut_dist` (50).
- Capacity `hear_max/inc/decay` = 1/1/1: each listener gets at most one message per group (teammates, opponents) per sim step; extras are dropped silently.
- Delivery happens on the sense_body tick (100 ms), not immediately.
- Formats: `(hear T DIR our N "msg")`, `(hear T DIR opp "msg")`, `(hear T self "msg")`, `(hear T referee msg)`.
- **Open the full file when:** you parse or filter heard messages, estimate message loss, or touch `attentionto`.

## Say command

- `Player::say` `player.cpp:1704-1713`: message longer than `say_msg_size` (default 10, `serverparam.cpp`) is dropped with no reply. Allowed chars `-0-9a-zA-Z ().+*/?<>_` (`player.cpp:629,636`).
- `audio_cut_dist` 50 (`serverparam.cpp:196`): listeners farther than this hear nothing. Distance is not sent.

## Capacity

- `hear_max/inc/decay` = 1/1/1 (`serverparam.cpp:282-284`). `Player::updateCapacity` runs each sim step (`stadium.cpp:873`, `player.cpp:2705-2717`); `canHearFullFrom` `player.cpp:2588`; `decrementHearCapacity` `player.cpp:2575`.
- Capacity is tracked separately for teammates and opponents. With defaults: at most 1 teammate message and 1 opponent message per step.
- Ear flags default complete=true, partial=false (`audio.h:476-477`): over-capacity messages are dropped silently. Short forms `(hear T our N)` / `(hear T opp)` (content hidden) appear only if the client enabled partial ear (`audio.cpp:564-615`).

## Delivery and selection

- Audio is cached and delivered in `newCycle`, called from `doSendSenseBody` (`stadium.cpp:2286-2291`): arrives on the 100 ms tick.
- Listener order is shuffled in `Stadium::sendPlayerAudio` (`stadium.cpp:2084-2095`).
- Unfocused: one message picked at random among candidates (`audio.cpp:617-639`).
- Focused (`attentionto`): messages from the focused player come first (`audio.cpp:641-701`). Note a use-after-erase risk at `audio.cpp:699-700`.
- `attentionto` handling `player.cpp:1977-2056`.

## Message format

- Serializer `serializerplayerstdv8.cpp:112-146`: `(hear T DIR our UNUM "msg")`, `(hear T DIR opp "msg")`. DIR = `Rad2IDeg` (truncating) of the sound direction relative to the body, **not** the neck (`audio.cpp:575`). v7 rounds (`audio.cpp:370`).
- Self: `(hear T self "msg")` (`serializerplayerstdv7.cpp:66-72`).
- Referee: sender name `REFEREE_NAME` (e.g. `referee`); online coach: `online_coach_left/right`; these are not subject to distance.
- Registration of listeners `audio.cpp:970-988`; v8 predicate `audio.cpp:411-417`.

## Denoising notes

- Heard direction error: truncation only, so true direction is in [DIR, DIR+1) deg (for positive angles; floor toward zero for negative) relative to body.
- No distance info: infer sender from `our N` plus the latest visual estimate.
- Loss model: treat each cycle as a lottery of 1 per group; messages beyond that are lost.

## Manual discrepancies

- Manual says hear is "immediate": code delivers on the sense_body tick.
- Manual says one message per team every second cycle: with default decay 1 it is one per cycle per group.

## See Also

- [observation-sensors](observation-sensors.instructions.md)
- [player-protocol](player-protocol.instructions.md)
- [config-params](config-params.instructions.md)
- [visual-observation-detail](visual-observation-detail.instructions.md)
