---
applyTo: "src/{visualsender,bodysender,audio,fullstatesender,initsender,serializer,sender,stdtimer}*"
---

# Observation Sensors (See, Sense Body, Hear, Full State, Init)

## TL;DR

Clients receive five observation streams: visual (`see`), body (`sense_body`), audio (`hear`), full state (`fullstate`), and init (`init` plus parameter dumps). Timing comes from `StandardTimer`/`SyncTimer`; byte format from the per-version serializers. This file is the **hub**: cadence, sense_body, fullstate, init. Object-level `see` detail and communication have their own files.
- **Open the full file when:** you change what a player perceives, when messages are sent, sense_body fields, or fullstate/init content.
- Visual cadence: async every 37.5 ms tick × interval (75/150/300 ms); synch once per sim step (every 1/2/3 steps by width); v18+ forced synch. Details in [visual-observation-detail](visual-observation-detail.instructions.md).
- Sense body: every `sense_body_step` (100 ms). Only **speed magnitude** is quantized (0.01); everything else is exact (ints truncated).
- Fullstate: noise-free (0.001 steps), only for sides with `fullstate_l/r = true`.
- Hear: delivered on the sense-body tick, capacity-limited; see [communication-hearing](communication-hearing.instructions.md).
- Noise/quantization formulas: [noise-model](noise-model.instructions.md) (hub), [noise-quantize-model](noise-quantize-model.instructions.md), [noise-gaussian-model](noise-gaussian-model.instructions.md).

## Overview

Each `Player` owns observers (`M_observer` visual, `M_body_observer`, `M_init_observer`, `M_fullstate_observer`) and an audio sender. Senders are created per protocol version in `Player::setSenders()` (`src/player.cpp` L2439-2524). The timer calls `Stadium::doSendSenseBody` (`src/stadium.cpp` L2253-2298), `doSendVisuals` (L2301-2319) and `doSendSynchVisuals` (L2322-2340). Each shuffles `M_remote_players` with the RNG first (random send order), and sends only to `isEnabled() && connected()` players.

## Visual cadence (current behavior, server v19)

| Mode | Trigger | Interval (`M_visual_send_interval`) | Resulting period |
|---|---|---|---|
| Async (v<18 default) | `StandardTimer` tick every `send_step*0.25` = 37.5 ms (`stdtimer.cpp` L131-139) | narrow 2 / normal 4 / wide 8 ticks (`player.cpp` L1795-1826) | 75 / 150 / 300 ms |
| Synch (`(synch_see)`; **forced for v>=18**) | once per sim step at `simStep*(c-1)+synch_see_offset` (`stdtimer.cpp` L141-150; offset default 0, `serverparam.cpp` L345) | narrow 1 / normal 2 / wide 3 steps (`player.cpp` L2358-2373) | 100 / 200 / 300 ms |

- Throttle is `VisualSenderPlayerV1::sendVisual()` (`visualsenderplayer.cpp` L364-375): `++sendcnt >= interval` then reset. `Player::sendVisual()`/`sendSynchVisual()` are mutually exclusive on `M_synch_see` (`player.cpp` L2414-2429).
- `Player::initObservationMode()` (L376-387) calls `synch_see()` for `version() >= 18`, called from `Stadium::initPlayer` (`stadium.cpp` ~L631). Synch is irreversible; `view_quality low` is rejected in synch (`player.cpp` L1785-1789).
- Defaults at player construction: async, interval 4, high quality, NORMAL width, visible angle = `visible_angle` (90 deg) (`player.cpp` L184-189).
- `view_quality low` (async only): `M_high_quality=false`, interval halved (`player.cpp` L1846-1852), `see` contains **directions only, no distances** (see visual-observation-detail). Still reachable for v<18 clients; monitors report it (`dispsender.cpp`, `logger.cpp`).

## Sense body (`sense_body`)

- Base `BodySenderPlayer` (`src/bodysender.h` L71); versions V1 L174, V5, V6, V7, V8, V12, V13, V14, V18; registration `bodysender.cpp` L423-441. Cadence `sense_body_step` (100 ms, `serverparam.cpp` L126), independent of the sim step.
- Field order/versions (`bodysender.cpp`): V1 L118-136 `view_mode`, `stamina`, `speed`, counts; V5 `head_angle` (L185-191); V6 speed direction (L216-224); V7 `catch/move/change_view` counts (`serializerplayerstdv7.cpp` L174-181); V8 `arm`, `focus`, `tackle` (L266-298); V12 `collision` (L323-331); V13 stamina capacity (`serializerplayerstdv13.cpp` L141-150); V14 `foul` (L379-384); V18 `change_focus` count (`serializerplayerstdv18.cpp` L44-50) and `focus_point <dist> <dir deg>` (L96-104, `bodysender.cpp` L408-412).
- Noise/precision (all exact except one):
  - `speed` magnitude = `Quantize(|vel|, 0.01)` (L139-144, L218) -> client sees speed on a 0.01 grid (max error 0.005).
  - speed direction = `Rad2IDeg(vel.th - body - neck)`: **integer, truncated toward zero** (L219-222; max error <1 deg). `head_angle` = `Rad2IDeg(neckCommitted)` truncated (L188).
  - `stamina`, `effort`, `capacity` are raw doubles (stream default 6 significant digits), `focus_point` dist/dir raw (dir in degrees, double).
  - `view_mode` string: `high|low`, `narrow|normal|wide` (L120-128).
- Velocity is the **true** velocity (post-noise physics), so `speed` already contains actuator noise (see noise-model).
- Manual (`sensor-models-body-sensor-model.rst`) lists the same fields but has TODO for arm/focus/tackle/collision/foul; speed quantization is not documented there (code-only fact).

## Full state (`fullstate`)

- `FullStateSenderPlayer` (`src/fullstatesender.h` L74); versions V1, V5, V8, V13, V18; registration `fullstatesender.cpp` L526-544. Sent from `doSendSenseBody` only if `fullstate_l`/`fullstate_r` is true for the player's side (`stadium.cpp` L2270-2276; params `serverparam.cpp` L784-785, default false L1348-1349).
- Noise-free: positions and velocities `Quantize(., 0.001)`, body/neck angle (deg) `Quantize(., 0.001)`, effort/recovery `Quantize(., 0.0001)`, stamina truncated to int (V5 path, `fullstatesender.cpp` L224-261). It lists **all enabled players** (both teams) and the ball; this is the only exact ground truth a client can get (use it as the reference when evaluating denoisers).
- Known quirk: V5 `sendScore` copies the left score into right (L211-214 tests `teamLeft().enabled()` twice).

## Init (`init` replies and parameter dumps)

- `Player::sendInit()` (`player.cpp` L2393-2401): `sendInit`, `sendServerParams`, `sendPlayerParams`, `sendPlayerTypes`, `sendChangedPlayers`.
- `InitSenderCommonV8::sendServerParams` (`src/initsender.cpp` ~L418-425) iterates `ServerParam::verMap()`: only parameters whose version <= client version are sent (see `config-params`). The noise rates (`dist_noise_rate`, ... version 19), `visible_distance`/`quantize_step*` (7), `synch_see_offset` (12) arrive this way, so clients can read them at init.
- Player types (`HeteroPlayer`, `heteroplayer.cpp` L380-490) carry `unum_far_length ... flag_max_observation_length` (v18) and the four Gaussian rates (v19).

## Serializers

- `SerializerPlayerStdv1/5/7/8/13/14/18` (`serializerplayerstdv*.cpp`); factory key = integer protocol version (see `player-protocol`). Numbers are written with default `ostream` formatting (6 significant digits, no `std::fixed`); only fullstate sets `precision(6)` explicitly (`serializerplayerstdv1.cpp` L293).
- Visual helpers: `serializer.h` L343-397 (`serializeVisualObject` overloads); player-specific `serializeVisualPlayer` in Stdv8/Stdv13.

## Patterns & Conventions

- New version: subclass sender and serializer, register with `factory().autoReg(&create<T>, version)`; never patch V1 for newer behavior.
- Senders write to `transport()` then `std::flush`; one message per call.
- Timing constants come from `ServerParam` (`simStep`, `sendStep`, `senseBodyStep`, `synchSeeOffset`), already divided by `slow_down_factor` (`serverparam.cpp` L560-564, L1063-1081).

## Integration Points

- Timer: `StandardTimer::run` (`stdtimer.cpp` L95-168) / `SyncTimer::run`.
- Stadium step: `updateStamina()`/`updateCapacity()` per sim step (`stadium.cpp` L868-874).

## Important Notes

- `(synch_see)` is ignored for coaches; for players it switches off async visuals.
- Manual cross-check (code wins): manual names the async period parameter `sense_step`; the code param is `send_step` (150 ms). The manual's view-frequency width factors (narrow 2 / wide 0.5) are inverted vs code (narrow is **faster**). See visual-observation-detail "Discrepancies".
- Deprecated protocol versions (1-4 visual, 1-6 body) remain registered; do not extend them.

## See Also

- [visual-observation-detail](visual-observation-detail.instructions.md)
- [communication-hearing](communication-hearing.instructions.md)
- [noise-model](noise-model.instructions.md)
- [player-protocol](player-protocol.instructions.md)
- [config-params](config-params.instructions.md)
- [server-core](server-core.instructions.md)
- [net-logging](net-logging.instructions.md)
- [../copilot-instructions.md](../copilot-instructions.md)
