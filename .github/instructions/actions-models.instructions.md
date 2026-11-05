---
applyTo: "src/{player,leg,arm,object}*"
---

# Action Models (Kick, Dash, Turn, Tackle, Catch, Move, Say, Focus, Point)

## TL;DR

Each client action is applied in two phases: the command handler on `Player` (during receive) sets flags/state or pushes the ball immediately, and `Stadium::step()` realizes the rest (dash accel, object movement, noise) once per cycle.
- **Open the full file when:** you change how an action is applied, its cycle timing, or stamina/effort/noise terms.
- Dash: stored in `Leg` (`src/leg.h`), realized in `applyDashEffect()` at the start of `Stadium::step()`.
- Turn/turn_neck/change_focus: immediate angle/focus updates, one per cycle.
- Kick/tackle/long_kick: compute an acceleration (with noise) and call `Stadium::kickTaken` / `tackleTaken`, which pushes the ball immediately.
- Catch (goalie): probabilistic success; success pulls the ball to the body, failure punches it (`src/player.cpp` ~L1633-1697).
- Move: only in before-kickoff / after-goal / free-kick-with-catch playmodes.

## Overview

The action models describe how a `Player` command becomes physical change. The manual's current models are in `manual/source/soccerserver/action-models-*.rst` and `movement-models.rst`, `bipedal-dash-model.rst`. This file documents the code path in `rcssserver-denoise/src/`.

## Architecture

### Cycle order (authoritative: `Stadium::step`, `src/stadium.cpp` ~L789-899)
1. `Player::applyLegsEffect()` → `applyDashEffect()` (`src/player.cpp` ~L1197-1254): consumes stamina, builds left/right leg accelerations, pushes the player, and updates body angle from leg asymmetry (`omega × (1 + U(-randp,randp))`, ~L1248-1250).
2. `resetCommandFlags()` per player (`player.cpp` ~L2754) and `incArmAge()`.
3. Playmode-dependent update: `turnMovableObjects()` before kickoff, else `incMovableObjects()` → each movable `_inc()` (`src/object.cpp` ~L475-520): clamp accel and speed, `updateAngle()`, add `noise()` and `wind()`, post collision. Then `collisions()`.
4. Stamina/capacity update (`updateStamina`, `updateCapacity`, `stadium.cpp` ~L868-874).
5. `sendDisp()` then `resetState()`.

Commands themselves arrive during `doRecvFromClients` (see `player-protocol`), i.e. between cycles, and are applied to state that step 1-3 consumes.

### Per-action behavior

| Action | Entry (`src/player.cpp`) | Immediate effect | Realized in step | Key constants / params |
|---|---|---|---|---|
| `dash(power[,dir])` | `dash` ~L1094 → `dashLeftLeg/dashRightLeg` ~L1165-1195 | stores power in each `Leg` (`Leg::dash`, `src/leg.cpp`) | `applyDashEffect` ~L1205-1254 | `dash_power_rate`, `effort`, `extra_stamina`, `player_rand` |
| `turn(moment)` | ~L1263-1277 | `M_angle_body` updated with `(1+U(-randp,randp)) × moment / (1 + inertia × |vel|)`; `M_command_done` | `updateAngle` in `_inc` | `inertia_moment`, `player_rand` |
| `turn_neck(moment)` | ~L1280-1289 | `M_angle_neck` (clamped) once per cycle | none (state) | `min/max_neck_angle`, `min/max_neck_moment` |
| `change_focus(dist,dir)` | ~L1292-1302 | `M_focus_dist` ∈ [0,40]; `M_focus_dir` clamped to visible angle | used by visuals | `visible_angle` |
| `kick(power,dir)` | ~L1305-1419 | rejected in stoppage modes (`KICK_FAULT`); if ball kickable, computes accel + noise and calls `M_stadium.kickTaken` (ball `push`) | ball moves in `_inc` | `kick_power_rate`, `kick_rand` (type × team), `kickable_margin`, `max_power` |
| `long_kick(power,dir)` | ~L1422-1520 | staged; executed by `doLongKick()` ~L1442 with delay | ball pushed later | `LONG_KICK_POWER_FACTOR=2`, `LONG_KICK_DELAY=2` (`serverparam.cpp` ~L389-390) |
| `tackle(power[,foul])` | ~L2059-2250 | sets `tackle_cycles`; ball-relative geometry; success calls `M_stadium.tackleTaken` with accel + tackle noise | ball moves in `_inc` | `tackle_dist`, `tackle_back_dist`, `tackle_exponent`, `tackle_cycles`, `tackle_rand_factor` |
| `catch(dir)` (goalie) | `goalieCatch` ~L1523-1700 (the `#if 0` block at ~L1569-1604 is dead) | outside catch area → `CATCH_FAULT`; success → player moved to ball, `ballCaught`; failure → `ballPunched` with reflected velocity | none | `catch_probability`, `catch_area_length/width`, `catch_area_l_stretch`, `catch_ban_cycle` |
| `move(x,y)` | ~L1728-1779 | only in `PM_BeforeKickOff`, `PM_AfterGoal_*`, or free kick with ball catcher; otherwise ignored | none | `goalie_max_moves` |
| `say(msg)` | ~L1704-1713 | length check (`say_msg_size`), `Stadium::sendPlayerAudio` | none | `say_msg_size` |
| `sense_body` | ~L1716-1719 | immediate `BodyObserver::sendBody` | see `observation-sensors` | `sense_body_step` |
| `pointto(dist,head)` | ~L1962-1974 | `M_arm.pointTo(...)` (arm, `src/arm.h`) or `stopPointing` | arm age incremented per cycle | `arm` params |
| `attentionto` | ~L1977-2050 | sets focus target by team/unum; `off` clears | used by audio/visual | |
| `change_view(width[,quality])` | ~L1782-1927 | sets `M_visible_angle`, `M_visual_send_interval`, `M_high_quality` | visual cadence | `visible_angle` |

Notes on kick/tackle noise and the full formulas are in `noise-model`. Summary: kick noise radius `U(0, kick_rand × power/maxPower × (pos_rate+speed_rate))` with `pos_rate, speed_rate ∈ [0.5,1]`; tackle uses `tackle_rand_factor` (2.0 default, `serverparam.cpp` ~L361).

### Objects and physics (`src/object.cpp`)
- `MPObject::_inc()` (~L475-520): `accel` capped by `maxAccel()` (player: slower on top half, `player.cpp` ~L2550-2559), `vel += accel`, speed capped by `maxSpeed()`, `vel += noise()`, `vel += wind()`, then post-collision push.
- `MPObject::noise()` (~L446-454): radius `U(0, randp×|vel|)`, random angle. Ball `randp = ball_rand`; players `randp = player_rand` (× team factor when `team_actuator_noise`).
- `Ball` decays and collides via `Stadium::collisions()` (~L1649).

### Leg and Arm
- `Leg` (`src/leg.h`): `CommandType { NONE, DASH, TURN, KICK, TACKLE, MOVE? }` (enum ~L33-38), stores `dashPower`, `dashDir`, `kickPower`, `kickDir`. `calcDashAccel(consumed_stamina)` gives leg acceleration.
- `Arm` (`src/arm.h`): `pointTo`, `stopPointing`, `incArmAge` (called in `Stadium::step`).

## Patterns & Conventions

- Checks guard with `M_command_done` (one movement per cycle): a second command in the same cycle is ignored.
- Failure outcomes set state bits: `KICK_FAULT`, `TACKLE_FAULT`, `CATCH_FAULT`, `POST_COLLIDE`, read later by the referee and sensors.
- Constants are read through `ServerParam::instance()`; per-type values through `HeteroPlayer` (`src/heteroplayer.h`).

## Integration Points

- Referees receive `kickTaken`, `tackleTaken`, `ballCaught`, `ballPunched`, `failedKickTaken` (`src/referee.h`).
- Monitor/logger record `KICK`, `DASH`, etc. through `Logger::writePlayerLog` (`src/logger.cpp`).
- Sensors read the committed state (`angleBodyCommitted`, `angleNeckCommitted`, `stamina`, `effort`) (see `observation-sensors`).

## Build & Test

- Behavior is verified by running a match or a unit harness; no action-level unit tests exist (`serverparamtest.cpp`/`playerparamtest.cpp` are not built).
- Changing physics affects replay compatibility: keep log/monitor fields stable.

## Important Notes

- The code's `applyKickEffect` is commented out (`player.cpp` ~L1256-1260); kicks apply immediately, not in `step()`.
- `goalieCatch` has two implementations: the live one (~L1600-1700) and a disabled `#if 0` block (~L1569-1604). Edit the live one.
- `dash(power, dir)` has an `#if 1 … #else` split (`player.cpp` ~L1099-1160); the `#else` branch is inactive.
- Manual vs code: the dash/kick/tackle formulas in `manual/…/action-models-*.rst` should be checked against `player.cpp` before updating either; code is authoritative.

## See Also

- [noise-model](noise-model.instructions.md) — formulas for kick, tackle, dash, and object noise
- [player-protocol](player-protocol.instructions.md) — command parsing and version gates
- [server-core](server-core.instructions.md) — cycle order and object classes
- [observation-sensors](observation-sensors.instructions.md)
- [../copilot-instructions.md](../copilot-instructions.md)
