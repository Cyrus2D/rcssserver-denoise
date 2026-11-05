---
applyTo: "src/{main,stadium,field,object,weather,referee,team,timer,stdtimer,synctimer}*"
---

# Server Core (Main Loop, Stadium, World Objects, Referee, Timers)

## TL;DR

The server is a single-process, timer-driven simulator: `main()` builds a global `Stadium`, a timer (`StandardTimer` or `SyncTimer`) calls back into the stadium for receive, simulate, and send events, and each simulation cycle runs `Stadium::step()`.
- **Open the full file when:** you change the cycle order, the timer schedule, playmode/referee logic, or world-object physics.
- `main.cpp` → `ServerParam::init` → `Stadium::init` → timer `run()` (`src/main.cpp` L56-105).
- `Stadium::step()` (`src/stadium.cpp` ~L789-899): apply leg effects, update movables, update stamina, send monitors/logs, reset state.
- Objects: `PObject` (static: landmarks, lines), `MPObject` (movable: `Ball`, `Player`); physics in `src/object.cpp`.
- Referees are a list of `Referee` subclasses built in the `Stadium` constructor (`src/stadium.cpp` ~L85-94).
- Coach / online-coach messaging is out of scope; see `rcss/clang/` and `src/coach.cpp` for CLang.

## Overview

`rcssserver` runs the RoboCup 2D soccer simulation. `main.cpp` (L1-105) is short: it sets the C locale, parses config (`ServerParam::init`), installs SIGINT/SIGTERM/SIGHUP handlers that call `Std.finalize(...)`, calls `Std.init()`, selects the timer (`SyncTimer` when `ServerParam::instance().synchMode()` is true, else `StandardTimer`) and runs it.

## Architecture

### Entry and timers
- Global `Stadium Std;` in an anonymous namespace (`main.cpp` L45).
- Timer interface: `src/timeable.h` L28-55 — the timer calls `recvFromClients`, `newSimulatorStep`, `sendSenseBody`, `sendVisuals`, `sendSynchVisuals`, `sendCoachMessages`, and `quit`. `Stadium` implements the `do*` hooks (`src/stadium.h` L78-85).
- `StandardTimer::run()` (`src/stdtimer.cpp` L47-171): sleep-based loop with a TIMEDELTA tick and `lcm_step`-periodic counters. Uses `simStep`, `recvStep`, `sendStep*0.25`, `senseBodyStep`, `coachVisualStep`, `synchSeeOffset`.
- `SyncTimer::run()` (`src/synctimer.cpp` L39-140): sync mode; sends visuals at `sendStep*0.25` ticks and synch visuals at `simStep*(c-1)+synchOffset` (L60-130). Synch mode is set by the `synch_mode` setter, which recomputes `lcm_step` (`src/serverparam.cpp` ~L1089-1090).

### Stadium
- Constructor pushes referees (`src/stadium.cpp` ~L85-94): TimeRef, BallStuckRef, OffsideRef, FreeKickRef, TouchRef, CatchRef, FoulRef, IllegalDefenseRef, KeepawayRef, PenaltyRef.
- `Stadium::init()` (`src/stadium.cpp` ~L178-306): loads field, weather (`M_weather.init()` ~L288), creates objects.
- `Stadium::createObjects()` (~L514-558): one `Ball` (`MPObject`) with `setConstant(ballSize, ballDecay, ballRand, ballWeight, ballSpeedMax, ballAccelMax)`; two `Team`s (`LEFT`, `RIGHT`); `MAX_PLAYER*2` `Player`s; one `Coach`; two `OnlineCoach`es.
- `Stadium::initPlayer()` (~L562-634): assigns a team by name, calls `Team::newPlayer(version, goalie)`, `connect(addr)`, `addListener`, `M_remote_players.push_back`, `sendInit()`, `initObservationMode()`. Error replies such as `(error no_more_team)` and `(error no_more_player_or_goalie_or_illegal_client_version)`.
- `Stadium::reconnectPlayer()` (~L638-685) re-binds a disconnected player slot.
- `Stadium::playerType(id)` (~L498) returns the `HeteroPlayer` type.

### Per-cycle order (`Stadium::step()`, `src/stadium.cpp` ~L789-899)
1. Each player: `applyLegsEffect()` (dash), `resetCommandFlags()`, `incArmAge()`.
2. Online coaches and coach: reset command flags and process messages.
3. Playmode branch: `turnMovableObjects()` before kickoff; `incMovableObjects()` otherwise; referees `analyse()` each cycle.
4. Enabled players: `updateStamina()`, `updateCapacity()`.
5. Every `nrNormalHalfs*halfTime` cycles: awards online-coach free-form messages.
6. `sendDisp()` (~L940-956): monitors (`sendShow`) and `Logger` game log/profile.
7. `resetState()` for each player.

`incMovableObjects()` (~L913-936): shuffles movables, calls `_inc()` on enabled objects, runs `collisions()` (~L1649), and keeps a caught ball in front of the catcher.

### World objects
- `PObject` (positional, static): landmarks (`src/field.cpp`), goal posts, field lines.
- `MPObject` (movable): velocity, acceleration, `M_randp`, `M_weight`, max speed/accel. `_inc()` (`src/object.cpp` ~L475-520) clamps accel and speed, updates angle, adds `noise()` and `wind()`, and pushes the object out of goal posts.
- `Ball` and `Player` both derive from `MPObject`. Players are in `M_movable_objects` (`stadium.cpp` ~L627).

### Teams and players
- `Team` (`src/team.cpp`) owns up to `MAX_PLAYER` players and one coach. `Team::newPlayer(version, goalie)` (~L93-125) calls `Player::init`.
- Team-level actuator-noise factors come from `team.cpp` ~L71-77.

### Referee and play modes
- `Referee` base class (`src/referee.h` L39-90) defines hooks such as `kickTaken`, `failedKickTaken`, `tackleTaken`, `ballCaught`, `ballPunched`, `ballTouched`, `analyse()`, `playModeChange`.
- Concrete referees (referee.h): `TimeRef` (L286), `BallStuckRef` (L329), `OffsideRef` (L376), `IllegalDefenseRef` (L456), `FreeKickRef` (L512), `TouchRef` (L586), `CatchRef` (L658), `FoulRef` (L741), plus Keepaway and Penalty.
- Playmode changes: `Stadium::changePlayMode` (~L1429), `kickOff` (~L1049), `callHalfTime` (~L1113), `callFoul` (~L1151), `score` (~L1280), `placeBall` (~L1356).

## Patterns & Conventions

- Use `ServerParam::instance()` for every tunable; never hard-code values in the cycle code.
- Use `DefaultRNG::instance()` for randomness; `std::shuffle` on the player/movable lists each cycle keeps event order fair.
- Logging goes through `Logger::instance()` with `writeProfile` timing labels such as `"SIM"`, `"SB"`, `"VIS"`, `"VIS_S"`, `"COACH"` (`stadium.cpp` ~L2249, L2297, L2318, L2339, L2362).
- Stadium `do*` methods time themselves with `std::chrono::system_clock` and write profile records.
- Keep `Stadium` the single owner of cycle-level state; players and referees call back via `Stadium` methods (`kickTaken`, `tackleTaken`, `ballCaught`, `sendPlayerAudio`).

## Integration Points

- Clients: `Stadium::doRecvFromClients()` (`src/stadium.cpp` ~L2191) → `udp_recv_message()` (~L2579) → `Player::parseMsg` (see `player-protocol`).
- Sensors: `doSendSenseBody` (~L2253), `doSendVisuals` (~L2301), `doSendSynchVisuals` (~L2322) (see `observation-sensors`).
- Config: `ServerParam::init` (`src/serverparam.cpp` ~L399-480) and `PlayerParam::init` called from `serverparam.cpp` ~L478 (see `config-params`).
- Logs: `Logger` (`src/logger.cpp`) and savers (see `net-logging`).
- Coach/CLang: `src/coach.cpp`, `rcss/clang/` — out of scope here.

## Build & Test

- Source is compiled into `RCSSServer` by `src/CMakeLists.txt` (L24-115) and `src/Makefile.am` (see `build-system`).
- No unit test targets exist for the core loop; the CI smoke run only starts the binary for 3 s (see `build-system`).

## Important Notes

- `main.cpp` returns after `timer->run()` ends; `ServerParam::instance().clear()` is called on every exit path.
- `Stadium::step()` resets command flags before physics; any new command effect must be queued or applied consistently with that order.
- `synch_mode` changes `lcm_step` at runtime; timer counters assume the lcm period (`stdtimer.cpp` L163-167).
- Weather: `Stadium::init` calls `M_weather.init()` (`stadium.cpp` ~L288); wind effect per object is in `MPObject::wind()` (see `noise-model`).
- Field-specific geometry (`src/field.cpp`, `src/field.h`) is loaded at init; `Field::landmarks()` is used by visual senders.

## See Also

- [player-protocol](player-protocol.instructions.md) — client command parsing and `Player` lifecycle
- [actions-models](actions-models.instructions.md) — how each action changes state per cycle
- [observation-sensors](observation-sensors.instructions.md) — sensor timing and output
- [config-params](config-params.instructions.md) — parameter loading
- [net-logging](net-logging.instructions.md) — sockets, monitors, logs
- [build-system](build-system.instructions.md)
- [../copilot-instructions.md](../copilot-instructions.md)
