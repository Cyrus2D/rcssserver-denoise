# rcssserver-denoise — Project Instructions

## TL;DR
RoboCup Soccer Simulation 2D server (rcssserver 19.0.0, C++17, autotools + CMake).
- `src/` = server executable; `rcss/` = libs (net, conf, clang, gzip).
- Clients (e.g. helios-base on librcsc) connect via UDP; players get observations and send actions.
- Noise: quantized see model by default, Gaussian via `(gaussian_see)`; v19 bool params `full_view`, `see_no_fade_unum`, `see_no_fade_all`, `see_noise_free` (exact distance, wins over Gaussian) support denoise experiments; see noise-model instruction.
- **Open the full file when:** you need build commands or to pick the right scoped instruction.

## Project Overview
Runs a 2D soccer game: world simulation, referee, player/coach/monitor protocols, logging (rcg/rcl). Sibling repos: `../librcsc-denoise` (client library), `../helios-base-denoise` (sample client). Protocol/model docs: `../manual/source/` (Sphinx; not a Git repo). For denoise work, start with the [experiment index](../experiments/README.md) and [research guide](../experiments/RESEARCH.md).

## Architecture and Entry Points
- `src/main.cpp` -> `Stadium` (`src/stadium.*`) main loop; players in `src/player.cpp`; senders/serializers per protocol version.
- `rcss/net` sockets, `rcss/conf` config parser, `rcss/clang` coach language (out of focus), `rcss/gzip`.
- Config: `server.conf`, `player.conf` in `~/.rcssserver/` (override `RCSS_CONF_DIR`).

## Build, Test, Run
- Autotools: `./configure && make` (README.md); needs Boost, flex, bison, autotools, C++17 (README says C++14: stale).
- CMake: `mkdir build && cd build && cmake .. && make`.
- CI smoke: `.circleci/config.yml` runs rcssserver ~3 s. No test runner; `*test.cpp` files are not built. See build-system instruction.

## Conventions
Edit `src/*.ypp`/`*.lpp` grammars, not generated output. Document/modify current models only; legacy paths are marked in scoped files.

## Instruction Index
Open the entries relevant to the files you touch (on demand, not auto-loaded). For denoising work start with the observation/noise entries.

| Instruction | Covers (applyTo) | Summary | Path |
|---|---|---|---|
| Action Models (Kick, Dash, Turn, Tackle, Catch, Move, Say, Focus, Point) | `src/{player,leg,arm,object}*` | Each client action is applied in two phases: the command handler on `Player` (during receive) sets flags/state or pushes the ball immediately, and `Stadium::ste | [instructions/actions-models.instructions.md](instructions/actions-models.instructions.md) |
| Build System (CMake, Autotools, Flex/Bison, CI) | `CMakeLists.txt,**/CMakeLists.txt,configure.ac,**/Makefile.am,.circleci/**,bootstrap` | rcssserver builds with both CMake (17.0.0+) and GNU Autotools; both use C++17, Boost `system`, zlib, and generate the player command parser with Bison and Flex. | [instructions/build-system.instructions.md](instructions/build-system.instructions.md) |
| Communication and Hearing (say / hear / attentionto) | `src/{audio,player,serializerplayerstd}*` | How a `(say ...)` from one agent becomes a `(hear ...)` for others. Hearing carries **direction (integer degrees, truncated) but never distance**, and messages  | [instructions/communication-hearing.instructions.md](instructions/communication-hearing.instructions.md) |
| Configuration Parameters (server.conf / player.conf) | `src/{serverparam,playerparam,param}*,rcss/conf/**` | Server and player parameters are registered through `addParam` into a `rcss::conf::Builder`, then loaded from `server.conf` / `player.conf` and overridden by co | [instructions/config-params.instructions.md](instructions/config-params.instructions.md) |
| Networking, Monitors and Logging | `rcss/net/**,rcss/gzip/**,src/{monitor,logger,csvsaver,mysqlsaver,stdoutsaver,serializermonitor,remoteclient}*` | The server talks to clients over UDP sockets (`rcss/net/`), streams state to monitors through `DispSender*` classes, and writes game logs (`.rcg`, `.rcl`, `.kwy | [instructions/net-logging.instructions.md](instructions/net-logging.instructions.md) |
| Noise Model: Gaussian (`gaussian_see`, server v19) | `src/{visualsenderplayer,player,serverparam,heteroplayer,random}*` | `(gaussian_see)` swaps the quantizing observation for `GaussianObservation`: distance gets additive zero-mean Gaussian noise whose sigma grows with the true dis | [instructions/noise-gaussian-model.instructions.md](instructions/noise-gaussian-model.instructions.md) |
| Noise Model Hub (Visual + Actuator Noise) | `src/{player,object,heteroplayer,serverparam,visualsenderplayer,serializer}*` | rcssserver adds noise in the **visual sensor** (quantization by default, Gaussian after `(gaussian_see)`) and in **actuators/physics**. Details of each visual m | [instructions/noise-model.instructions.md](instructions/noise-model.instructions.md) |
| Noise Model: Quantization (default visual model) | `src/{visualsenderplayer,player,serverparam,utility,heteroplayer}*` | The default `see` noise model is **deterministic**: distances are rounded on a log grid, then to 0.1; no random numbers are drawn (only the hide/show ramps are  | [instructions/noise-quantize-model.instructions.md](instructions/noise-quantize-model.instructions.md) |
| Observation Sensors (See, Sense Body, Hear, Full State, Init) | `src/{visualsender,bodysender,audio,fullstatesender,initsender,serializer,sender,stdtimer}*` | Clients receive five observation streams: visual (`see`), body (`sense_body`), audio (`hear`), full state (`fullstate`), and init (`init` plus parameter dumps). | [instructions/observation-sensors.instructions.md](instructions/observation-sensors.instructions.md) |
| Player Protocol (Client UDP Commands, Player Lifecycle, Versions) | `src/{player,pcom,heteroplayer,serializer,sender,player_command}*` | Player clients send S-expression commands over UDP; a bison/flex parser turns them into `Player` method calls, and the protocol version (1-19) selects a family  | [instructions/player-protocol.instructions.md](instructions/player-protocol.instructions.md) |
| Server Core (Main Loop, Stadium, World Objects, Referee, Timers) | `src/{main,stadium,field,object,weather,referee,team,timer,stdtimer,synctimer}*` | The server is a single-process, timer-driven simulator: `main()` builds a global `Stadium`, a timer (`StandardTimer` or `SyncTimer`) calls back into the stadium | [instructions/server-core.instructions.md](instructions/server-core.instructions.md) |
| Visual Observation Detail (`see` per object, per distance, per view mode) | `src/{visualsender,visualsenderplayer,serializerplayerstd,serializercommon,object,field,player,heteroplayer,stdtimer}*` | A `see` message is built by `VisualSenderPlayerV1::sendVisual()` as flags/goals -> ball -> players -> lines. What each object carries depends on the view cone,  | [instructions/visual-observation-detail.instructions.md](instructions/visual-observation-detail.instructions.md) |

## Related Areas
- Client library: `../librcsc-denoise/.github/copilot-instructions.md` (separate repo)
- Sample client: `../helios-base-denoise/.github/copilot-instructions.md` (separate repo)

## PR Review — Instruction Maintenance
GitHub Copilot PR reviewers: use the Instruction Index to read scoped guidance
relevant to the changed paths, respecting each repository/submodule's ownership.
When substantive changes to architecture, interfaces, workflows, build or
configuration behavior, or safety constraints make relevant instructions stale
or incomplete, report the maintenance need as a review finding/recommendation.
Identify the affected instruction and the claim or gap, with change evidence;
judge significance semantically, not by a changed-line threshold. This note does
not authorize reviewers to edit instructions and does not guarantee Copilot loads
all instruction files; follow relevant links on demand.
