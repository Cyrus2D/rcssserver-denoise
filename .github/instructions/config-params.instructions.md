---
applyTo: "src/{serverparam,playerparam,param}*,rcss/conf/**"
---

# Configuration Parameters (server.conf / player.conf)

## TL;DR

Server and player parameters are registered through `addParam` into a `rcss::conf::Builder`, then loaded from `server.conf` / `player.conf` and overridden by command-line options.
- **Open the full file when:** you add, rename, or change the default of a server/player parameter, or debug why a conf value is ignored.
- Config directory: `.` on Windows/Cygwin, `~/.rcssserver/` elsewhere; `RCSS_CONF_DIR` overrides it.
- Each `addParam` call also records a protocol version in `M_ver_map`, which the server uses when it sends parameters to clients.
- Parse order: conf file first, then command-line options (`M_conf_parser->parse(argc, argv)`).

## Overview

- `ServerParam` (`src/serverparam.h/.cpp`) is a singleton holding every server tunable (ports, timing, game rules, noise rates, logging).
- `PlayerParam` (`src/playerparam.h/.cpp`) holds player-side tunables (`player.conf`).
- `CSVSaverParam` (`src/csvsaver.cpp`) holds its own `CSVSaver.conf`; its `init` is called from `ServerParam::init`.
- The generic conf engine lives in `rcss/conf/` (`builder.cpp/.hpp`, `parser.cpp/.hpp`, `paramsetter.hpp`, `paramgetter.hpp`, `statushandler.*`, `streamstatushandler.*`).

## Architecture

- **File location** (`src/serverparam.cpp`):
  - `CONF_DIR` is `"."` on `RCSS_WIN` / `__CYGWIN__` and `"~/.rcssserver/"` otherwise (~L230-238).
  - `SERVER_CONF = "server.conf"`; `OLD_SERVER_CONF = "rcssserver-server.conf"`.
  - `ServerParam::init(argc, argv)` (~L400) reads `RCSS_CONF_DIR` if set, creates the directory if missing, and calls `parseCreateConf(conf_path, "server")` (~L455). If the file does not exist it is created with defaults.
- **Version gate** (~L463): `M_builder->version()` must match `M_builder->parsedVersion()`, otherwise the file must be regenerated.
- **Sub-modules:** `PlayerParam::init(builder)` (~L478, `src/playerparam.cpp` ~L165) and `CSVSaverParam::init` (~L485) register into the same builder. Player file is `player.conf` (`PLAYER_CONF`, `src/playerparam.cpp` ~L84-89).
- **Registration:** `ServerParam::addParams()` (`src/serverparam.cpp` ~L625 onward) calls `addParam(name, member, desc, version)` or the setter/getter overload `addParam(name, setter, getter, desc, version)` (~L603-622). Both forward to `M_builder->addParam` and store `M_ver_map[name] = version`.
- **Version numbers:** examples from `addParams()`: `goal_width` is 7, `port` / `text_log_compression` are 8, `kick_off_wait` is 9. Parameters added later use higher numbers (e.g. the Gaussian noise rates and the bool params `full_view`, `see_no_fade_unum`, `see_no_fade_all`, `see_noise_free` (all default false; `constexpr` defaults near `serverparam.cpp` L386-389, `addParam` L964-967) are registered at v19, see [noise-model](noise-model.instructions.md)).
- **Default ports** (`src/serverparam.cpp` ~L119-121): `DEFAULT_PORT_NUMBER = 6000`, `COACH_PORT_NUMBER = 6001`, `OLCOACH_PORT_NUMBER = 6002`; registered as `port`, `coach_port`, `olcoach_port` (~L726-728).

## Patterns & Conventions

- To add a server parameter: add the member and accessor in `ServerParam` (`serverparam.h`), initialise the default in the constructor/`clear` path, and call `addParam(...)` in `addParams()` with a version number matching the protocol version it appears in.
- Use the setter/getter `addParam` overload when a setter must validate or derive state (e.g. `player_size` uses `rcss::conf::makeSetter( this, &ServerParam::setPlayerSize )`, ~L629).
- Player-side parameters follow the same pattern in `PlayerParam::addParams` / `playerparam.cpp`.
- Command-line parsing (`M_conf_parser->parse(argc, argv)`, ~L492) runs after the file is loaded, so the parser's own option handling is applied on top of file values (check `rcss/conf/parser.cpp` for the exact option syntax before documenting it).
- `ServerParam::init` returns `false` on any failure, and then `instance().clear()` runs. Do not rely on partially initialised state.

## Integration Points

- **Clients:** parameters are sent to players in the init message. `sendServerParams` iterates `M_ver_map` (`src/initsender.cpp` ~L419-425), so the version number decides which clients receive a parameter.
- **Builder version check:** `rcss/conf/builder.cpp` ~L691-694 compares versions when parsing.
- **Heterogeneous players:** `HeteroPlayer` copies the noise rate parameters (`src/heteroplayer.cpp` ~L266-274). See [noise-model](noise-model.instructions.md).
- **Logging:** `text_log_dir`, `game_log_dir`, `text_log_compression`, `game_log_compression` are registered at `src/serverparam.cpp` ~L793-811; the logger uses them (see [net-logging](net-logging.instructions.md)).
- **Server core:** `Stadium::init` reads these values (see [server-core](server-core.instructions.md)).

## Build & Test

- `serverparamtest.cpp` and `playerparamtest.cpp` are not listed in `CMakeLists.txt`, `src/Makefile.am`, or `src/SConscript`, so they are not built. Treat them as manual scratch code unless someone adds them.
- No `add_test` or `enable_testing` in `CMakeLists.txt`; config changes are verified by starting the server and checking the output of `--help`/the created conf file.

## Important Notes

- The default conf directory on Windows is the current working directory, so `server.conf` is created wherever the server is started.
- The version check (`ServerParam::init` ~L463) fails hard when the file version differs. Regenerate the file rather than editing the version line.
- `CONF_DIR` has a commented-out alternative (`~\\.rcssserver\\`) in `serverparam.cpp` ~L232; ignore it.
- Some legacy file names (`rcssserver-server.conf`, `~/.rcssserver-player.conf`) are migrated by `convertOldConf` / `PlayerParam::init` (`src/playerparam.cpp` ~L232-249).

## See Also

- [server-core](server-core.instructions.md) — consumers of server parameters
- [noise-model](noise-model.instructions.md) — noise parameters and their registration
- [net-logging](net-logging.instructions.md) — log directory and compression parameters
- [player-protocol](player-protocol.instructions.md) — protocol versions that gate parameter visibility
