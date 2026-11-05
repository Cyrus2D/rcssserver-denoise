---
applyTo: "rcss/net/**,rcss/gzip/**,src/{monitor,logger,csvsaver,mysqlsaver,stdoutsaver,serializermonitor,remoteclient}*"
---

# Networking, Monitors and Logging

## TL;DR

The server talks to clients over UDP sockets (`rcss/net/`), streams state to monitors through `DispSender*` classes, and writes game logs (`.rcg`, `.rcl`, `.kwy`) through `Logger` and result savers.
- **Open the full file when:** you change socket handling, monitor output formats, log file naming, or compression.
- Default ports: player 6000, coach 6001, online coach 6002 (`src/serverparam.cpp` ~L119-121).
- Logs are gzip-compressed when `game_log_compression` / `text_log_compression` > 0 (`src/logger.cpp` ~L297-309, ~L385-397).
- Monitor output has three versions plus JSON (`src/dispsender.cpp` ~L1137-1142); `monitor_protocol-v3.txt` documents v3.
- Savers (CSV, stdout, MySQL) are plugins/sources; only some are built (see Build & Test).

## Overview

- `rcss/net/` is a small socket library: `udpsocket.*`, `tcpsocket.*`, `socket.*`, `addr.*`, `handler.*`, plus stream adapters `socketstreambuf.*`, `isocketstream.hpp`, `osocketstream.hpp`, `iosocketstream.hpp`.
- `rcss/gzip/` provides `gzofstream` and related gzip stream classes used by the logger.
- `src/` holds the server-side consumers: `logger.*` (game/text log writers), `dispsender.*` (monitor and logger senders), `serializermonitor.*`, `csvsaver.*`, `stdoutsaver.cpp`, `mysqlsaver.cpp`, and `remoteclient.*`.

## Architecture

- **Sockets:** `UDPSocket` in `rcss/net/udpsocket.cpp/.hpp` is used by `Stadium` for client messages. The receive path (`Stadium::udp_recv_message`) is covered in [server-core](server-core.instructions.md); the body of the dispatch was not reviewed in depth.
- **Ports:** player 6000, coach 6001, online coach 6002 (`src/serverparam.cpp` ~L119-121; registered as `port`, `coach_port`, `olcoach_port` ~L726-728).
- **Log file naming:** `Logger` writes files with suffixes `.rcl` (text log), `.rcg` (game log), and `.kwy` (keepaway log). The paths come from `text_log_dir` / `game_log_dir` (`src/serverparam.cpp` ~L793-797).
- **Compression:** when the compression parameter is > 0 the logger uses `gzofstream` instead of `std::ofstream` (`src/logger.cpp` ~L297-309 and ~L385-397).
- **Monitor senders:** `DispSenderMonitor` supports V1, V2, V3, and JSON output (`src/dispsender.cpp` ~L1137-1142). `DispSenderLogger` writes the log files (~L1152-1158).
- **Monitor protocol:** `monitor_protocol-v3.txt` in the repo describes the v3 wire format. The file is EUC-JP encoded; read it with an appropriate decoder. The log header magic is `ULG4`.
- **Savers:** `CSVSaver` is registered through the `ResultSaver` mechanism (`src/csvsaver.cpp` ~L420) and reads `CSVSaver.conf` (~L100). `stdoutsaver.cpp` is also built as a shared library through `src/SConscript` (~L117-124).

## Patterns & Conventions

- Logs and monitors are write-only from the server's perspective; keep formatting changes backward compatible, since existing external monitor and log readers depend on them.
- Gzip output is selected by a runtime parameter, not a build option; keep both paths working when changing `Logger`.
- Monitor variants are registered through a per-class `factory()` (`DispSenderMonitor::factory()`, `src/dispsender.cpp` ~L79); add new versions as new `DispSenderMonitor*` subclasses rather than changing v1-v3 output. The exact selection path was not traced.

## Integration Points

- `Stadium::init` opens the logger (`src/stadium.cpp` ~L298), and `Stadium` calls `Logger::instance()` during the cycle, e.g. `writeGameLog` / `flush` (~L951-955) and `writeTextLog` (~L1581, ~L1645) (see [server-core](server-core.instructions.md)).
- Parameters for ports, log directories, and compression come from [config-params](config-params.instructions.md).
- Player/coach message handling is in [player-protocol](player-protocol.instructions.md); the sensor data sent to clients is in [observation-sensors](observation-sensors.instructions.md).
- `remoteclient.*` and `serializermonitor.*` were listed but their roles were not traced in this pass.

## Build & Test

- `rcss/CMakeLists.txt` and `rcss/Makefile.am` add `net`, `conf`, `gzip`, and `clang` subdirectories (`rcss/CMakeLists.txt` L1-4; `rcss/Makefile.am` L2).
- `src/CMakeLists.txt` lists `csvsaver.cpp`, `dispsender.cpp` (~L28-29) and `stdoutsaver.cpp` (~L78).
- `src/Makefile.am` lists the same sources (`csvsaver.cpp` ~L11, `dispsender.cpp` ~L12, `stdoutsaver.cpp` ~L61) and their headers.
- `src/SConscript` builds `libstdoutsaver` (~L117) and `libcsvsaver` (~L127) as shared libraries.
- `mysqlsaver.cpp` does not appear in `src/CMakeLists.txt` or `src/Makefile.am`; treat it as not built by those systems. It reads `<module>.conf` from a user directory (`src/mysqlsaver.cpp` ~L75).

## Important Notes

- Logs are binary/compressed when compression is enabled; use the existing readers rather than `read_file`.
- Do not change the default ports without updating client documentation (`manual/`).
- The monitor protocol document is authoritative for v3 clients; JSON output is a separate path.

## See Also

- [server-core](server-core.instructions.md) — who drives the logger and monitors each cycle
- [config-params](config-params.instructions.md) — log directory, compression, and port parameters
- [observation-sensors](observation-sensors.instructions.md) — data sent to players
- [build-system](build-system.instructions.md) — which savers and libraries are built
