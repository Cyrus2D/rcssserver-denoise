---
applyTo: "CMakeLists.txt,**/CMakeLists.txt,configure.ac,**/Makefile.am,.circleci/**,bootstrap"
---

# Build System (CMake, Autotools, Flex/Bison, CI)

## TL;DR

rcssserver builds with both CMake (17.0.0+) and GNU Autotools; both use C++17, Boost `system`, zlib, and generate the player command parser with Bison and Flex.
- **Open the full file when:** you add a source file, change dependencies or the C++ standard, touch the parser grammar/lexer, or edit CI.
- CMake sets `CMAKE_CXX_STANDARD 17` (`CMakeLists.txt` L5-6); `configure.ac` checks C++17 and sets version `19.0.0` (L6).
- `README.md` L14 says "C++14", which is stale; the build and README L17 require C++17.
- Parser generation: `src/CMakeLists.txt` `bison_target` / `flex_target` (L2-8), and `src/Makefile.am` `.ypp`/`.lpp` rules (~L172-190).
- CI (`.circleci/config.yml`) only runs the autotools build and a 3-second smoke run that cannot fail.

## Overview

- Top-level `CMakeLists.txt` configures the project, finds dependencies, and adds subdirectories.
- `configure.ac` / `Makefile.am` provide the autotools path; `bootstrap` is the helper script for it (not reviewed in depth).
- `rcss/` is a set of sub-libraries (`net`, `conf`, `gzip`, `clang`), added by `rcss/CMakeLists.txt` L1-4 and `rcss/Makefile.am` L2 (`SUBDIRS = net conf gzip clang`).
- `src/` produces `rcssserver`, `rcssclient`, and the saver plugins.

## Architecture

- **Dependencies (CMake):** `find_package(ZLIB REQUIRED)` (L15), `find_package(BISON REQUIRED)` and `find_package(FLEX REQUIRED)` (L19-20), `find_package(Boost 1.44.0 COMPONENTS system REQUIRED)` (L22). A commented-out `filesystem` component is on L21.
- **Autotools:** `configure.ac` `AC_INIT([RCSSServer],[19.0.0],...)` (L6); uses `AX_PROG_FLEX` (L120) and `AX_PROG_BISON`; C++17 is enforced through `AX_CXX_COMPILE_STDCXX_17` (verified in the earlier research pass).
- **Parser generation:**
  - CMake: `bison_target(player_command_parser player_command_parser.ypp ...)` and `flex_target(player_command_tokenizer player_command_tok.lpp ...)` in `src/CMakeLists.txt` (L2-8), plus an `add_custom_command` (~L13).
  - Autotools: `src/Makefile.am` lists `player_command_parser.ypp` and `player_command_tok.lpp` (~L73, ~L217-218). `.lpp.cpp` suffix rule (~L176-182) runs `$(FLEX)` and renames `lex.yy.c` to the target name.
- **Library layout:** `rcss/net` (sockets), `rcss/conf` (config builder/parser), `rcss/gzip` (gzip streams), `rcss/clang` (coach language; out of scope, see [server-core](server-core.instructions.md)).
- **Source lists:** `src/CMakeLists.txt` lists `csvsaver.cpp` and `dispsender.cpp` (L28-29) and `stdoutsaver.cpp` (L78); `src/Makefile.am` mirrors them (~L11-12, ~L61, ~L82-83, ~L136).
- **Non-CMake/Autotools build:** `src/SConscript` builds `libstdoutsaver` (L117-124) and `libcsvsaver` (L127-136) as shared libraries.

## Patterns & Conventions

- Keep CMake and Autotools source lists in sync: a new `.cpp` must be added to `src/CMakeLists.txt` and `src/Makefile.am` (and `SConscript` if the SCons build should include it).
- Generated parser/lexer sources come from `.ypp`/`.lpp` at build time; edit the grammar/lexer, not generated `.cpp` files.
- Prefer the CMake path for new development; `configure.ac` also contains commented-out legacy checks (e.g. `#  AC_CHECK_PROG([FLEX],...)`), ignore those.

## Integration Points

- The parser output feeds `Player` command handling; see [player-protocol](player-protocol.instructions.md).
- Saver shared libraries are loaded at runtime; see [net-logging](net-logging.instructions.md).
- Runtime config location (`RCSS_CONF_DIR`, `server.conf`) is set by the CI smoke step; see [config-params](config-params.instructions.md).

## Build & Test

- **CMake:** `cmake` + build; requires ZLIB, Bison, Flex, Boost `system` 1.44+.
- **Autotools (README L29-39):** install `build-essential automake autoconf libtool flex bison libboost-all-dev`, then `./configure && make`.
- **CI (`.circleci/config.yml`):**
  - Image `ubuntu-2004`; installs build deps; runs `aclocal`, `autoreconf --install`, `automake --add-missing`, then `./configure`, `make`, `make install`, `ldconfig`.
  - Smoke step (L36-38): `RCSS_CONF_DIR=. timeout 3 rcssserver || echo Run successfully`.
- **Unit-style test files:** `serverparamtest.cpp` and `playerparamtest.cpp` are not referenced by `CMakeLists.txt`, `src/Makefile.am`, or `src/SConscript`, so they are not built or run by any configured target.
- **No test framework:** no `add_test` / `enable_testing` in CMake.

## Important Notes

- **CI always passes:** `timeout 3 rcssserver || echo Run successfully` runs the echo on any exit status, including failure. A broken server binary will not fail the build.
- **Stale doc:** `README.md` L14 says C++14; the code and CMake use C++17.
- **Version drift:** `configure.ac` says 19.0.0; README L58 says CMake support began in 17.0.0. Check both when changing the release version.
- Do not edit generated outputs (`lex.yy.c`, bison-generated `.cpp/.hpp`); regenerate from `.ypp` / `.lpp`.
- `bootstrap` and the exact autotools macro set (`m4/`) were not reviewed in depth.

## See Also

- [player-protocol](player-protocol.instructions.md) — parser grammar and command handling
- [net-logging](net-logging.instructions.md) — saver and log sources in the build lists
- [config-params](config-params.instructions.md) — runtime configuration used by the smoke test
- [server-core](server-core.instructions.md) — main server sources
