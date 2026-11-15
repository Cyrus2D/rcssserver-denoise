# Exp4 step 08 · Inspect stock and corrected state in Soccer Window

## Ready to use

The inspector is implemented and built in
`/home/nader/workspace/denoise/helios-base-denoise/build-debug/bin`.
Its private librcsc is installed under `/home/nader/local-denoise-debug`.
The original `build/bin` and `/home/nader/local/lib` runtime remain separate,
so the current batch keeps its original executables and library.

The models, guards, forecasts and decisions are unchanged by this work.
Normal `start.sh` defaults to inspector off. `start-debug.sh` now enables
the focus inspector and saves `.dcl` replay files as well as ordinary
offline/debug logs. Run an inspection game separately from the scored batch.

## Known display limitation discovered during live inspection

Circle colors are currently omitted by librcsc's CirclePrinter for nonempty
color strings, so several outlines can appear in the viewer's default
color. The table below is the intended legend. Coordinates and numeric
annotations were validated; styling was not. See the
[closeout audit](09-closeout-ball-stability.md). No runtime fix was made
during that read-only audit.

## What the colors mean

| Color | Meaning |
|---|---|
| Gray | Stock WM state captured immediately before decision overrides |
| Cyan | Final WM state actually used for this decision |
| Purple | Available model prediction for a field that was not applied |
| Green | Current server truth, only with reference fullstate |
| Orange connector | Position displacement between stock and used |

Circles show positions. Arrows show velocity multiplied by three cycles by
default, for visibility; they are **not learned next-cycle forecasts**.
Short direction lines show body angle. The positions themselves are not
magnified: zoom in to see small corrections.

The banner shows exact stock/used position and velocity, body angle and
ages, position displacement, raw observation distance, error against truth,
and field application reasons. Units are metres, metres/cycle and degrees.
`ep`, `ev`, `eb` mean position distance error, velocity distance error and
wrapped body-angle error. No invalid field is silently displayed as zero.

Reasons include `applied`, `shadow`, `protected:range<5.000`,
`protected:range<10.000`, `observed:preserved`, and
`unavailable:warmup/ineligible`. The last covers insufficient history,
non-PlayOn or an unsupported observation configuration.

The inspector always details the ball and optionally one target player.
For that player it also shows the stock-to-used ball distance and nominal
kicking radius. The self kickable flag is shown before and after overrides.
Geometric proximity alone is not proof that an opponent will kick.

## Launch

Use the same server experiment settings, including reference fullstate.
Open Soccer Window and enable its debug-server listener. With the standard
game port 6000, the default debug port is 6032.

Example left team with the current models:

```sh
cd /home/nader/workspace/denoise/helios-base-denoise/build-debug/bin

DEBUG_DIR=/home/nader/workspace/denoise/logs/exp4/debug/manual-01
mkdir -p "$DEBUG_DIR"

./start-debug.sh -p 6000 -P 6002 -t lt --fullstate reference \
  --player-position-denoise apply \
  --player-position-model /home/nader/workspace/denoise/logs/exp3-context-runs/step26-polar/player-position.weights \
  --player-position-routing corrected \
  --exp4-state apply --exp4-model-dir /home/nader/workspace/denoise/logs/exp4/weights \
  --denoise-debug-observer 6 \
  --denoise-debug-target opponent:7 \
  --debug-server-host localhost --debug-server-port 6032 \
  --log-dir "$DEBUG_DIR" \
  --denoise-log-dir "$DEBUG_DIR/dataset" --denoise-game-id manual-01
```

Launch the right team normally with its original settings. In Soccer Window,
select **left player 6's debug view**. It shows how that observer estimates
the ball and opponent 7; selecting player 7's own view is a different
observer. Retain the matching server game log for replay.

If adapting your runner, change the inspection team's BASE_PATH to
`.../helios-base-denoise/build-debug/bin`, use `start-debug.sh`, keep all
model flags, and append the inspector options above. For a nonstandard game
port, set the debug host/port to the viewer listener explicitly; the script's
automatic debug port is game port + 32.

## Controls

| Option | Values and default |
|---|---|
| `--denoise-debug` | `off`, `focus`, `overview`; normal start defaults off, debug wrapper defaults focus |
| `--denoise-debug-observer` | Agent unum 1–11; 0 means every observer, the default |
| `--denoise-debug-target` | `nearest` (player nearest the estimated ball), `ball`, `teammate:N`, `opponent:N`; default nearest |
| `--denoise-debug-velocity-scale` | Positive value up to 10; default 3 |

`focus` draws detailed ball and target state. `overview` additionally
draws stock/used positions of all known players, while keeping numeric
detail restricted to the ball and target. If the target is unavailable,
the banner says so.

This dedicated inspector replaces planner geometry and its banner to keep
the display readable. Ordinary behavior debug logs remain available.
Pass `--denoise-debug off` to the debug wrapper to restore the ordinary
debug view.

With position apply, the inspector requires corrected routing: legacy routing
can apply position during visual processing before the baseline capture.
Invalid options are rejected before launching players.

## Replay and interpretation

`start-debug.sh` now saves `<team>-<unum>.dcl` under `--log-dir`, using
the existing viewer format, and ordinary agent `.log` and offline files.
The `.dcl` contains the overlays; point Soccer Window's debug-log directory
to that folder and use the matching game log. Full detail is also emitted
on Logger::WORLD when that category is enabled.

Stock means **the same observer's pre-override WM this cycle**. It is not
the raw visual localization and is not a separately simulated original
team. Subsequent normal WM processing is included in the final used state;
application labels identify fields actually supplied by a model.
Fullstate is read only to annotate the inspector and never feeds the models
or decisions.

The library keeps optional baseline snapshots even during model warmup,
guarded cycles and when models are off. The snapshot is timestamped by
cycle and stopped count. Inspection packets preserve stopped counts during
BeforeKickOff, avoiding the legacy display's collapsed startup times.

## Validation

| Check | Result |
|---|---|
| librcsc build/install to private prefix; separate Helios build | Passed |
| Snapshot immutability, player identity mapping, invalid-field masks | Passed |
| Existing F4 restoration, guards, body preservation and forecast-path fixture | Passed |
| Oversized banner/comments and complete packet serialization | Passed actual viewer parser |
| Left apply / right stock, overview, 600 scheduled cycles | 22 finalized agents |
| Right apply / left stock, focus on opponent 7, 600 scheduled cycles | 22 finalized agents |
| Actual Soccer Window parser: captured UDP plus saved DCL | 2,618 packets/replay entries accepted |
| Numeric parity with corresponding CSVs, both sides | 17,902 position/velocity/body fields matched within 0.00056 units |
| Largest left overview / right focus UDP packet | 6,321 / 4,167 bytes, including NUL |
| Model-write mismatches, range-guard violations, observed-body overwrites, applied forecasts | Zero in both matches |
| Invalid mode, target, observer, arrow scale and missing value | Rejected |

Both matches exercised warmup/ineligible cases, applied corrections,
observed-body preservation and the 5 m / 10 m guards. Right-side parity
includes coordinate and angle conversion to the server-frame CSVs.

The viewer has separate limits: 8,192-byte receive buffer, 1,023-character
message, 256-character object comments. Inspector packets cap total size
at 7,800 bytes and use shorter text/comment limits, removing complete
optional elements instead of truncating protocol syntax. Unusually dense
data is marked `TEXT CLIPPED` or `OVERLAY CLIPPED`. The normal inspector
matches required neither marker.

These tests validate data and parser compatibility. They are not a new
match-performance experiment or a manual GUI inspection.

[Saved verification](debug-overlay/verification.json) ·
[Left parity](debug-overlay/l-parity.json) ·
[Right parity](debug-overlay/r-parity.json)

## Code and reproduction

- librcsc: optional WorldModel snapshot before the decision overrides;
  DebugClient inspection helpers and opt-in protocol limits.
- Helios: `src/player/denoise/debug_overlay.*` owns presentation, selection
  and truth annotations. No network weights or rendering logic moved into
  librcsc.
- Both CMake and autotools source lists include the new helper.
- Test sources: `verification/debug_snapshot.cpp`,
  `debug_viewer_parse.cpp`, `debug_overlay_parity.py`; the existing
  `run_smoke.py` accepts `--base`, `--debug-overlay`,
  `--debug-target` and `--debug-observer`.

Build the private debug runtime from the workspace:

```sh
cmake --build librcsc-denoise/build -j4
cmake --install librcsc-denoise/build --prefix /home/nader/local-denoise-debug
cmake -S helios-base-denoise -B helios-base-denoise/build-debug \
  -DLIBRCSC_INSTALL_DIR=/home/nader/local-denoise-debug -DCMAKE_BUILD_TYPE=Release
cmake --build helios-base-denoise/build-debug -j4
```

Do not replace the runtime under an ongoing scored batch. After switching
the normal installation to these changed headers/library, rebuild all
linked agents together.
