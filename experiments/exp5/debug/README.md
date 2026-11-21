# Exp5 visual debugger: original WM versus used position

**Ready in the rebuilt `helios-base-denoise/build/bin`.** This build includes
the cache-v2 integration correction described in the [final report](../reports/final/README.md).
The old200-game results apply to the earlier implementation, not this build.

## What the symbols mean

| Symbol | Meaning |
|---|---|
| Orange outline circle | Original WM position (`trackedPos`) |
| Cyan outline square | Actually used position (`pos`) |
| Cyan connecting line | Position correction; not velocity |
| `APPLY pc=8>0` | Forecast used; original count8, effective count0 |
| `WM:fresh` | Current raw count-zero position was preserved |
| `WM:gated` | Eligible forecast rejected; WM used |
| `SHADOW` | Selected proposal logged, but WM used |
| `WM:off` | Model disabled; both symbols coincide |
| `t7` / `o7` | Teammate7 / opponent7 relative to the selected observer |

Circle radius0.30m and square width0.60m are constant drawing sizes. They are
not uncertainty bounds, player sizes or kickable areas. When position is kept,
circle and square have the same center and no displacement line appears.
The standard monitor players show reality; this overlay adds no third truth
position. It shows player positions only; velocity/body and ball are stock.

All11 players produce their own debug data. There is no observer6-only filter.
Select any observer in SoccerWindow to see that agent's world.

## Run one game in separate terminals

Use a fresh RUN directory for each visual game. Ports below are6000/6001/6002;
live debug packets go to6032. Close a previous server on those ports first.
The shell variable RUN is set in each terminal independently.

### 1. Server

```bash
RUN=/home/nader/workspace/denoise/logs/exp5-debug/one-01
mkdir -p "$RUN/server" "$RUN/left" "$RUN/right" "$RUN/csv"
cd /home/nader/workspace/denoise/rcssserver-denoise/build
./rcssserver \
  include=../experiments/exp5/server-baseline.conf \
  server::auto_mode=false \
  server::port=6000 server::coach_port=6001 server::olcoach_port=6002 \
  server::game_log_dir="$RUN/server" server::text_log_dir="$RUN/server" \
  server::game_log_fixed=true server::game_log_fixed_name=exp5-debug \
  server::text_log_fixed=true server::text_log_fixed_name=exp5-debug
```

This retains Exp5 unquantized fields, finite views, no hearing/fading, fullstate
for reference and randomized synchronous command ordering. Start kickoff from
SoccerWindow after the teams connect.

### 2. SoccerWindow

```bash
/home/nader/workspace/soccerwindow2/src/qt/soccerwindow2 --host localhost --port 6000
```

### 3. Left team: apply, zero count, live and dlog output

```bash
RUN=/home/nader/workspace/denoise/logs/exp5-debug/one-01
mkdir -p "$RUN/left" "$RUN/csv"
cd /home/nader/workspace/denoise/helios-base-denoise/build/bin
./start-debug.sh -p 6000 -P 6002 -t lt \
  --fullstate reference --view-policy adaptive \
  --player-gap apply \
  --player-gap-model-dir /home/nader/workspace/denoise/logs/exp5-s1/live-model \
  --player-gap-pos-count zero \
  --player-gap-debug both --player-gap-debug-target all \
  --debug-server-port 6032 --debug-server-logging \
  --log-dir "$RUN/left" \
  --denoise-log-dir "$RUN/csv" --denoise-game-id one-01 \
  --no-denoise-raw --no-denoise-landmarks
```

### 4. Right team: original WM, with its own observer logs

```bash
RUN=/home/nader/workspace/denoise/logs/exp5-debug/one-01
mkdir -p "$RUN/right" "$RUN/csv"
cd /home/nader/workspace/denoise/helios-base-denoise/build/bin
./start-debug.sh -p 6000 -P 6002 -t rt \
  --fullstate reference --view-policy adaptive \
  --player-gap off --player-gap-debug both \
  --debug-server-port 6032 --debug-server-logging \
  --log-dir "$RUN/right" \
  --denoise-log-dir "$RUN/csv" --denoise-game-id one-01 \
  --no-denoise-raw --no-denoise-landmarks
```

For the original non-debug right team, use start.sh with the normal reference/
dataset flags and no model options. The right debug command above also keeps
the model off; its two position markers coincide.

## Where to open the logs

- **Colored dlog, left:** `logs/exp5-debug/one-01/left`, containing
  `lt-1.log` through `lt-11.log`. Right is `.../right`, with `rt-*.log`.
- In SoccerWindow's **Debug Message** window, use **Open debug log directory**,
  choose Left/Right correctly and the corresponding directory. Enable **WM**
  and debug-log objects. Select the observer number1–11. These files include
  `[GAP]` text, coordinates, raw/effective count, selection status and score.
- **Live agent debug:** select an observer and enable agent debug drawings;
  the incoming packets carry the same colored comparison. No directory is
  needed for incoming packets. `*.dcl` files also retain these packets.
- **Server replay:** the RCG is under `.../server/exp5-debug.rcg*`.
  Open it as the game log, then load the matching left/right debug directories.
- `.../csv` is the dataset; it is not the Debug Message log directory.

Use either incoming-agent geometry or dlog geometry for inspection. Enabling
both views together can paint the same comparison twice. dlog directory loading
indexes the file at load time in this SoccerWindow version; for later cycles
of a running game, reload the directory, or use the live agent view. Full replay
after the game is the simplest dlog workflow.

## Half-count comparison

The same normal/debug launch accepts --player-gap-pos-count half. Applied
predictions expose max(1,floor(original posCount/2)); dlog pc=raw>effective
shows the result. Real seen age is unchanged.
[Launch and verification](../exp5-s1/reports/half-count-policy.md).

## Optional focused view and output modes

Replace the left target with `--player-gap-debug-target opponent:10` to inspect
only opponent10, or `teammate:7`. The observer is still selectable among all11.

`--player-gap-debug off|live|dlog|both` defaults to off. `dlog` requires `--debug`
and the WM category enabled; `live` requires a debug-server connection or packet
logging. start-debug.sh supplies debug/connection options; it works with the
normal Release executable and does not require a separate Debug CMake build.

The banner shows `revision=cache-v2`, `cache=N/N` and `cache_ok=1`. A mismatched
count or cache_ok=0 is a structural problem, not neural uncertainty. Scores
are routing scores, not calibrated confidence percentages.

## Verification

The short all-observer and focused matches finalized all22 agents. Packet
geometry matched original/used CSV coordinates on both sides. All derived player
containers remained unique; full feature replay passed after the cache fix.
The synthetic packet test checks color serialization, old-drawing removal,
valid syntax and viewer-size limits. No full GUI interaction was automated.
Evidence is in `logs/exp5-s1/debug-checks` and the final report's JSON artifact.
