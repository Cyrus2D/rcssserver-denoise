---
applyTo: "src/{visualsender,visualsenderplayer,serializerplayerstd,serializercommon,object,field,player,heteroplayer,stdtimer}*"
---

# Visual Observation Detail (`see` per object, per distance, per view mode)

## TL;DR

A `see` message is built by `VisualSenderPlayerV1::sendVisual()` as flags/goals -> ball -> players -> lines. What each object carries depends on the view cone, the (noisy) distance and three probability ramps. Direction is **never noisy** (only rounded to int degrees); distance noise is in [noise-quantize-model](noise-quantize-model.instructions.md) / [noise-gaussian-model](noise-gaussian-model.instructions.md).
- **Open the full file when:** you parse/denoise `see`, change view width/cone, name/unum/team hiding, or velocity (`dist_chg`/`dir_chg`) info.
- Cone: visible iff `|dir| < visible_angle/2` (strict) and true dist < max-observation (125.1 default = pitch diagonal, i.e. unlimited); else only objects within `visible_distance` (3.0) appear, as capitalized `(B) (P) (F) (G)` with noisy dist + dir.
- Default (v18+, high quality): player team name is **always** given; unum drops w.p. `(d-20)/20` over 20..40 m; ball/flag `dist_chg,dir_chg` drop with the same ramp.
- Hiding decisions use the **noisy** distance; each ramp draw is independent per object per message.
- Lines are not Gaussian and not focus-dependent; inside the pitch exactly one line is sent.

## Message format (`see`)

`(see <time> <obj>...)`, `<obj>` = `(<name> args...)`; names for v6+ are the short ones (`visualsenderplayer.h` L557-602). Serializer: `serializer.h` L343-397, `serializerplayerstdv8.cpp` L44-108, `serializerplayerstdv13.cpp` L45-137.

| Object (v18, high quality) | Full form | Reduced forms |
|---|---|---|
| Ball | `((b) d dir dch dirch)` | no vel: `((b) d dir)`; outside cone <=3 m: `((B) d dir)` |
| Flag / goal | `((f c) d dir dch dirch)` / `((g r) ...)` | no chg: `((f c) d dir)`; near: `((F) d dir)`, `((G) d dir)` |
| Player, all known | `((p "Team" N [goalie]) d dir dch dirch body head [point] [t\|k])` | unum hidden: `((p "Team") d dir [point] [t\|k])`; team hidden: `((p) d dir)`; near: `((P) d dir)` |
| Line | `((l b) d dir)` | none |

- Names: `PLAYER_NAME_FORMAT_SHORT "(p \"%s\" %d%s)"`, far `(p "%s")`, too far `(p)`, goalie suffix `" goalie"` only in the **full** name (`param.h` L214-225; built in `Player::init` `player.cpp` L314-338). Long names (v<6): `(player T N)`, `(player T)`, `(player)`, no goalie marker.
- Opponent names use `Team::fixedName()` (`team.cpp` L258-272, params `fixed_teamname_l/r`), own team uses the real name (`visualsenderplayer.h` L442-463).
- `d` = noisy distance, `dir` = int degrees; `dch` = dist_chg, `dirch` = dir_chg (deg/cycle); `body`/`head` ints, V4+/V5+ (`visualsenderplayer.h` L497-540). `t` tackling, `k` kicked-last-cycle (v13+; `t` wins). V8-V12 have `t` only.
- Direction `dir` = `normalize(angleFromBody(obj) - neckCommitted)` (`visualsenderplayer.h` L331-335, `player.h` L454-458; **committed** angles, true positions at send time). Int conversion: truncation for v<7 (`Rad2IDeg`), round-to-nearest for v>=7 (`visualsenderplayer.h` L620-634, `utility.h` L77-88) -> error <= 0.5 deg.
- `body = round(playerBody - selfBody - selfNeck)`, `head = round(playerNeck + playerBody - selfBody - selfNeck)`; both exact apart from int rounding, only present in the "all known" player form.
- `point` (V8+, only if the observed player is pointing, `arm().isPointing()`): the only **angle noise** in `see`: `sigma = (d_true/team_too_far_length)^4 * (pi - 0.0218) + 0.0218` rad (min 1.25 deg), then `Normal(dir, sigma)`, truncated/rounded to int (`visualsenderplayer.cpp` L1102-1145). With the fork's default `team_too_far_length`=125.1: sigma ~1.3 deg @10 m, ~1.8 deg @30 m, ~10.7 deg @60 m. Not a distance, no distance info about the pointed spot.
- Floats printed with default `ostream` precision (6 significant digits).

## Visibility rules

Per object (high quality; `sendHigh*`, `visualsenderplayer.cpp` L500-690):
1. `in_cone = |ang| < visibleAngle()*0.5` and `d_true < max_obs_length` (ball `ballMaxObservationLength`, flags `landMaxObservationLength`, players `playerMaxObservationLength`; all default `sqrt(105^2+68^2)=125.1`, `heteroplayer.cpp` L248-263) -> full info.
2. else if `d_true <= visible_distance` (3.0, `serverparam.cpp` L195, param v7) -> capitalized close name + **noisy** distance + dir only.
3. else not sent. Self is skipped; players need `isEnabled()` and `objectVersion() <= version`.
- View cone angle (`player.cpp` L1795-1832, L2358-2373; `visible_angle` default 90, `serverparam.cpp` L194):

| width | async (v<18) angle / period | synch (v18+) angle / period |
|---|---|---|
| narrow | 45 deg / 75 ms (interval 2) | 60 deg (=2/3*90) / every step |
| normal | 90 deg / 150 ms (4) | 120 deg (=4/3*90) / every 2 steps |
| wide | 180 deg / 300 ms (8) | 180 deg / every 3 steps |

- `view_width` does **not** change noise: the per-width noise terms are initialised to 1.0 (`player.cpp` L239-241; the 0.5/0.75 values at L383-385 are commented out, "disabled in v18.1.0").
- `view_quality low` (async, v<18 only): `sendLow*` (`L479-639`) emit **only `(name dir)`**, no distance/chg/body info; interval halved. Players still get the team/unum ramps (names only). Lines: `(name dir)`.
- Flags/goals come from the landmark XML (`landmark_file`, `stadium.cpp` L236; `Field::landmarks()`), 55 incl. goals per manual; `(f c)`, `(f l t)`, `(f p l b)`, `(f g l t)`, `(f t l 20)` ... Flag distance is unlimited in practice (125.1).
- Lines (`sendLines` L430-476, `sendLine` L693-804): inside pitch -> exactly one line (first of l,r,t,b that qualifies); outside -> up to two. A line qualifies if the view axis hits it (`|sight_to_normal| < 90 deg`, intersection inside the segment). `dir = Rad2IDeg(ang -+ pi/2)` (`visualsenderplayer.h` L351-357); `d` = distance along the **view axis** `|p2l / cos(ang)|` quantized (`L382-390`). No cone/width test, no Gaussian.

## Distance-dependent information (the ramps)

`decide(p)` returns true (= **hide**) if `p>=1`, false if `p<=0`, else Bernoulli(p) (`visualsenderplayer.h` L399-404). All use the noisy distance `nd`.

| Quantity hidden | p(hide) | v18+ params (`heteroplayer.cpp` L252-262) | v<18 senders (`player.cpp` L168-171) |
|---|---|---|---|
| Ball dist_chg/dir_chg | `(nd-ballVelFar)/(ballVelTooFar-ballVelFar)` (`h` L732-736) | 20 / 40 | unum_far/too_far 20 / 40 (`h` L272-275) |
| Flag dist_chg/dir_chg | `(nd-landVelFar)/(landVelTooFar-landVelFar)` (`h` L725-729) | 20 / 40 | 20 / 40 |
| Player unum (+ dch, dirch, body, head) | `(nd-unumFar)/(unumTooFar-unumFar)` (`h` L745-750) | 20 / 40 | 20 / 40 |
| Player team name | `(nd-teamFar)/(teamTooFar-teamFar)` (`h` L738-743) | 125.1 / 125.1 -> division by 0 gives -inf -> **never hidden** (fork change 2022-10-13, was 40/60) | 40 / 60 |

- Order for players: team test first (hidden -> `(p)` form, no t/k flag), then unum test (hidden -> `(p "T")` + point + t/k), else full (`L642-690`).
- Close/outside-cone objects never carry velocity or names.
- A client should treat missing `dch/dirch` on ball/flags beyond 20 m as "unknown", not zero.

## Summary table: object x distance band x info (v18+, defaults, high quality, in cone)

| Object | d <= 3 (any angle) | 3 < d <= 20 | 20 < d < 40 | d >= 40 (to 125) | Noise (Quantize / Gaussian) |
|---|---|---|---|---|---|
| Ball | full (even out of cone: `(B) d dir`) | d dir dch dirch | dch/dirch dropped w.p. (d-20)/20 | d dir | dist: yes (step 0.1 / rate 0.0125+0.0125); dir: int only; dch: see noise files |
| Teammate/opp player | full in cone; `(P) d dir` outside | team, N, d, dir, dch, dirch, body, head, [point], [t/k] | N dropped w.p. (d-20)/20 -> `(p "T")` | `(p "T") d dir` | same as ball; body/head int only; point sigma above |
| Goalie | like player, `goalie` token only in full form | idem | idem | idem | idem |
| Flag / goal | full in cone; `(F)/(G) d dir` outside | d dir dch dirch | chg dropped w.p. (d-20)/20 | d dir | step 0.01 / rate 0.00125+0.00125 |
| Line | n/a | `(l x) d dir` | idem | idem | quantized step 0.01, no focus term |

View width changes only cone angle and period (table above), not the columns. `(p)` (team hidden) only occurs for v<18 clients beyond 40 m.

## Focus point (v18+)

- `Player::change_focus(dist, dir)` (`player.cpp` L1292-1302): `focus_dist` clamped to [0, 40]; `focus_dir` clamped to `+-visible_angle/2` (relative to neck); also re-clamped on `change_view` (L1922-1924). `focusPoint() = pos + polar(focus_dist, body+neck+focus_dir)` (L2601-2605).
- `VisualSenderPlayerV18::updateCache()` caches it per message (`visualsenderplayer.cpp` L1308-1311); V1-V13 cache `pos()` (`.h` L266-270) so `f == d` for them. It changes **only noise** (distance), never visibility.
- Body sense reports `(focus_point dist dir_deg)`.

## Discrepancies code vs manual (`sensor-models-vison-sensor-model.rst`)

1. Manual L151-164 width factors inverted: code narrow = more frequent (interval 2), wide = rarer (8). Param is `send_step`, not `sense_step`.
2. Manual L391 `Quantize = ceiling(V/Q)*Q`; code is `rint(V/Q)*Q` (round to nearest) (`utility.h` L98-107).
3. Manual quantized-distance formula and Gaussian formula differ from code (detail in the two noise files).
4. Manual says BodyDir/HeadDir/dist_chg are given for any player; code drops them (and the unum) together when the unum ramp hides the number.
5. Manual does not mention `view_quality low` = direction-only output, nor the point-direction noise.
6. Manual flag/line note: "(l ..) distance is to where the view center crosses the line" matches code.

## Not found / open

- No per-object velocity noise beyond quantization; no persistent (bias) noise; no occlusion.
- `quantize_step_dir`/team qstep params exist only under undefined `NEW_QSTEP` (`serverparam.cpp` L704-712).

## See Also

- [observation-sensors](observation-sensors.instructions.md)
- [noise-model](noise-model.instructions.md)
- [noise-quantize-model](noise-quantize-model.instructions.md)
- [noise-gaussian-model](noise-gaussian-model.instructions.md)
- [communication-hearing](communication-hearing.instructions.md)
- [player-protocol](player-protocol.instructions.md)
