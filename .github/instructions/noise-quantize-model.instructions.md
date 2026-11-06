---
applyTo: "src/{visualsenderplayer,player,serverparam,utility,heteroplayer}*"
---

# Noise Model: Quantization (default visual model)

## TL;DR

The default `see` noise model is **deterministic**: distances are rounded on a log grid, then to 0.1; no random numbers are drawn (only the hide/show ramps are random). It is `QuantizeObservation` in `src/visualsenderplayer.cpp`, active for every player until the client sends `(gaussian_see)`, unless the server param `see_noise_free` is on (see Noise-free variant).
- **Open the full file when:** you invert/denoise quantized distances, `dist_chg`/`dir_chg`, flags, lines, or need error bounds.
- Distance (ball, players, flags, goals): `qd = exp(Quantize(ln(d+EPS), q))`, `qf` same for focus distance `f`, `r = Quantize(max(0, d - ((f-qf)+(d-qd))/2), 0.1)`; with `focus = self` (always for v<18, default for v18+) `f=d` so `r = Quantize(qd, 0.1)`.
- `q = quantize_step = 0.1` (movable), `quantize_step_l = 0.01` (flags/goals/lines). Relative error <= ~5.1 % (movable) / ~0.5 % (landmarks) plus 0.05 rounding.
- `dir` has no distance-dependent noise (int degrees); `dir_chg` step 0.1 deg; `dist_chg = nd * Quantize(dch/d, 0.02)`.
- `Quantize(v,q) = rint(v/q)*q` (round to nearest; manual says ceiling - wrong).

## Overview

`Quantize` and constants: `utility.h` L53 (`EPS=1e-10`), L98-107 (returns `v` unchanged if `q < EPS`). Step sources: `ServerParam::quantizeStep()=0.1`, `landmarkQuantizeStep()=0.01` (`serverparam.cpp` L206-207 defaults, L1316-1317 assign, L702-703 `addParam(..., 7)`), multiplied by the view-angle noise term. Per-player accessors `Player::distQStep()/landDistQStep()/dirQStep()` (`player.h` L268-306): `dirQStep = 0.1 * term`; **term = 1.0 for all view widths** (`player.cpp` L239-241; adaptive 0.5/0.75 commented out L383-385). So the model is independent of view width, view quality, protocol version and team.

## Distance formula (movable objects and landmarks)

`QuantizeObservation::calcDistImpl` (`visualsenderplayer.cpp` L143-153), `calcDist` uses `distQStep` (L156-160), `calcDistLandmark` uses `landDistQStep` (L162-166):

```
qd = exp(Quantize(ln(d + EPS), q));   qf = exp(Quantize(ln(f + EPS), q))
r  = Quantize(max(0, d - ((f - qf) + (d - qd)) / 2), 0.1)
```
- `d` = true distance observer->object; `f` = true distance focus point->object (`obj.pos().distance(cachedFocusPoint())`, `visualsenderplayer.cpp` L827-838). The focus point is `self.pos()` unless v18+ `change_focus` moved it.
- Applied to: ball (L566), players (L646), flags and goals (`calcNoisyDistLandmark`, L504), including the capitalized close-name path (<=3 m outside cone). Not applied to lines.
- Possible log-grid values: `qd = e^(0.1 k) = 1.10517^k` (movable), `e^(0.01 k)` (landmarks). Printed value `r` is a multiple of 0.1 (6-digit float print, e.g. `23.4`).
- Manual formula (L374-379) is `d'' = Quantize(max(0, d - (f - f')), 0.1)` - it **omits the `d - qd` term and the /2 average**; with `f=d` the manual would give `Quantize(qf,0.1)` as well, but with a displaced focus they differ. Code wins.

### Error bounds (focus = self, so `r = Quantize(qd, 0.1)`), numerically scanned

| true d | q=0.1 error range | q=0.01 (landmark) error range |
|---|---|---|
| 1 | -0.05 .. +0.05 | -0.05 .. +0.05 |
| 5 | -0.21 .. +0.29 | -0.05 .. +0.07 |
| 10 | -0.49 .. +0.51 | -0.07 .. +0.08 |
| 20 | -1.0 .. +1.1 | -0.15 .. +0.15 |
| 40 | -2.1 .. +2.2 | -0.26 .. +0.25 |
| 60 | -3.1 .. +3.3 | -0.35 .. +0.36 |
| 100 | -5.1 .. +5.3 | -0.57 .. +0.59 |

Rule of thumb: `|err| <= d*(e^0.05 - 1) + 0.05 = 0.0513 d + 0.05` (movable), `0.00501 d + 0.05` (landmark). The manual's "max noise ~10 at 100 m" (L395) overstates by ~2x. With displaced focus the error is the mean of the two log-rounding errors (`d` and `f`), e.g. d=20: f=0 -> -0.5..+0.5, f=20 -> -0.44..+0.57, f=40 -> -0.28..+0.73. The mapping is a function of `(d, f)` only: the same geometry yields the same output every cycle (no jitter), so denoising across cycles gains nothing unless geometry changes; combine many landmarks / objects instead.

## Lines

`serializeHighLine` (`visualsenderplayer.cpp` L815-823) -> `calcLineDist` (`.h` L382-390): `Quantize(exp(Quantize(ln(|p2l/cos(ang)| + EPS), landDistQStep)), 0.1)`; no focus term; unchanged in Gaussian mode. Matches manual L403 (step `quantize_step_l`).

## Velocity fields (`dist_chg`, `dir_chg`)

`QuantizeObservation::calcVel` (L168-197), `vtmp = v_obj - v_self` (true velocities), `e = (p_obj - p_self)/d`:
- `dch_raw = vtmp . e`; `dist_chg = nd * Quantize(dch_raw / d, 0.02)` (L190) -> absolute step `0.02*nd` (e.g. 0.2 at nd=10), uses **noisy** `nd`, true `d`.
- `dirch_raw = (vtmp.y*e.x - vtmp.x*e.y)/d * RAD2DEG` (deg/cycle); `dir_chg = 0` if exactly 0 else `Quantize(dirch_raw, dirQStep()=0.1)` (L183-189).
- `d == 0` -> both 0 (L192-196). Flags: `v_obj = 0` (`sendHighFlag` L519) so values only reflect own motion.
- Presence is probabilistic (ramps in [visual-observation-detail](visual-observation-detail.instructions.md)).

## Noise-free variant (`see_noise_free`, v19, default false)

`ExactObservation` (`visualsenderplayer.cpp` L299-330) derives from `QuantizeObservation` and overrides only `calcDist`/`calcDistLandmark` to return the true distance. It is chosen in `sendVisual()` before the Gaussian check, so it also overrides `(gaussian_see)`. `dir` stays an int, `dist_chg = nd*Quantize(dch/d, 0.02)` and `dir_chg` rounding are unchanged (`nd` = true distance), and `serializeHighLine` sends the exact `|p2l/cos(ang)|` (no log-grid, no 0.1 rounding). Printed distance is still a 6-significant-digit float. Applies to the high-quality (v18+) senders.

## Direction, body, head

`dir`, `body`, `head` carry **no** noise: `Rad2IDegRound` (v>=7; <=0.5 deg) or truncation (v<7) (`visualsenderplayer.h` L620-634). Only `point` direction has Gaussian angle noise (see visual-observation-detail).

## Disabled/legacy

- `NEW_QSTEP` per-team steps (`quantize_step_dir`, `quantize_step_dist_team_*`, `serverparam.cpp` L704-712, `player.h` L277-279) never compiled.
- View-width noise terms (0.5/0.75/1.0 in v18.0.0) disabled since v18.1.0.
- `calcQuantDist` (`.h` L376-380) exists but is unused by the senders.

## Denoise hints (client side)

- Output `r` is deterministic; interval inversion: `d in {x : Quantize(exp(Quantize(ln x, 0.1)), 0.1) = r}` (for focus=self).
- For v18+ with focus moved off the object the offset contributes a (d,f)-dependent shift of up to ~ +-2.5 % of `f`; keep focus at self (default) when you want a pure function of `d`.

## See Also

- [noise-model](noise-model.instructions.md) - hub, actuator noise, parameter table
- [noise-gaussian-model](noise-gaussian-model.instructions.md)
- [visual-observation-detail](visual-observation-detail.instructions.md)
- [observation-sensors](observation-sensors.instructions.md)
- [config-params](config-params.instructions.md)
