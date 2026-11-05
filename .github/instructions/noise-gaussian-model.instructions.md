---
applyTo: "src/{visualsenderplayer,player,serverparam,heteroplayer,random}*"
---

# Noise Model: Gaussian (`gaussian_see`, server v19)

## TL;DR

`(gaussian_see)` swaps the quantizing observation for `GaussianObservation`: distance gets additive zero-mean Gaussian noise whose sigma grows with the true distance and the object's distance from the focus point. Direction, body/head dir and lines are unaffected.
- **Open the full file when:** you denoise Gaussian `see` data, tune the four noise rates, or need the exact sigma / clamp / output precision.
- Formula: `sigma = d*r_d + f*r_f`, `nd = max(0, N(d, sigma))` (**no** multiplication by `d`; manual is wrong).
- Movable (ball, players): `r_d = dist_noise_rate = 0.0125`, `r_f = focus_dist_noise_rate = 0.0125`; landmarks (flags, goals): `0.00125` each.
- With focus = self (`f = d`): sigma = 2.5 % of d (movable) / 0.25 % (landmarks).
- `dist_chg = Quantize(true_dch * nd/d, 0.01)`; `dir_chg = Quantize(., dirQStep=0.1)` (same as quantize model); distance itself is **not rounded**.

## Overview

- Command: `Player::gaussian_see()` (`player.cpp` L2387-2391) sets `M_gaussian_see`, replies `(ok gaussian_see)`; grammar `player_command_parser.ypp` (rule ~L155, `gaussian_see_com` ~L376-378).
- Selection: every `sendVisual()` (all sender versions, async and synch) does `if (self().isGaussianSee()) M_noisy_observation = new GaussianObservation(self())` (`visualsenderplayer.cpp` L377-380). Sticky: never reset by `change_view`, `synch_see`, reconnect-free life. Works for all protocol versions (manual agrees, L413). Interaction with synch/version: none beyond which focus is used (v<18: focus = self).
- Class: `GaussianObservation` (`visualsenderplayer.cpp` L209-295). RNG `ndrand(mean, std)` = `std::normal_distribution` on `DefaultRNG` (`random.h` L98-104).

## Formula and parameters

`calcDistImpl` (L236-244):
```
sigma = d * dist_rate + f * focus_rate        // d: true observer->object, f: true focus->object
nd    = max(0, ndrand(d, sigma))
```
| Object | rates used | call site |
|---|---|---|
| Ball, players (incl. goalie, close-name path) | `distNoiseRate`, `focusDistNoiseRate` (L247-254) | `sendHighBall` L566, `sendHighPlayer` L646 |
| Flags, goals | `landDistNoiseRate`, `landFocusDistNoiseRate` (L256-263) | `sendHighFlag` L504 |
| Lines | none (quantized, `calcLineDist`) | `serializeHighLine` L815-823 |

- Rates are read from `observer().playerType()` (`HeteroPlayer`), but all types copy the server values: `setDefaultGaussianObservationParams()` (`heteroplayer.cpp` L266-274, called L127 and L244). Server defaults `0.0125/0.0125/0.00125/0.00125` (`serverparam.cpp` L382-385, assign L1478-1481, `addParam(..., 19)` L956-959). Only `server.conf` can change them; no `player.conf` key. Clients below protocol v19 never receive them in `server_param`.
- Focus: `f` uses `cachedFocusPoint()` = `self.pos()` for v<18, `self.focusPoint()` for v18+ (`.cpp` L1308-1311); `change_focus` moving the focus toward an object **lowers** nothing: sigma uses `f` (distance object<->focus), so keep the focus *near the object of interest* to reduce `f*r_f`; focus at self makes `f=d` (maximum for objects in front).

## Sigma tables (default rates)

| true d | movable sigma, focus=self | movable, focus on object (f=0) | landmark sigma, focus=self |
|---|---|---|---|
| 5 | 0.125 | 0.0625 | 0.0125 |
| 10 | 0.25 | 0.125 | 0.025 |
| 20 | 0.5 | 0.25 | 0.05 |
| 40 | 1.0 | 0.5 | 0.10 |
| 60 | 1.5 | 0.75 | 0.15 |
| 100 | 2.5 | 1.25 | 0.25 |

- Zero-noise region: only `d = f = 0` (sigma 0; `normal_distribution` with sigma 0 is unguarded - undefined by the standard, returns mean in practice). Otherwise noise is non-zero at every distance, including <= 3 m close-name objects.
- Clamp: only `max(0, .)` at the lower end; no upper clamp, no rounding to 0.1.
- Noise is i.i.d. per object per message (fresh draw each `see`), unbiased (before the clamp) -> averaging across cycles is valid for static objects (unlike the quantize model).

## Output precision and format

`nd` is a raw double printed with default `ostream` precision (6 significant digits, e.g. `23.4567`, `101.234`) (`serializer.h` L355). A distance with more than one decimal reveals Gaussian mode (quantize mode always prints multiples of 0.1). Direction stays an int.

## Velocity fields

`calcVel` (L265-294): `dch_raw = (v_obj - v_self) . e`; `dist_chg = Quantize(dch_raw * (nd/d), 0.01)` (L286-287); `dir_chg = Quantize((vt x e)/d*RAD2DEG, 0.1)` or 0 (L282-284); both 0 if `d == 0`. So `dist_chg` shares the same noisy ratio as `nd` (relative radial speed `dch/nd` is preserved: `dch_obs/nd = true dch/d` up to the 0.01 rounding) -> a client can recover `true dch/d` exactly-ish. Presence rules are the same ramps (visual-observation-detail).

## Hide ramps with Gaussian

`decide()` uses `nd` (noisy) so Gaussian noise shifts the unum/velocity thresholds randomly by about +-sigma. Team name remains always visible (v18+ default).

## Discrepancies vs manual (`sensor-models-vison-sensor-model.rst` L416-441)

1. Manual `d' = d * max(0, N(d, s))` - code has no leading `d` (code L243).
2. Manual table "Parameters in player_type" (L449-458) lists the rates as player_type; they are server.conf values copied to every type; manual's own text L439-440 agrees.
3. Manual says the model changes only distance; code additionally rescales `dist_chg` by `nd/d` and rounds it to 0.01 (not documented).

## See Also

- [noise-model](noise-model.instructions.md)
- [noise-quantize-model](noise-quantize-model.instructions.md)
- [visual-observation-detail](visual-observation-detail.instructions.md)
- [config-params](config-params.instructions.md)
- [player-protocol](player-protocol.instructions.md)
