---
applyTo: "src/{player,object,heteroplayer,serverparam,visualsenderplayer,serializer}*"
---

# Noise Model Hub (Visual + Actuator Noise)

## TL;DR

rcssserver adds noise in the **visual sensor** (quantization by default, Gaussian after `(gaussian_see)`) and in **actuators/physics**. Details of each visual model live in sibling files.
- Default visual model = `QuantizeObservation`; `(gaussian_see)` swaps in `GaussianObservation` (sticky, all versions); server param `see_noise_free` selects `ExactObservation` (exact distance) first, so it wins over Gaussian: `visualsenderplayer.cpp:414-421`, `player.cpp` `gaussian_see`.
- Formulas: [noise-quantize-model](noise-quantize-model.instructions.md), [noise-gaussian-model](noise-gaussian-model.instructions.md).
- Objects within `visible_distance` (3.0) are NOT noise free: they are sent as capitalized `(B)/(P)/(F)/(G)` with noisy dist/dir (high quality); low quality sends direction only.
- Actuator noise: `ball_rand` 0.05, `player_rand` 0.1, `kick_rand` 0.1, `tackle_rand_factor` 2.0, `wind_rand` 0.
- **Open the full file when:** you need the param table, actuator noise formulas, or model selection logic.

## Model selection

- `VisualSenderPlayer` ctor installs `QuantizeObservation`; `VisualSenderPlayerV1::sendVisual` replaces it (`visualsenderplayer.cpp:414-421`): `ExactObservation` if `ServerParam::seeNoiseFree()`, else `GaussianObservation` if `isGaussianSee()`. All sender versions V1..V18 inherit it.
- `ExactObservation` (derived from `QuantizeObservation`, `:299-330`): `calcDist`/`calcDistLandmark` return the true distance; direction stays int degrees; `dist_chg`/`dir_chg` keep the quantize rounding (computed with the true distance as `nd`); lines get the exact `|p2l/cos(ang)|` in `serializeHighLine`. Details: [noise-quantize-model](noise-quantize-model.instructions.md).
- v18+ is forced to synch_see (`player.cpp:376-387`), so low quality is reachable only by v<18 clients (`player.cpp:1846-1852`).
- Lines never use Gaussian (`calcLineDist`, `visualsenderplayer.h:382-390`); they are exact only with `see_noise_free`.
- Hiding ramps (team/unum/velocity) apply with either model, on the noisy distance.
- `NEW_QSTEP` per-team steps are never compiled in.

## Visual noise parameters

| Name | Default | Applies to | Source |
|---|---|---|---|
| quantize_step | 0.1 | movable objects dist (quantize model) | `serverparam.cpp:206` |
| quantize_step_l | 0.01 | flags/goals dist (quantize model) | `serverparam.cpp:207` |
| dist_noise_rate | 0.0125 | Gaussian, movable, `d` term | `serverparam.cpp:382`, addParam v19 :956 |
| focus_dist_noise_rate | 0.0125 | Gaussian, movable, focus-dist term | `:383` |
| land_dist_noise_rate | 0.00125 | Gaussian, landmarks | `:384` |
| land_focus_dist_noise_rate | 0.00125 | Gaussian, landmarks focus term | `:385` |
| visible_angle / visible_distance | 90 / 3.0 | view cone, near-object rule | `:194-195` |
| full_view | false | cone = 360 deg and a `see` every cycle for synch_see players (see [visual-observation-detail](visual-observation-detail.instructions.md)) | `serverparam.cpp` addParam v19 |
| see_no_fade_unum / see_no_fade_all | false / false | disable the team/unum (and with `_all` also velocity/body/head) distance fade of far objects, v18+ high quality | same |
| see_noise_free | false | `ExactObservation`: exact distance (wins over Gaussian) | same |
| unum far / too-far | 20 / 40 | player number ramp | `heteroplayer.cpp:248-263` |
| team far / too-far | 125.1 / 125.1 (fork) | team ramp (effectively never hidden) | same |

Gaussian rates are copied into every HeteroPlayer (`heteroplayer.cpp:266-274`); no player.conf key.

## Actuator noise (hidden noise the client must absorb)

| Noise | Location | Formula |
|---|---|---|
| Velocity noise (all movables) | `object.cpp:446-454`, `_inc` :499 | radius U(0, randp*|vel|), angle U(-pi,pi) |
| Wind | `object.cpp:456-473`, :500 | zero if `wind_rand`<EPS; default 0 (`serverparam.cpp:224`) |
| Ball randp | `stadium.cpp:526-531` | `ball_rand` 0.05 (`serverparam.cpp:153`) |
| Player randp | `player.cpp:350-355` | `player_rand` 0.1 (`:162`); x `prand_factor` if `team_actuator_noise` |
| Kick | `player.cpp:1385-1401`, long kick :1498-1515 | `max_rand = kick_rand*(power/maxPower)*(pos_rate+speed_rate)`, rates in [0.5,1]; `kick_rand` 0.1 (`:181`), x `kick_rand_factor` if team noise |
| Tackle | `player.cpp:2233-2248` | `kick_rand*tackle_rand_factor(2.0)*power_rate*(pos_rate+speed_rate)` (`serverparam.cpp:361`) |
| Dash / turn angle | dash `player.cpp:1248-1250`, turn :1270-1273 | factor `(1+U(-randp,randp))`; left-leg test at :1245-1246 tests left twice (typo) |
| kick_rand per type | `heteroplayer.cpp:94-100` | `kick_rand + delta*kick_rand_delta_factor` |

Not found: persistent bias noise, occlusion, per-object velocity noise beyond quantization/Gaussian dist_chg.

## Manual discrepancies

- Gaussian: manual writes `d' = d*max(0,N(d,s))`; code has no leading `d` (`visualsenderplayer.cpp:243`).
- Quantize: manual says ceiling, code uses `rint`; manual omits the `d-qd` term and /2 average.
- Manual's "max noise ~10 at 100 m" is ~5.1 m in code.
- Manual's player_type table for Gaussian rates is misleading (server.conf values only).
- Manual's view-frequency width factors are inverted; param is `send_step`.

## Notes

- Random: `ndrand`/`drand` in `random.h`; `decide()` uses Bernoulli.
- No automated tests cover noise (`serverparamtest.cpp`/`playerparamtest.cpp` not built).

## See Also

- [noise-quantize-model](noise-quantize-model.instructions.md)
- [noise-gaussian-model](noise-gaussian-model.instructions.md)
- [visual-observation-detail](visual-observation-detail.instructions.md)
- [observation-sensors](observation-sensors.instructions.md)
- [communication-hearing](communication-hearing.instructions.md)
- [config-params](config-params.instructions.md)
- [actions-models](actions-models.instructions.md)
