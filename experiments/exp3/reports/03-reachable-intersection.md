# Exp3 Step 03 — Reachable intersection

The position-only intersection changed validation mean relative error from
**71.480 to 71.459 cm**.
The reachability constraint reduced the selected ray's range interval in
**9.01%** of consecutive observations across the sequence dataset.

## Method

1. Convert each fresh range and bearing interval into a thin sector in the
   common field coordinate frame using the observer's estimated pose.
2. Store support bounds along the field axes and two diagonals. Expand the
   previous bounds by the largest player speed limit from the server's
   player-type list plus a pose-error allowance.
3. Intersect the current sector's sampled rays with those bounds. Keep all
   surviving rays for propagation. For the current position estimate,
   select the surviving point closest to the bounded blend.
4. If the intersection is empty, use the current sector and start a new
   envelope. Also reset after a missing observation or cycle gap.

This approximates a reachable set with four strips and six sampled rays.
It uses estimated observer poses. The pose allowance and sampling can
affect whether the overlap contains the true position.

## Paired validation

The table uses **162,099** endpoints from the same 20 games and
the same eight-observation history requirement for every method.

| Method | Relative mean | Relative p90 |
|---|---:|---:|
| Fresh librcsc | 76.557 cm | 161.932 cm |
| Bounded blend | 71.480 cm | 158.979 cm |
| Intersection | 71.459 cm | 158.979 cm |
| Step 02 GRU | 53.432 cm | 121.578 cm |

Across **40,797,918** consecutive observations in the sequence
dataset, **3,674,656** narrowed by at least 1 cm and
**481** used the empty-overlap fallback (0.0012%).
The speed bound alone rarely makes the range narrow enough to materially
change the point estimate. The next step tests whether a directional
motion prediction helps choose a point within the constrained region.

[Step 04 · Learned motion](04-learned-motion.md)
