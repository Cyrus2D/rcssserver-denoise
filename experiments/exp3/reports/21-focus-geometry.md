# Exp3 Step 21 — Active focus geometry diagnostic

We sampled true distances from fresh, explicitly identified observations
in the normal-view test games. We simulated two readings of the same
stationary geometry. The first has focus distance zero. The second points
along the target bearing with a fixed offset. The range estimator only
sees the two quantized outputs and intersects their compatible candidate
ranges. Truth provides simulator input and evaluation.

| Second focus offset | Mean radial error (cm) | Mean surviving range width (cm) |
|---:|---:|---:|
| 0 m | 48.54 | 193.88 |
| 2 m | 31.01 | 123.22 |
| 5 m | 26.93 | 106.83 |
| 10 m | 20.14 | 79.50 |
| 20 m | 11.77 | 43.02 |

Validation selected 20 m from the five tested offsets. The test has
282,032 paired distances. The comparison is a fixed-geometry radial
diagnostic. It assumes that the target bearing is known and that the
second reading can be taken at the same relative geometry. A live policy
would have to choose a target, aim from estimated bearing, and account
for movement and view delay. The next active-sensing test requires
controlled games that vary the focus policy.

[Radial metrics](focus-geometry-metrics.json)
