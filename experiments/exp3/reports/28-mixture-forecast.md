# Exp3 Step 28 — Multiple one-cycle movement hypotheses

The Step 27 forecast head predicts a single diagonal Gaussian. Step 28
replaces it with three Gaussian components and learned weights. Both
forecast models use the same recurrent width, causal inputs, game split,
12 epochs and 49,152 sampled windows per epoch. The three components may
capture different displacements. They carry no hand-assigned dash or turn
labels.

Forecast position error uses the mixture mean as its point prediction.
Displacement negative log likelihood scores the full distribution, and
marginal interval coverage checks whether an 80% interval contains the
true displacement in each of the observer-frame axes. The post-sighting
comparison uses the full mixture density to weight position candidates.

| Test metric | One component | Three components |
|---|---:|---:|
| Pre-sighting position mean, cm | 61.49 | 62.08 |
| Displacement negative log likelihood | −0.562 | −0.954 |
| Marginal 80% interval coverage | 81.4% / 82.4% | 81.7% / 83.0% |
| Post-sighting relative mean, cm | 48.65 | 48.64 |
| Post-sighting relative p90, cm | 110.52 | 110.67 |

Three components assigned more density to the observed movements.
The mixture mean had higher pre-sighting position error. The best
positive weight on validation for both forecasts was 0.25. With the
mixture, the corrected position still trailed **48.16 / 109.83 cm**
from Step 26 alone. Keep the movement models as research candidates
for gaps and use the Step 26 position-only model in the first 360°
live test. This result is specific to the tested reweighting rule and
the case where a new quantized observation arrives every cycle.

[Step 27](27-one-cycle-forecast.md) · [Paired metrics](forecast-360-metrics.json)
