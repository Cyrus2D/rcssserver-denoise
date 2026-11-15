# Exp3 Step 22 — One-cycle teammate fusion

We replayed one additional same-team observer for each receiver. The sender
uses only raw visual observations with an explicit identity and its own
estimated pose. We project the position by one cycle using the motion-bin
midpoint when available. Validation chooses between Step 13, the sender
estimate and 25% or 50% blends for each receiver observation age.

| Paired test cohort | Rows | Step 13 mean (cm) | Selected fusion mean (cm) |
|---|---:|---:|---:|
| All | 2,000,986 | 165.60 | 158.60 |
| Sender estimate available | 251,150 | 123.01 | 67.21 |
| Receiver fresh | 62,352 | 50.54 | 40.75 |
| Receiver age 1–2 | 81,448 | 85.10 | 67.96 |
| Receiver age 3 or more | 107,350 | 193.86 | 82.02 |

The selected policy uses the 50% blend through age two and the sender
estimate at age three or more. Overall p90 fell from **327.94 to 314.34 cm**.
This experiment makes every eligible sender observation available one
cycle later and assumes full coordinate precision. It isolates the value
of a second viewpoint. The added packet experiment tests limited target
selection and coordinate quantization.

[Paired metrics](teammate-fusion-metrics.json)
