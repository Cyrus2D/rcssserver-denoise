# Exp3 Step 19 — Masked nearby-player context

The Step 07 context model uses distance-ordered slots for two neighboring
players on each team. We kept the same two players per team, history,
ball inputs and uniform-number embedding. A shared encoder now processes
each neighbor. It masks unavailable players and pools within each team.
The resulting feature vector keeps the original width. A permutation
check confirmed that swapping neighbors within a team leaves it unchanged.

| Paired 360° cohort | Rows | Step 07 ordered slots (cm) | Masked set (cm) |
|---|---:|---:|---:|
| Validation | 162,099 | 51.21 | 51.05 |
| Test | 3,902,451 | 48.65 | 48.48 |

The masked set won on validation. Held-out p90 fell from 109.70 to
109.08 cm. This is a small gain on continuous 360° observations. Its
normal-view coverage has yet to be measured.

[Validation](masked-set-validation.json) · [Test](masked-set-test.json)
