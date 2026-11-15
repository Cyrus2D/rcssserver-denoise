# Exp3 Step 20 — Uniform number and observable role context

Starting with Step 19, we compared a four-value uniform-number embedding,
no embedding, and a four-value role proxy using the target's last
estimated field x/y, estimated distance to the ball and ball availability.
The role proxy uses the previous observation and needs no fullstate or
hand-assigned position label.

| Paired 360° validation, 162,099 rows | Mean relative error (cm) |
|---|---:|
| Step 19 with uniform embedding | 51.05 |
| Remove uniform embedding | 51.48 |
| Use the role proxy in place of identity | 51.66 |

Removing identity raised mean error by **0.42 cm**. Adding this role proxy
to the no-identity variant raised it by a further **0.19 cm**. We kept the
uniform embedding for the selected test checkpoint. Both sides in this
dataset use the same Helios policy. Cross-opponent generalization requires
games against a different team. Defer interpreting the embedding as a
team-independent role feature until that comparison is available.

[Validation](role-validation.json)
