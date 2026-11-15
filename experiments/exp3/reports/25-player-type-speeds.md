# Exp3 Step 25 — Teammate-specific running speeds

Step 14 used the fastest advertised player speed as a common cap. We
replayed each receiver's incoming `change_player_type` messages. A
notification with a type ID identifies a teammate's current type. The
opponent messages omit the type ID, so opponent priors keep the shared cap.
The replay also keeps the shared cap until a type is known. It takes
type assignments from the preceding cycle and uses no fullstate types.

All advertised types in the 20 validation and test games had the same
hard `player_speed_max` of **1.05 m/cycle**. Applying these hard caps
individually produced identical position scores. We then used librcsc's
type-specific steady-state running speed, derived from dash power, dash
rate, maximum effort and decay and capped by the advertised speed.
This is a regularizer on the directional velocity prior. It does not
test a hard reachable-set intersection.

| Test cohort | Rows | Shared speed mean (cm) | Type-specific mean (cm) |
|---|---:|---:|---:|
| All paired teammates | 962,277 | 157.5277 | 157.5277 |
| Fresh explicit observations | 145,601 | 44.3399 | 44.3399 |
| 3–12 cycles since last explicit observation | 207,812 | 175.7418 | 175.7418 |

Validation kept the shared-speed directional method on fresh observations
and Step 13 at every other age. We keep the shared cap for this filter.
The logs do not identify individual opponent types.

[Paired metrics](individual-running-speeds-metrics.json)
