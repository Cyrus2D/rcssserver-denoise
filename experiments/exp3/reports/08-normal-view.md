# Exp3 Step 08 — Normal-view replay

The selected model could run on **5,252 of 562,375** fresh,
explicitly identified player observations (**0.93%**) in a seeded
sample of 20 standard Exp1 games. This is **0.13%** of all
observer-cycle-target slots. The test uses one observer per side per game,
with no parameter updates.

| Check | Count |
|---|---:|
| Observer PlayOn decisions | 192,638 |
| Observer decisions with a see update | 110,028 |
| Fresh target observations matched by uniform number | 562,375 |
| Fresh target observations with eight consecutive valid cycles | 5,252 |

The replay resets a target's history when a cycle or identified observation
is missing. It evaluates position only on the 5,252 eligible
endpoints from 20 games with eligible data.

| Method on eligible observations | Relative mean (cm) | Relative p90 (cm) | Global mean (cm) | Global p90 (cm) |
|---|---:|---:|---:|---:|
| Fresh librcsc | 33.14 | 70.29 | 34.67 | 70.90 |
| Step 02 GRU | 25.39 | 55.03 | 26.57 | 55.64 |
| Selected context GRU | 24.89 | 54.05 | 26.10 | 54.00 |

These endpoints are an unusually continuous subset of normal play. The
model's low coverage is the practical blocker for a live A/B test. The
next offline experiment should represent elapsed cycles explicitly and
test variable observation gaps and identity loss, including the fallback
on decisions where the model has insufficient history.

[Step 07 · Selected model](07-unum-embedding.md) · [Metrics](context-metrics.json)
