# Denoise experiments

Measure RoboCup 2D observation errors and test ways to reduce them.

Completed Exp3/Exp4 work is summarized in [Exp3–Exp4](exp3-exp4/README.md):
methods, paired noise reductions, match outcomes and the handoff to visual-gap prediction.

New to the code? Start with the [research guide](RESEARCH.md) for server
changes, the planned experiment matrix, and the measurement rules.

| Experiment | What we ran | Results |
|---|---|---|
| [Exp1 · Baseline](exp1/README.md) | 100 games with standard quantized observations | [Full report](exp1/reports/report.html) |
| [Exp2 · Self localization](exp2/README.md) | Train a recurrent position tracker, then run 100 games with it active on the left | [Full report](exp2/reports/report.html) · [Exp1 comparison](exp2/reports/comparison.html) |
| [Exp3 · Player quantization](exp3/README.md) | 200 games with 360° views, uniform-number fading disabled and no teammate payloads | [Exp1 comparison](exp3/reports/report.html) · [Live results](exp3/reports/31-live-results.md) |
| [Exp4 · Player and ball state](exp4/README.md) | Learn player velocity and body direction plus ball position, velocity and next-cycle motion | [Offline results](exp4/README.md) · [155-game live audit](exp4/reports/03-live-f3-audit.html) · [F4 repair](exp4/reports/04-live-f4-fix.html) · [60-game audit](exp4/reports/05-live-f4-60-games.html) · [Random command priority](exp4/reports/06-server-command-order.html) · [Closeout: ball stability](exp4/reports/09-closeout-ball-stability.html) |

Raw logs, generated training data, and model weights live in the workspace's
`logs/` directory outside this repository.
