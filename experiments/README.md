# Denoise experiments

Measure RoboCup 2D observation errors and test ways to reduce them.

New to the code? Start with the [research guide](RESEARCH.md) for server
changes, the planned experiment matrix, and the measurement rules.

| Experiment | What we ran | Results |
|---|---|---|
| [Exp1 · Baseline](exp1/README.md) | 100 games with standard quantized observations | [Full report](exp1/reports/report.html) |
| [Exp2 · Self localization](exp2/README.md) | Train a recurrent position tracker, then run 100 games with it active on the left | [Full report](exp2/reports/report.html) · [Exp1 comparison](exp2/reports/comparison.html) |

Raw logs, generated training data, and model weights live in the workspace's
`logs/` directory outside this repository.
