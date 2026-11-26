# Exp4 · Player and ball state

[All experiments](../README.md) · [Closeout report](reports/09-closeout-ball-stability.html) · [Exp3–Exp4 archive](../exp3-exp4/README.md)

## Results at a glance

The repaired F4 stack reduced mean position and motion errors on paired
PlayOn samples from its first 60 games. The ball could still follow the wrong
direction after a kick when the visual motion fields were missing.

| Field | Live stock → applied |
|---|---:|
| Player velocity | **10.43 → 7.57 cm/cycle** |
| Player body direction | **25.17° → 16.98°** |
| Ball position | **75.17 → 69.92 cm** |
| Ball velocity | **32.49 → 24.64 cm/cycle** |

The combined F3 run finished **671–671 goals** in 155 games. Repaired F4
finished **222–225** in the first 60, and the separate randomized-priority
snapshot was **122–143** in 35 games. The ball correction remained experimental.

## Setup

Reuse Exp3's 200 quantized 360° games. Train player velocity/body and ball
position/velocity models, also testing next-cycle ball forecasts. Preserve
directly observed player body angles. Audit difficult cases around kick
and control boundaries.

## Steps and reports

1. Compare [offline state models](reports/01-state-models.md) and [live integration](reports/02-live-integration.md).
2. Diagnose the [155-game F3 run](reports/03-live-f3-audit.html). Repair
   observation history and ball-state routing, then recheck the
   [first 60 F4 games](reports/05-live-f4-60-games.html).
3. Add [opt-in random command priority](reports/06-server-command-order.html)
   and verify it in the [first 15 randomized games](reports/07-random-order-first15.html).
4. Use the [SoccerWindow overlays](reports/08-debug-overlay.html) to inspect
   [ball velocity and position jumps](reports/09-closeout-ball-stability.html).

## Reproduce

[Methods, model commands and historical reports](METHODS.md)
