#!/usr/bin/env python3
"""Summarize the fixed-pose range replay and short live controls."""
import argparse
import json
from pathlib import Path
import shutil

import markdown
from render_report import Run, DISTANCES, num, pct, table, plot, plt

MODES = ("stock-gaussian", "new-gaussian", "stock-quantized", "new-quantized")


def render(metrics, pilot_audit, output):
    run = Run(metrics)
    pilots = json.loads(pilot_audit.read_text())
    assert run.data["games"] == 200 and run.data["observers"] == 4400
    assert all(pilots[k]["confirmed"] == (22 if k.endswith("-gaussian") else 0) for k in MODES)
    mean = run.mean
    stock, corrected = mean("replay_stock"), mean("replay_raw")
    stock_ball, corrected_ball = mean("replay_stock","ball"), mean("replay_raw","ball")
    groups = ("all","teammate","opponent","ball")
    lines = []
    for group, name in zip(groups, ("Players","Teammates","Opponents","Ball")):
        count, before, _ = run.metric("replay_stock",group)
        paired_count, after, _ = run.metric("replay_raw",group)
        assert count == paired_count
        lines.append((name, f"{count:,}", num(before*100), num(after*100), pct(1-after/before),
                      pct(mean("replay_improved_1cm",group)), pct(mean("replay_worsened_25cm",group))))
    offline_table = table(("Target","Paired rows","Stock cm","Raw range cm","Mean reduction",
                           "Improved >1 cm","Worsened >25 cm"), lines)
    distance_table = table(("Player distance (m)","Stock cm","Raw range cm","Worsened >25 cm"), [
        (d,num(100*mean("replay_stock",distance=d)),num(100*mean("replay_raw",distance=d)),
         pct(mean("replay_worsened_25cm",distance=d))) for d in DISTANCES])
    live_table = table(("Short match","PlayOn observer cycles","Player WM cm","Ball WM cm","Self WM cm"), [
        (name, f"{pilots[name]['play']:,}", *(num(100*pilots[name]["samples"][metric]["mean"])
         for metric in ("all_track_position","ball_track_position","self_self_position")))
        for name in MODES])
    routes = table(("Short match","Player ranges checked","Ball ranges checked","Localized radial source"), [
        (name, f"{pilots[name]['player_checked']:,}", f"{pilots[name]['ball_checked']:,}",
         "reported distance" if name=="new-gaussian" else "quantization table")
        for name in MODES])

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9,4),layout="constrained")
    plot(ax,"Player position error with the same estimated observer pose","Mean global error (cm)",[
        ("Stock Gaussian localization",[100*mean("replay_stock",distance=d) for d in DISTANCES],"#64748b"),
        ("Reported range",[100*mean("replay_raw",distance=d) for d in DISTANCES],"#007d85")])
    fig.savefig(output/"range-replay.svg")
    plt.close(fig)

    report = f"""# Exp6 step 1 · Use reported Gaussian ranges in librcsc

With observer position and angle held fixed, player position error fell from
**{stock*100:.2f} to {corrected*100:.2f} cm** across the 200 Exp6 games.
Ball position error fell from **{stock_ball*100:.2f} to {corrected_ball*100:.2f} cm**.
The rebuilt librcsc applies the Gaussian range path after the server
acknowledges the mode. The live checks verified the reported-range path for
Gaussian and the original lookup for quantized observations.

[Original Exp6 baseline](../report.html) · [Replay metrics](replay-metrics.json) · [Pilot verification](pilot-audit.json)

## Paired replay on existing games

For each valid fresh localized observation, we kept the direction and origin
used by librcsc. The new position is the old global position plus
(reported range − localized range) times the unit localized relative vector.
Fullstate provides the reference for both error measurements. PlayOn rows
require both position estimates and the same-cycle truth, with a known
identity for players. Each sample has equal weight.

{offline_table}

An improvement or regression means the change in the individual sample's
error against truth. Some Gaussian draws are closer to truth after the old
lookup, which can partially offset an underestimated range. The replay tests
the global visual fix along the recorded trajectories. The live changes can
also affect landmark localization, player/ball velocities and later actions.

![Paired replay by distance](range-replay.svg)

{distance_table}

## What changed in librcsc

- Acknowledged Gaussian observations use reported distance for landmarks,
  players and the ball. The existing estimators get a working radial interval
  of ±2 estimated sigma, derived from the observer's player-type noise rates
  and measured range and focus geometry.
- Ball radial velocity uncertainty uses the Gaussian 0.01 m/cycle rounding
  step. At a reported range of zero it leaves direct velocity unavailable.
- Server exact-observation flags take precedence. The logged
  **gaussian_range_localization** field identifies the active range path.

## Short live controls

One stock and one updated match per sensor mode used the Exp6 360° preset,
the same simulator and player-type seeds and 22 finalized observers each.
Trajectories and the number of PlayOn rows vary by match.

{live_table}

{routes}

Every checked radius matched its expected source within 0.15 mm, including
allowance for CSV rounding near a table boundary. Updated Gaussian player
distance intervals also matched the expected two-sigma width in this
zero-focus run. These short matches test routing and estimate quality.
The paired replay is the population comparison.

## Reproduce

    exp=rcssserver-denoise/experiments/exp6
    python3 "$exp/reporting/analyze_logs.py" logs/exp6-logs logs/exp6-analysis/raw-range-replay-200 --autotest AutoTest2D/out/exp6logs --autotest AutoTest2D/out/exp6logs2
    .venv-exp5-s1/bin/python "$exp/reporting/analyze_range_pilot.py" {pilot_audit.parent} logs/exp6-analysis/raw-range-replay-200/scan_logs librcsc-denoise/rcsc/player/object_table.cpp {pilot_audit}
    .venv-exp5-s1/bin/python "$exp/reporting/render_range_study.py" logs/exp6-analysis/raw-range-replay-200/metrics.json {pilot_audit} "$exp/reports/range-replay"
"""
    (output/"README.md").write_text(report,encoding="utf-8")
    html = markdown.markdown(report,extensions=["tables"])
    css = ("body{font:16px/1.65 system-ui,sans-serif;color:#1e293b;background:#f7fafc;margin:0}"
           "main{max-width:1050px;margin:auto;padding:28px 24px}img{width:100%;background:white}"
           "table{border-collapse:collapse;display:block;overflow:auto;margin:18px 0;background:white}"
           "td,th{padding:8px 12px;border:1px solid #d5e2e8}th{background:#e4eff1}"
           "pre{overflow:auto;padding:16px;background:#edf4f7}a{color:#075e71}")
    (output/"report.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp6 Gaussian range handling</title><style>'+css+
        '</style></head><body><main>'+html+'</main></body></html>', encoding="utf-8")
    packed = {k:run.data[k] for k in ("source","games","observers")}
    packed["metrics"] = [r for r in run.data["metrics"] if r["metric"].startswith("replay_")]
    (output/"replay-metrics.json").write_text(json.dumps(packed,indent=2,ensure_ascii=False)+"\n")
    shutil.copy2(pilot_audit,output/"pilot-audit.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("metrics","pilot_audit","output"):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    render(args.metrics,args.pilot_audit,args.output)
