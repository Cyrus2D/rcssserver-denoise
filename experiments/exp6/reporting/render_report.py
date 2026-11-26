#!/usr/bin/env python3
"""Render Exp6 and compare its player measurements with archived Exp3 metrics."""
import argparse
import json
import math
from pathlib import Path
import shutil

import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_logs import DISTANCES

LABELS = ("Exp3 quantized", "Exp6 Gaussian")
COLORS = ("#64748b", "#007d85")


class Run:
    def __init__(self, path):
        self.data = json.loads(path.read_text())
        self.coverage = self.data["coverage"]
        self.rows = {(r["target"], r["distance"], r["metric"]): r for r in self.data["metrics"]}

    def metric(self, key, group="all", distance=None):
        distances = [distance] if distance else DISTANCES
        rows = [r for d in distances if (r := self.rows.get((group, d, key))) and r["n"]]
        count = sum(r["n"] for r in rows)
        if not count:
            return 0, math.nan, math.nan
        return (count, sum(r["n"] * r["mean"] for r in rows)/count,
                sum(r["n"] * r.get("mean_square", math.nan) for r in rows)/count)

    def mean(self, key, group="all", distance=None):
        return self.metric(key, group, distance)[1]


def num(value, digits=2):
    if not math.isfinite(value):
        return "—"
    return f"{0.0 if round(value,digits) == 0 else value:,.{digits}f}"


def pct(value):
    return num(value*100) + "%" if math.isfinite(value) else "—"


def table(headers, rows):
    def row(cells):
        return "| " + " | ".join(map(str, cells)) + " |"
    return "\n".join([row(headers), row(["---"] + ["---:"]*(len(headers)-1)),
                      *(row(cells) for cells in rows)]) + "\n"


def plot(ax, title, ylabel, series, signed=False):
    for label, values, color in series:
        ax.plot(DISTANCES, values, "o-", color=color, label=label)
    ax.set(title=title, ylabel=ylabel, xlabel="True observer-to-player distance (m)")
    if signed:
        ax.axhline(0, color="#64748b", lw=.7)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)


def charts(runs, gaussian, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, title, key in zip(axes.flat,
            ("Raw range", "Raw relative position", "Fresh visual localization", "Fused world model"),
            ("raw_range", "raw_position", "paired_visual", "paired_track")):
        plot(ax, title, "Mean error (cm)", [
            (label, [100*r.mean(key, distance=d) for d in DISTANCES], color)
            for r, label, color in zip(runs, LABELS, COLORS)])
    fig.savefig(out/"player-error-by-distance.svg")
    fig.savefig(out/"player-error-by-distance.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    plot(axes[0], "Which measurement limits position?", "Mean relative error (cm)", [
        (label, [100*gaussian.mean(key, distance=d) for d in DISTANCES], color)
        for label, key, color in [
            ("Raw distance and bearing","raw_position","#007d85"),
            ("True distance substituted","exact_range_position","#c66a24"),
            ("True bearing substituted","exact_bearing_position","#64748b")]])
    plot(axes[1], "Range bias before and after localization", "Mean signed error (cm)", [
        (label, [100*gaussian.mean(key, distance=d) for d in DISTANCES], color)
        for label, key, color in [
            ("Raw range","signed_range","#007d85"),
            ("Range after localization","signed_localized_range","#c66a24")]], signed=True)
    fig.savefig(out/"range-diagnostics.svg")
    plt.close(fig)


def render(base, gaussian, out, lookup):
    runs = (base, gaussian)
    c = gaussian.coverage
    qmean, gmean = base.mean, gaussian.mean
    games = gaussian.data["scores"]
    score = [sum(row[side] for row in games) for side in ("left","right")]
    wins = sum(row["left"] > row["right"] for row in games)
    draws = sum(row["left"] == row["right"] for row in games)
    verified = sum(row["server_verified"] for row in games)
    targets, known, _ = gaussian.metric("known_share")
    n_ranges = c["known_ranges"]
    cover = table(("Check", *LABELS), [
        ("PlayOn decisions", *(f"{r.coverage['play']:,}" for r in runs)),
        *[(label, *(pct(r.coverage[key]/r.coverage["play"]) for r in runs)) for label,key in [
            ("360° view","full_view"),("High-quality view","high_quality"),
            ("Exactly one same-cycle see","one_see"),("Visual update used","see"),
            ("Valid reference fullstate","truth")]],
        ("Identified on-field player rows", *(pct(r.mean("known_share")) for r in runs)),
        ("Teammate payloads", *(f"{r.coverage.get('teammate_payload',0):,}" for r in runs)),
        ("Skipped-decision placeholders, all modes", *(f"{r.coverage.get('skipped',0):,}" for r in runs)),
    ])

    def means(spec):
        return table(("Mean error", *LABELS), [
            (label, *(num(scale*r.mean(key), digits) for r in runs))
            for label,key,scale,digits in spec])

    overall = means([
        ("Raw range MAE (cm)","raw_range",100,2),
        ("Raw bearing MAE (°)","raw_bearing",1,3),
        ("Raw relative position (cm)","raw_position",100,2),
        ("Relative position with true range (cm)","exact_range_position",100,2),
        ("Relative position with true bearing (cm)","exact_bearing_position",100,2),
        ("Localized relative position (cm)","visual_relative",100,2),
        ("Visual global position (cm), paired","paired_visual",100,2),
        ("WM global position (cm), paired","paired_track",100,2),
        ("WM global position (cm), all available","track_position",100,2),
    ])
    per_range = table(("True distance (m)","Raw range (cm)","Raw relative (cm)",
                       "Visual global (cm)","WM global (cm)"), [
        (d, *[" / ".join(num(100*r.mean(key,distance=d)) for r in runs)
              for key in ("raw_range","raw_position","paired_visual","paired_track")])
        for d in DISTANCES])
    sides = table(("Target","Exp3 raw relative (cm)","Exp6 raw relative (cm)",
                   "Exp3 WM position (cm)","Exp6 WM position (cm)"), [
        (g.title(), *(num(100*r.mean(key,g)) for key in ("raw_position","track_position")
                      for r in runs)) for g in ("teammate","opponent")])

    common_mix = sum(base.metric("raw_range",distance=d)[0]*gmean("raw_range",distance=d)
                     for d in DISTANCES) / base.metric("raw_range")[0]
    zrows = []
    for group,label in [("all","Players"),("ball","Ball"),("landmark","Flags and goals")]:
        n,mean,square = gaussian.metric("z",group)
        zrows.append((label,f"{n:,}",num(mean,4),num(math.sqrt(max(0,square-mean*mean)),4),
                      *(pct(gmean(f"z_abs_over_{k}",group)) for k in (1,2,3))))
    zrows.append(("Standard normal","—","0","1","31.73%","4.55%","0.27%"))
    noise_model = table(("Object","Samples","Mean z","Std dev z","abs(z) > 1","abs(z) > 2","abs(z) > 3"),zrows)
    motion = means([
        ("Raw distance-change MAE (m/cycle)","n_dist_chg",1,4),
        ("Raw bearing-change MAE (°/cycle)","n_dir_chg",1,4),
        ("Raw body MAE (°)","n_body_rel",1,4),
        ("Raw head MAE (°)","n_head_rel",1,4),
        ("Visual velocity error (m/cycle)","visual_velocity",1,4),
        ("WM velocity error (m/cycle)","track_velocity",1,4),
    ])
    motion_coverage = table(("True distance (m)","Exp3 motion fields","Exp6 motion fields",
                             "Exp3 body angle","Exp6 body angle"), [
        (d, *(pct(r.mean(k,distance=d)) for k in ("motion_share","body_share") for r in runs))
        for d in DISTANCES])
    other = table(("Exp6 object and measure","Mean error","Samples"), [
        (label, num(scale*gmean(key,group),digits)+unit, f"{gaussian.metric(key,group)[0]:,}")
        for label,group,key,scale,digits,unit in [
            ("Ball raw distance","ball","raw_range",100,2," cm"),
            ("Ball raw relative position","ball","raw_position",100,2," cm"),
            ("Ball visual global position","ball","paired_visual",100,2," cm"),
            ("Ball WM position","ball","paired_track",100,2," cm"),
            ("Ball WM velocity","ball","track_velocity",1,4," m/cycle"),
            ("Landmark raw distance","landmark","raw_range",100,2," cm"),
            ("Landmark raw bearing","landmark","raw_bearing",1,3,"°"),
            ("Self fresh visual position fix","self","self_fix_position",100,2," cm"),
            ("Self WM position","self","self_position",100,2," cm"),
            ("Self WM velocity","self","self_velocity",1,4," m/cycle"),
            ("Self WM looking direction","self","self_face",1,3,"°"),
        ]])
    raw_bias = 100*gmean("signed_range")
    localized_bias = 100*gmean("signed_localized_range")
    bias_rows = table(("True distance (m)","Raw signed error (cm)","Localized signed error (cm)",
                       "Raw MAE (cm), paired","Localized MAE (cm), paired"), [
        (d, *(num(100*gmean(key,distance=d)) for key in (
            "signed_range","signed_localized_range","paired_raw_range","localized_range")))
        for d in DISTANCES])
    text = f"""# Exp6 · Gaussian visual noise versus Exp3 quantization

**Gaussian player range MAE was {100*gmean('raw_range'):.1f} cm, compared with
{100*qmean('raw_range'):.1f} cm under quantization. Player WM position error was
{100*gmean('track_position'):.1f} cm versus {100*qmean('track_position'):.1f} cm.**
Each experiment used 200 games with the stock agents and 360° vision.

[Exp3 report](../../exp3/reports/report.html) · [Exp6 metrics and scores](metrics.json) · [Run settings](../README.md) · [Step 1 range correction](range-replay/report.html)

## 1. Collection checks

All **4,400** Exp6 observer logs finalized. Both sides requested Gaussian and
used reference fullstate for labels. The runs share synchronized 360° vision,
persistent uniform numbers, standard motion-field fading and no player
messages. Exp6 disabled both exact-vision switches and used the standard
Gaussian rates for movable objects and landmarks. Random command-order
settings by observer were {gaussian.data['command_order']}.

{cover}

Exp6 had **{targets:,}** eligible on-field player rows with reference geometry.
Among available WM player positions, **{pct(gmean('fresh_wm'))}** had position
count 0. **{pct(c['off_grid_ranges']/n_ranges)}** of known ranges fell outside
the 0.1 m grid. The batch retained the parsed fields in CSV. Gaussian
acknowledgment packets were verified in the collection pilot.

All **{verified}** match scores agree with AutoTest2D. Goals were
**{score[0]}–{score[1]}**, with **{wins}/{draws}/{len(games)-wins-draws}**
left wins/draws/losses. Both teams used the same setup.

## 2. Player error from the sensor to the world model

{overall}

![Player error by distance in both experiments](player-error-by-distance.svg)

Each cell below shows **Exp3 / Exp6**. Raw relative position compares the
observed polar vector with the true vector in the observer's own frame.
Visual and WM global errors include observer pose error. The visual and WM
paired columns require both estimates for the same observer, target and
cycle within each experiment.

{per_range}

{sides}

Gaussian range MAE was **{gmean('raw_range')/qmean('raw_range'):.2f}×** the
quantized baseline. Weighting Gaussian by Exp3's distance-bin counts gives
**{100*common_mix:.1f} cm**. These are independent games, so differences in
geometry can remain within each bin.

The analysis includes PlayOn only and gives every eligible sample equal
weight. Raw and visual player errors require an unambiguous uniform-number
association. All-available WM errors include any valid matched track. Exp3
uses its archived aggregates, which limit comparisons to shared measurements.

## 3. The Gaussian distance model and the localization step

Substituting true distance on the same Exp6 rows reduces mean raw relative
position error from **{100*gmean('raw_position'):.1f} to
{100*gmean('exact_range_position'):.1f} cm**. Substituting true bearing leaves
**{100*gmean('exact_bearing_position'):.1f} cm**. Each substitution changes one
measurement and uses truth for diagnosis.

![Range versus bearing and range bias](range-diagnostics.svg)

Mean signed raw Gaussian range error is **{raw_bias:+.2f} cm**. The norm of
librcsc's localized relative vector minus true distance averages
**{localized_bias:+.2f} cm**. On rows with both ranges, MAE is
**{100*gmean('paired_raw_range'):.1f} cm before localization** and
**{100*gmean('localized_range'):.1f} cm after localization**. Comparing
the two ranges removes observer position and heading from this check.
In librcsc, the lookup selects the first quantization-table bin whose
reported value is at least the observed distance minus 0.001 m. Continuous
Gaussian values between bins therefore often move to the next bin. This
explains the upward bias. Replaying the stock table reproduced
**{lookup["matches_allow_csv_rounding"]:,} / {lookup["n"]:,}** logged localized
ranges within 0.15 mm on four observer streams from the first and last games.
[Lookup replay evidence](lookup-replay.json)

{bias_rows}

Exp6 player WM position error exceeded 50 cm in
**{pct(gmean('wm_over_50cm'))}** and 1 m in **{pct(gmean('wm_over_100cm'))}**
of available tracks.

The server sets σ = d × distance rate + f × focus rate, where d is the
true observer-to-object distance and f is the true focus-to-object distance.
It clamps the noisy distance at zero. The table checks
**z = (observed distance − true distance) / σ**. Nonzero observer focus distance
appears in **{pct(c.get('focus_nonzero',0)/c['focus_labeled'])}** of labeled
PlayOn decisions.

{noise_model}

## 4. Velocity and body fields

Both raw motion fields were available for **{pct(qmean('motion_share'))}**
of Exp3 targets and **{pct(gmean('motion_share'))}** of Exp6 targets.

{motion}

{motion_coverage}

Coverage uses all eligible on-field targets. Error means use observations
with the relevant field or a valid estimate. The server scales raw radial
change with the same noisy distance ratio before rounding. Bearing, body
and head use the same angular encoding as the quantized mode.

## 5. Ball, landmarks and self in Exp6

{other}

These measures give Exp6 reference points for later modeling. The archived
Exp3 player metrics cover the player comparisons above.

## Reproduce

Run from the denoise workspace. The first command streams the compressed
CSVs and retains per-game totals for resuming interrupted runs.

    exp=rcssserver-denoise/experiments/exp6
    mkdir -p "$exp/reports"
    python3 "$exp/reporting/analyze_logs.py" logs/exp6-logs logs/exp6-analysis/baseline-200 --autotest AutoTest2D/out/exp6logs --autotest AutoTest2D/out/exp6logs2
    .venv-exp5-s1/bin/python "$exp/reporting/verify_lookup.py" logs/exp6-logs librcsc-denoise/rcsc/player/object_table.cpp "$exp/reports/lookup-replay.json"
    .venv-exp5-s1/bin/python "$exp/reporting/render_report.py" rcssserver-denoise/experiments/exp3/reports/exp3-metrics.json logs/exp6-analysis/baseline-200/metrics.json "$exp/reports"

"""
    (out/"README.md").write_text(text,encoding="utf-8")
    css = """body{font:16px/1.6 system-ui,sans-serif;color:#1e293b;background:#f7fafc;margin:0}
main{max-width:1080px;margin:auto;padding:32px 24px}img{display:block;width:100%;background:white}
table{border-collapse:collapse;background:white;display:block;overflow:auto;margin:20px 0}
td,th{padding:7px 12px;border:1px solid #d5e2e8}th{background:#e4eff1;text-align:left}
td:not(:first-child){text-align:right}h2{margin:38px 0 12px}
pre{overflow:auto;background:#edf4f7;padding:16px}a{color:#075e71}"""
    html = markdown.markdown(text,extensions=["tables"])
    (out/"report.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp6 Gaussian versus Exp3 quantization</title><style>'+css+
        '</style></head><body><main>'+html+'</main></body></html>',encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp3",type=Path)
    parser.add_argument("exp6",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    base, gaussian = Run(args.exp3), Run(args.exp6)
    if base.data["games"] != 200 or gaussian.data["games"] != 200 or gaussian.data["observers"] != 4400:
        parser.error("the baseline report expects two complete 200-game cohorts")
    args.output.mkdir(parents=True,exist_ok=True)
    charts((base,gaussian),gaussian,args.output)
    lookup = json.loads((args.output/"lookup-replay.json").read_text())
    render(base,gaussian,args.output,lookup)
    dest = args.output/"metrics.json"
    if args.exp6.resolve() != dest.resolve():
        shutil.copy2(args.exp6,dest)


if __name__ == "__main__":
    main()

