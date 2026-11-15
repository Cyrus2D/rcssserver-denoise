#!/usr/bin/env python3
"""Render the Exp3 player denoising report from the pooled cohort summaries."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.exp1.reporting.render_denoise_report import COLORS, STYLE, image, legend, plt, style, table

from .analyze_players import DISTANCES


class Run:
    def __init__(self, path):
        self.data = json.loads(path.read_text())
        self.coverage = self.data["coverage"]
        self.rows = {(row["target"], row["distance"], row["metric"]): row for row in self.data["metrics"]}

    def metric(self, name, distance=None, target="all"):
        labels = (distance,) if distance else DISTANCES
        rows = [self.rows[target, label, name] for label in labels]
        count = sum(row["n"] for row in rows)
        value = sum(row["n"] * row["mean"] for row in rows if row["n"]) / count if count else np.nan
        return count, value

    def values(self, name, scale=1):
        return [self.metric(name, label)[1] * scale for label in DISTANCES]


def number(value, precision=2):
    return f"{value:,.{precision}f}" if np.isfinite(value) else "—"


def positive_count(result):
    count, share = result
    return round(count * share) if count else 0


def chart(series, ylabel, caption):
    figure, ax = plt.subplots(figsize=(8.8, 3.4), layout="constrained")
    for (label, values, pattern), color in zip(series, COLORS):
        ax.plot(DISTANCES, values, pattern, color=color, lw=2, label=label)
    style(ax, "True observer-to-player distance (m)", ylabel)
    ax.set_ylim(bottom=0)
    legend(ax, len(series))
    return image(figure, caption)


def write_report(base, exp, output):
    coverage_rows = []
    for label, metric in (("See used on PlayOn decisions", "see"), ("Exactly one see", "one_see"),
                          ("360° view", "full_view"), ("High quality", "high_quality"),
                          ("Valid reference fullstate", "truth")):
        coverage_rows.append((label, *(number(100 * run.coverage[metric] / run.coverage["play"], 2) + "%"
                                       for run in (base, exp))))
    coverage_rows.append(("Teammate payloads logged", *(f'{run.coverage["teammate_payload"]:,}' for run in (base, exp))))

    available = table(("True distance (m)", "Exp1 seen", "Exp3 seen", "Exp1 known identity", "Exp3 known identity",
                       "Exp1 motion fields", "Exp3 motion fields"), (
        (label, *(number(run.metric(metric, label)[1] * 100, 1) + "%"
                  for metric in ("seen_share", "known_share", "motion_share") for run in (base, exp)))
        for label in DISTANCES
    ))
    polar = table(("True distance (m)", "Known observations Exp1 / Exp3", "Range MAE Exp1 / Exp3 (cm)",
                   "Bearing MAE Exp1 / Exp3 (°)", "Raw relative error Exp1 / Exp3 (cm)",
                   "Exp3 librcsc visual relative (cm)"), (
        (label, " / ".join(f"{run.metric('raw_position', label)[0]:,}" for run in (base, exp)),
         *(" / ".join(number(run.metric(metric, label)[1] * scale, 3 if metric == "raw_bearing" else 2)
                      for run in (base, exp))
           for metric, scale in (("raw_range", 100), ("raw_bearing", 1), ("raw_position", 100))),
         number(exp.metric("visual_relative", label)[1] * 100))
        for label in DISTANCES
    ))
    tracks = table(("True distance (m)", "Exp1 visual (cm)", "Exp1 WM (cm)", "Exp3 visual (cm)", "Exp3 WM (cm)"), (
        (label, *(number(run.metric(metric, label)[1] * 100)
                  for run in (base, exp) for metric in ("paired_visual", "paired_track")))
        for label in DISTANCES
    ))
    global_tracks = table(("Target", "Exp1 all available tracks (cm)", "Exp3 all available tracks (cm)"), (
        (label, *(number(run.metric("track_position", target=key)[1] * 100) for run in (base, exp)))
        for label, key in (("Teammates", "teammate"), ("Opponents", "opponent"))
    ))
    motion = table(("True distance (m)", "Exp3 full motion fields", "Range change MAE (m/cycle)",
                    "Bearing change MAE (°/cycle)", "Body MAE (°)", "Head MAE (°)"), (
        (label, f"{positive_count(exp.metric('motion_share', label)):,}",
         *(number(exp.metric(metric, label)[1], 3)
           for metric in ("n_dist_chg", "n_dir_chg", "n_body_rel", "n_head_rel")))
        for label in DISTANCES
    ))
    coverage_chart = chart((
        ("Exp1 known identity", base.values("known_share", 100), "o--"),
        ("Exp3 known identity", exp.values("known_share", 100), "o-"),
        ("Exp3 motion fields", exp.values("motion_share", 100), "s-"),
    ), "Share of on-field targets (%)", "Identity uses a team and uniform-number match. Motion requires both change fields and a known identity.")
    diagnostic_chart = chart((
        ("Raw range and bearing", exp.values("raw_position", 100), "o-"),
        ("Exact range only", exp.values("exact_range_position", 100), "s-"),
        ("Exact bearing only", exp.values("exact_bearing_position", 100), "o--"),
    ), "Mean relative position error (cm)", "Exp3 diagnostic using the same known observations. Each exact component comes from fullstate for measurement only.")
    raw_n, raw = exp.metric("raw_position")
    baseline_track, exp_track = (run.metric("track_position")[1] for run in (base, exp))
    baseline_fresh, exp_fresh = (run.metric("raw_position", "10–20")[1] for run in (base, exp))
    range_only = exp.metric("exact_range_position")[1]
    bearing_only = exp.metric("exact_bearing_position")[1]
    focus = "range" if range_only < bearing_only else "bearing"
    output.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exp3 · Player quantization</title><style>{STYLE}</style></head><body><main>
<header><div class="kicker">RoboCup 2D · Denoise experiment</div><h1>Player quantization</h1>
<p class="lede">What still limits player localization when fresh observations and identities are available?</p>
<p>Exp3 used standard quantization, 360° every cycle, persistent uniform numbers and no teammate payloads.
Exp1 used normal view and update timing, distance-based fading and normal teammate communication.
Both teams contribute observations in both batches.</p>
<div class="cards"><div class="card"><strong>{exp.data['games']}</strong><span>Exp3 games</span></div>
<div class="card"><strong>{raw_n:,}</strong><span>Exp3 known player observations with range and bearing</span></div>
<div class="card"><strong>{100*raw:.1f} cm</strong><span>Exp3 mean raw relative-position error</span></div></div>
<nav aria-label="Report sections"><a href="#coverage">Coverage</a><a href="#quantization">Quantization</a>
<a href="#tracks">World model</a><a href="#motion">Motion fields</a><a href="#next">Next experiment</a></nav></header>
<section id="coverage"><h2>Did the configuration work?</h2>
{table(("Check", "Exp1", "Exp3"), coverage_rows)}
{coverage_chart}{available}
<p>Percentages in the distance table use all on-field player targets at the observer's PlayOn decisions.
The unum switch preserves identity. It leaves motion, body and head fields subject to fading.</p></section>
<section id="quantization"><h2>How much error remains in a fresh measurement?</h2>
<p>Raw relative position is the observed polar vector compared with the true polar vector in the observer's
own frame. Its Euclidean error excludes observer position and orientation error. Raw-error comparisons
use unambiguous uniform-number matches. Exp1's long-distance entries can represent a different and much
smaller subset even inside the same distance bin. The librcsc relative estimate uses its estimated
looking direction to rotate the visual vector into global axes. It still excludes observer position error.</p>
{polar}{diagnostic_chart}
<p>Replacing range alone leaves {100*range_only:.1f} cm mean error. Replacing bearing alone leaves
{100*bearing_only:.1f} cm. The larger reduction comes from correcting <strong>{focus}</strong>.
The exact-component curves use truth as a diagnostic. Their errors are separate counterfactuals.</p></section>
<section id="tracks"><h2>Fresh visual position versus the world model</h2>
<p>Across available tracks, mean error went from <strong>{baseline_track:.3f} to {exp_track:.3f} m</strong>.
Fresh raw relative error at 10–20 m was nearly the same, <strong>{baseline_fresh*100:.2f} and
{exp_fresh*100:.2f} cm</strong>. The gain in tracking accuracy came with much better coverage
while substantial fresh measurement error remained.</p>
<p>Both estimates use the same player and cycle within each run. They are scored in global coordinates,
so observer localization also affects them.</p>{tracks}
<h3>Track error across all available tracks</h3>{global_tracks}
<p>The all-track table includes stale estimates and the logger's heuristic associations.
The changes between runs describe the combined configuration. The batches do not isolate the individual
effects of view, identity and hearing.</p></section>
<section id="motion"><h2>Where can we evaluate measured velocity and angles?</h2>
<p>The following errors use Exp3 player observations with the relevant field and a known identity.
The count requires both motion fields. Missing values remain missing.</p>{motion}</section>
<section id="next"><h2>First denoising test</h2>
<ol><li>Keep identity fixed and start with position. Compare raw polar decoding, librcsc's fresh visual
estimate, and a constant-velocity predictor on the same samples.</li>
<li>Add a temporal filter that scores candidate positions with the server's forward quantizer.
Protocol v19 rounds bearing to the nearest degree. Distance uses a logarithmic step of 0.1 and a final
0.1 m rounding. When focus is displaced,
the forward model uses distance to the observer and distance to its focus point. Use the agent's own
focus and pose estimates. Treat identical consecutive readings as the same quantization constraint
until the relative geometry changes.</li>
<li>Split by whole match. Evaluate both relative and global error against fullstate by distance and
motion availability. Fullstate supplies labels and diagnostics. The filter consumes only observations
and the agent's own state estimates.</li></ol>
<p>A single quantized range can represent multiple true distances. Changes in relative geometry across
cycles are what can narrow that interval. Check whether the filter beats constant velocity before
adding a learned motion prior.</p>
<details><summary>Server measurement check</summary>
<p>On 206,703 identified player observations from one agent in each of two Exp3 games, replaying
the server's distance quantizer with logged true geometry reproduced 206,700 ranges. Rounding the
true bearing to the nearest degree reproduced 206,672 bearings modulo 360°. Both sampled agent
streams had zero focus distance throughout PlayOn. This check uses fullstate to verify the sensor
formula. In a live filter, candidate states and the observer's estimated focus supply the geometry.</p>
</details></section>
</main></body></html>
""", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp1", type=Path)
    parser.add_argument("exp3", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_report(Run(args.exp1), Run(args.exp3), args.output)
