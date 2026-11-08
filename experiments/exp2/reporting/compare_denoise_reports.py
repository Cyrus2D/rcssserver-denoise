#!/usr/bin/env python3
"""Render a comparison of the Exp1 and Exp2 left-observer reports."""

import argparse
import csv
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from experiments.exp1.reporting.render_denoise_report import COLORS, STYLE, TARGETS, image, legend, plt, style, table


BASE = "Exp1 · baseline"
LIVE = "Exp2 · apply"
MEAN_HEADERS = ("Metric", "Unit", BASE, LIVE, "Change (Exp2 − Exp1)")


def load(directory):
    run = json.loads((directory / "run.json").read_text())
    if run["coverage"].get("observer_side") != "left":
        raise ValueError(f"Expected a left-observer report in {directory}")
    with (directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        means = {(row["panel"], row["group"], row["metric"]): float(row["mean"])
                 for row in csv.DictReader(stream) if row["mean"]}
    return run, means


def scores(run):
    root = Path(run["source"])
    games = []
    for game in sorted(root.iterdir()):
        files = sorted(game.glob("l_*/meta.json"))
        if not files:
            continue
        records = [json.loads(path.read_text()) for path in files]
        if any(not row["final"] or row["status"] != "ok" for row in records):
            raise ValueError(f"Unfinalized match in {game}")
        results = {(row["score_left"], row["score_right"]) for row in records}
        if len(results) != 1:
            raise ValueError(f"Conflicting final scores in {game}")
        games.append(results.pop())
    if len(games) != run["coverage"]["games"]:
        raise ValueError(f"Score count differs from the report in {root}")
    return np.asarray(games)


def score_rows(base, live):
    def values(result):
        left, right = result.T
        wins, draws, losses = int((left > right).sum()), int((left == right).sum()), int((left < right).sum())
        return (
            f"{wins} / {draws} / {losses}",
            f"{left.sum()} : {right.sum()}",
            f"{wins * 3 + draws} : {losses * 3 + draws}",
        )
    return zip(("Left wins / draws / losses", "Goals left : right", "Points left : right"),
               values(base), values(live))


def paired_plot(ages):
    labels = ("Fresh", "1 cycle", "2 cycles")
    keys = ("age 0", "age 1", "age 2")
    figure, ax = plt.subplots(figsize=(7.6, 3.3), layout="constrained")
    for key, name, color in zip(("baseline", "model"), ("librcsc", "Applied model"), COLORS):
        ax.plot(labels, [ages[age][key]["mean"] for age in keys], "o-", lw=2.2, label=name, color=color)
    style(ax, "Time since last visual position fix", "Mean self position error (cm)")
    ax.set_ylim(bottom=0)
    legend(ax, 2)
    return image(figure, "Each pair uses the same left-team player and cycle in Exp2. Lower is better.")


def object_gap(base, live):
    figure, axes = plt.subplots(1, len(TARGETS), figsize=(10, 3.5), sharey=True, layout="constrained")
    rows = []
    for ax, target in zip(axes, TARGETS):
        names = ("Fresh", "3–5 cycles")
        pairs = [(base["object_visual_age_under_50m", f"{target} / {age}", "n_trk_pos_d"],
                  live["object_visual_age_under_50m", f"{target} / {age}", "n_trk_pos_d"])
                 for age in ("0", "3–5")]
        for index, (name, color) in enumerate(zip((BASE, LIVE), COLORS)):
            ax.plot(names, [pair[index] for pair in pairs], "o-", lw=2, label=name, color=color)
        ax.set_title(target, fontsize=11)
        style(ax, "", "Mean position error (m)" if target == TARGETS[0] else "")
        ax.set_ylim(bottom=0)
        for name, (before, after) in zip(names, pairs):
            rows.append((target, name, f"{before:.3f}", f"{after:.3f}", f"{after - before:+.3f}"))
    legend(axes[1], 2)
    return image(figure, "World-model position error within 50 m of the left-team observer. Each line spans two visual-age groups.") + table(
        ("Target", "Last visual fix", BASE + " (m)", LIVE + " (m)", "Change (m)"), rows)


def mean_table(base, live, specs):
    rows = []
    for label, unit, scale, panel, group, metric in specs:
        key = (panel, group, metric)
        a, b = scale * base[key], scale * live[key]
        rows.append((label, unit, f"{a:.3f}", f"{b:.3f}", f"{b - a:+.3f}"))
    return table(MEAN_HEADERS, rows)


def write_comparison(baseline, active, live_summary, output):
    original_report = Path(__file__).resolve().parents[2] / "exp1" / "reports" / "report.html"
    original_link = quote(Path(os.path.relpath(original_report, output.parent)).as_posix())
    base_run, base = load(baseline)
    live_run, live = load(active)
    summary = json.loads(live_summary.read_text())
    if (summary["games"] != live_run["coverage"]["games"]
            or summary["coverage"]["apply"]["players"] != live_run["coverage"]["agent_logs"]
            or summary["coverage"]["apply"]["scored"] != live_run["coverage"]["play_on_rows"]):
        raise ValueError("The paired live summary and Exp2 report cover different player samples")

    ages = summary["paired"]["apply"]["ages"]
    overall = ages["all"]
    gain = 100 * (1 - overall["model"]["mean"] / overall["baseline"]["mean"])
    ci = summary["paired"]["apply"]["mean_gain_ci_cm"]
    ready = 100 * overall["model"]["n"] / summary["coverage"]["apply"]["play_on"]
    paired_table = table(
        ("Visual position age", "librcsc mean (cm)", "Model mean (cm)", "librcsc p90 (cm)", "Model p90 (cm)"),
        ((label, *(f"{ages[key][method][metric]:.3f}" for metric in ("mean", "p90")
                   for method in ("baseline", "model")))
         for key, label in (("all", "All"), ("age 0", "Fresh"), ("age 1", "1 cycle"), ("age 2", "2 cycles"))),
    )
    self_table = mean_table(base, live, (
        ("Self position used in decisions", "cm", 100, "self", "All", "n_est_pos_d"),
        ("Raw visual self position", "cm", 100, "self", "All", "n_rawfix_pos_d"),
        ("Self velocity", "cm/cycle", 100, "self", "All", "n_est_vel_d"),
        ("Looking direction", "°", 1, "self", "All", "n_est_face"),
    ))
    tracks = mean_table(base, live, (
        (f"{target} track position", "m", 1, "objects", target, "n_trk_pos_d") for target in TARGETS
    ))
    forecasts = mean_table(base, live, (
        ("Self position", "cm", 100, "next_cycle", "All", "prediction_self_pos"),
        ("Self body angle", "°", 1, "next_cycle", "All", "prediction_self_body"),
        ("Ball position", "m", 1, "next_cycle", "All", "prediction_ball_pos"),
        ("Ball velocity", "m/cycle", 1, "next_cycle", "All", "prediction_ball_vel"),
    ))
    heard = mean_table(base, live, (
        ("Decoded ball position", "m", 1, "heard_payload", "All", "n_ball_pos_d"),
        ("Decoded opponent goalie position", "m", 1, "heard_payload", "All", "n_goalie_pos_d"),
    ))
    base_scores, live_scores = scores(base_run), scores(live_run)
    goal_delta = live_scores[:, 0] - live_scores[:, 1]
    points_delta = 3 * np.sign(goal_delta)
    indices = np.random.default_rng(42).integers(len(goal_delta), size=(10000, len(goal_delta)))
    goal_ci, points_ci = (np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
                          for delta in (goal_delta, points_delta))
    output.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exp1 vs Exp2 · Denoise comparison</title><style>{STYLE}</style></head><body><main>
<header><div class="kicker">RoboCup 2D · Denoise experiment</div><h1>Exp1 vs Exp2</h1>
<p class="lede">What changed after Helios began using the learned self position?</p>
<p>Exp1 is standard librcsc on both teams. In Exp2 the left team uses the model and the right team
uses standard librcsc. The cross-run sensor and track comparisons use left-team observers in both batches.
Exp1 contains {len(base_scores)} games and Exp2 contains {len(live_scores)} games.</p>
<div class="cards">
<div class="card"><strong>{gain:.1f}% lower</strong><span>Mean self position error on paired model cycles</span></div>
<div class="card"><strong>{ready:.1f}%</strong><span>Left-team PlayOn decisions used the model</span></div>
<div class="card"><strong>{goal_delta.mean():+.2f}</strong><span>Left minus right goals per game in Exp2</span></div>
</div>
<nav aria-label="Report sections"><a href="#self">Self</a><a href="#objects">Ball and players</a>
<a href="#forecast">Predictions</a><a href="#hear">Hearing</a><a href="#matches">Match results</a>
<a href="{original_link}">Original full report</a><a href="report.html">New full report</a></nav></header>
<section id="self"><h2>Self localization</h2>
<p>On {overall["model"]["n"]:,} model-applied player-cycles, compare the model position with the librcsc
position recorded before the replacement. Both are scored against the same fullstate position.
Mean error improved in {summary["paired"]["apply"]["games_improved"]} of {len(live_scores)} games.
The 95% game-bootstrap interval for the mean reduction is {ci[0]:.3f} to {ci[1]:.3f} cm.</p>
{paired_plot(ages)}{paired_table}
<h3>Position actually used across all PlayOn decisions</h3>
<p>This cross-run view includes warmup and fallback. The integration replaces self position during
the decision. Velocity and looking direction still come from librcsc.</p>{self_table}</section>
<section id="objects"><h2>Ball and other players</h2>
<p>The table summarizes all distances. The plot compares fresh tracks with tracks last seen 3–5 cycles
ago, within 50 m. Estimates are compared to fullstate. The runs contain different matches and observations,
so shifts here describe the two batches. The self correction runs after the current visual ball and
player localization. Those observations were already converted into world positions when the correction
reached the decision model.</p>{tracks}{object_gap(base, live)}</section>
<section id="forecast"><h2>One-cycle predictions</h2>
<p>Each prediction is compared with fullstate exactly one cycle later in uninterrupted PlayOn.</p>{forecasts}</section>
<section id="hear"><h2>Hearing</h2>
<p>Decoded position errors describe the received messages. The full reports include comparisons with
local references by message type, age, and distance.</p>{heard}</section>
<section id="matches"><h2>Match results</h2>
{table(("Outcome", BASE, LIVE), score_rows(base_scores, live_scores))}
<p>In Exp2 the left-minus-right difference was {goal_delta.mean():+.2f} goals per game
(95% interval {goal_ci[0]:+.2f} to {goal_ci[1]:+.2f}) and {points_delta.mean():+.2f} points per game
(95% interval {points_ci[0]:+.2f} to {points_ci[1]:+.2f}). The intervals show how uncertain the match effect is.
Since the model always played left, these games cannot separate its effect on results from a side effect.</p>
</section><section><h2>Method</h2>
<p>Player and object metrics pool eligible PlayOn samples with equal weight. The paired self comparison
uses live decision logs. Cross-run tables use the same filters as the full reports. Match intervals resample
whole games 10,000 times. Position error is the distance between the estimate and the server position.</p>
</section></main></body></html>
""", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Exp1 left-observer report directory")
    parser.add_argument("active", type=Path, help="Exp2 left-observer report directory")
    parser.add_argument("live_summary", type=Path, help="JSON from exp2.reporting.summarize_live")
    args = parser.parse_args()
    write_comparison(args.baseline, args.active, args.live_summary, args.active / "comparison.html")
