"""Render the denoise summary as a standalone HTML report and long-form CSV."""

import base64
import csv
import io
import json
import math
import struct
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from . import denoise_hearing as heard


COLORS = ("#007c78", "#325db2", "#c55a28")
AGES = ("0", "1", "2", "3–5", "6–10", "11+")
DISTANCES = ("0–3", "3–10", "10–20", "20–40", "40–60", "60+")
TARGETS = ("Ball", "Teammate", "Opponent")
VIEWS = ("Narrow", "Normal", "Wide")
PREDICTIONS = (("Self position", "self_pos", "cm", 100), ("Self body", "self_body", "°", 1),
               ("Ball position", "ball_pos", "m", 1), ("Ball velocity", "ball_vel", "m/cycle", 1))
MIN_PLOT_SAMPLES = 100
POSITION_CHART = ("Position error (cm)", 100, (("n_est_pos_d", "World model"),))
VELOCITY_CHART = ("Velocity error (cm/cycle)", 100, (("n_est_vel_d", "World model"),))
FACE_CHART = ("Looking direction error (°)", 1, (("n_est_face", "World model"),))
VISUAL_COMPARISON = (
    ("Position error (cm)", 100, (("n_rawfix_pos_d", "Fresh visual"), ("n_est_pos_d", "World model"))),
    ("Looking direction error (°)", 1, (("n_rawfix_face", "Fresh visual"), ("n_est_face", "World model"))),
)
SELF_LABELS = {
    "n_est_pos_d": "Self position error (distance)",
    "n_est_vel_d": "Self velocity error (magnitude)",
    "n_est_body": "Self body angle error",
    "n_est_face": "Self looking direction error",
    "n_est_neck": "Self neck angle error",
    "n_rawfix_pos_d": "Fresh visual position error (distance)",
    "n_rawfix_face": "Fresh visual looking direction error",
}
SIGNED_METRICS = {
    "n_est_body", "n_est_face", "n_est_neck", "n_rawfix_face",
    "n_sb_speed_mag", "n_sb_speed_dir", "n_dist", "n_dist_rel", "n_dir",
    "n_dist_chg", "n_dir_chg", "n_body_rel", "n_head_rel", "n_trk_body",
    "n_trk_face", "prediction_self_body", "persistence_self_body",
}

STYLE = """
:root{color-scheme:light;--ink:#162b3d;--muted:#4b6071;--line:#cbd7df;--accent:#00645f}
*{box-sizing:border-box}body{margin:0;background:#edf3f5;color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:32px 24px 64px}header,section{background:white;border:1px solid var(--line);
border-radius:14px;padding:clamp(16px,3vw,32px);margin:0 0 18px}
h1,h2,h3{line-height:1.17;margin:0 0 12px}h1{font-size:clamp(31px,5vw,52px);letter-spacing:-.035em}
h2{font-size:27px}h3{font-size:18px;margin-top:20px}p{max-width:83ch}
.kicker{font-weight:700;color:var(--accent);letter-spacing:.1em;text-transform:uppercase;font-size:13px}
.lede{font-size:19px;color:var(--muted)}a{color:var(--accent);text-underline-offset:3px}
nav{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:16px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:20px 0}
.card{background:#f0f7f6;border:1px solid #cbdfdd;border-radius:10px;padding:12px}.card strong{display:block;font-size:25px}
.card span,.hint,figcaption{color:var(--muted);font-size:14px}figure{margin:18px 0}img{max-width:100%;height:auto;display:block}
.table{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:10px 0 18px}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}
th,td{padding:6px 10px;border-bottom:1px solid #dfe7eb;text-align:right;font-variant-numeric:tabular-nums}
th{background:#e8eff3;color:#234252}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
tbody tr:nth-child(even){background:#f7fafb}details{border-top:1px solid var(--line);padding:8px 0}
summary{cursor:pointer;font-weight:650;color:var(--accent)}code{font-size:.92em}li+li{margin-top:8px}
"""


def number(value):
    if value is None:
        return "—"
    if math.isinf(value):
        return "&gt;1,000,000"
    if value == 0:
        return "0"
    if abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:,.4g}"


def table(headers, rows):
    head = "".join(f"<th scope='col'>{escape(str(key))}</th>" for key in headers)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def statistic_table(rows):
    if not rows:
        return "<p class='hint'>No eligible rows.</p>"
    metrics = dict.fromkeys(row["metric"] for row in rows)
    metric_order = {metric: index for index, metric in enumerate(metrics)}
    display = []
    for row in sorted(rows, key=lambda row: metric_order[row["metric"]]):
        group = escape(row["group"])
        metric = escape(SELF_LABELS.get(row["metric"], row["label"]))
        unit = escape(row["unit"])
        display.append((metric, group, unit, *(number(row[key]) for key in ("mean", "median", "p90", "bias"))))
    return table(("Metric", "Group", "Unit", "Mean", "Median", "P90", "Signed bias"), display)


def image(figure, caption):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    data = buffer.getvalue()
    width, height = struct.unpack(">II", data[16:24])
    uri = base64.b64encode(data).decode("ascii")
    return (f"<figure><img width='{width}' height='{height}' alt='{escape(caption, quote=True)}' "
            f"src='data:image/png;base64,{uri}'><figcaption>{escape(caption)}</figcaption></figure>")


class Report:
    def __init__(self, rows):
        self.rows = rows
        self.index = {(r["panel"], r["group"], r["metric"]): r for r in rows}

    def result(self, panel, group, metric, field="mean"):
        return self.index.get((panel, group, metric), {}).get(field)

    def series(self, panel, groups, metric, field="mean"):
        return [self.result(panel, group, metric, field) for group in groups]

    def show(self, panel):
        return statistic_table([r for r in self.rows if r["panel"] == panel])

    def measured_series(self, panel, groups, metric, scale=1):
        values = self.series(panel, groups, metric)
        counts = self.series(panel, groups, metric, "n")
        return np.array([value * scale if value is not None and n is not None and n >= MIN_PLOT_SAMPLES
                         else np.nan for value, n in zip(values, counts)])

    def metric_chart(self, panel, groups, xlabel, axes_specs, caption, *, line=False):
        figure, axes = plt.subplots(1, len(axes_specs), squeeze=False,
                                    figsize=(9, 3.2), layout="constrained")
        x = np.arange(len(groups))
        for ax, (ylabel, scale, series) in zip(axes[0], axes_specs):
            width = 0.7 / len(series)
            for index, (metric, label) in enumerate(series):
                points = self.measured_series(panel, groups, metric, scale)
                if line:
                    ax.plot(x, points, color=COLORS[index], label=label, marker="o", linewidth=2)
                else:
                    offset = width * (index - (len(series) - 1) / 2)
                    bars = ax.bar(x + offset, points, width, color=COLORS[index], label=label)
                    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
                    ax.margins(y=0.20)
            ax.set_xticks(x, groups)
            ax.set_ylim(bottom=0)
            if line and math.isfinite(ax.dataLim.ymax) and ax.dataLim.ymax > 0:
                ax.set_ylim(top=ax.dataLim.ymax * 1.14)
            style(ax, xlabel, ylabel)
            if len(series) > 1:
                ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=len(series),
                          frameon=False, fontsize=9)
        return image(figure, caption)

    def self_detail(self, title, panel, groups, xlabel, axes_specs, caption, *, line=False):
        chart = self.metric_chart(panel, groups, xlabel, axes_specs, caption, line=line)
        return expandable(title, chart + self.show(panel))

    def highlights(self):
        comparisons = []
        for panel, label in (("object_visual_age", "All distances"),
                             ("object_visual_age_under_50m", "True distance under 50 m")):
            rows = []
            for target in TARGETS:
                fresh, stale = (self.result(panel, f"{target} / {age}", "n_trk_pos_d")
                                for age in ("0", "3–5"))
                if fresh is None and stale is None:
                    continue
                delta = stale - fresh if fresh is not None and stale is not None else None
                rows.append((target, number(fresh), number(stale), number(delta)))
            comparison = (table(("Target", "Freshly seen (m)", "Last seen 3–5 cycles ago (m)",
                                 "Difference (m)"), rows)
                          if rows else "<p class='hint'>No measured samples in these age groups.</p>")
            comparisons.append(f"<h3>{label}</h3>{comparison}")
        return ("<section><h2>Position error after a visual gap</h2>"
                "<p>Mean position error for fresh versus older visual measurements. Each column includes the "
                "available tracks in that age group. The distance filter uses the server's true distance "
                "from the observer to the target at that cycle.</p>"
                "<p><strong>Position error for one sample</strong><br>"
                "<code>√[(x<sub>WM</sub> − x<sub>truth</sub>)² + "
                "(y<sub>WM</sub> − y<sub>truth</sub>)²]</code><br>"
                "WM is the agent’s world-model position for the target. Truth is the server’s fullstate "
                "position for that same target and cycle. For example, errors of 0.30 m in x and "
                "0.40 m in y give a position error of 0.50 m.</p>"
                "<p><strong>Mean in each table cell</strong> = sum of those position errors ÷ number of "
                "measured samples for that target group and visual age.</p>"
                + "".join(comparisons) + "</section>")

    def object_distance_charts(self, metrics, targets, caption, *, rates=False):
        figure, axes = plt.subplots(1, len(metrics), squeeze=False, figsize=(9.4, 3.5),
                                    layout="constrained")
        for ax, (metric, ylabel) in zip(axes[0], metrics):
            for target in targets:
                groups = [f"{target} / {distance}" for distance in DISTANCES]
                ax.plot(DISTANCES, self.measured_series("object_distance", groups, metric),
                        marker="o", color=COLORS[TARGETS.index(target)], label=target, linewidth=2)
            style(ax, "True distance from observer (m)", ylabel)
            if rates:
                ax.yaxis.set_major_formatter(PercentFormatter())
                ax.set_ylim(0, 105)
            else:
                ax.set_ylim(bottom=0)
        legend(axes[0][0], len(targets))
        return image(figure, caption)

    def reacquisition(self):
        ages = AGES[1:]
        figure, axes = plt.subplots(1, len(TARGETS), sharey=True, figsize=(10.3, 3.4),
                                    layout="constrained")
        for ax, target in zip(axes, TARGETS):
            groups = [f"{target} / {age}" for age in ages]
            for panel, label, color in (
                ("object_reacquired_previous", "Before", COLORS[2]),
                ("object_reacquired", "Seen again", COLORS[0]),
            ):
                ax.plot(ages, self.measured_series(panel, groups, "n_trk_pos_d"), marker="o",
                        color=color, label=label, linewidth=2)
            ax.set_title(target)
            style(ax, "Visual age before update", "Mean position error (m)" if ax is axes[0] else "")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(loc="upper left", frameon=False, fontsize=9)
        return image(figure, "Same observer and target on consecutive cycles. Age is measured before the target was seen again. Both cycles must have a position error measurement.")

    def view_tradeoff(self):
        figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
        rates = self.measured_series("self_view", VIEWS, "see_used")
        axes[0].bar(VIEWS, rates, color=COLORS[0], width=.58)
        style(axes[0], "View width", "Decisions using a see")
        for target, color in zip(TARGETS, COLORS):
            groups = [f"{target} / {view}" for view in VIEWS]
            axes[1].plot(VIEWS, self.measured_series("object_view", groups, "seen"), color=color,
                         label=target, marker="o", linewidth=2)
        style(axes[1], "View width", "Eligible targets seen")
        for ax in axes:
            ax.yaxis.set_major_formatter(PercentFormatter())
            ax.set_ylim(0, 104)
        axes[1].legend(frameon=False)
        return image(figure, "Observed visual update cadence and target visibility by current Helios view width. View choice depends on the game situation.")

    def hearing(self):
        ages = ("0", "1–2", "3–5", "6+")
        figure, axes = plt.subplots(1, 3, sharey=True, figsize=(10.3, 3.5), layout="constrained")
        for ax, target in zip(axes, TARGETS):
            for cohort, color in zip(("Fresh", "Older", "No recorded report"), COLORS):
                groups = [f"{target} / {age} / 10–39 / {cohort}" for age in ages]
                ax.plot(ages, self.measured_series("object_hearing_age_distance", groups, "n_trk_pos_d"),
                        marker="o", linewidth=2, label=cohort, color=color)
            ax.set_title(target)
            style(ax, "Visual age (cycles)", "Mean position error (m)" if ax is axes[0] else "")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(loc="upper left", frameon=False, fontsize=8)
        return image(figure, "Targets 10 to under 40 m away. Compare lines at the same visual age. Groups with fewer than 100 measurements are omitted.")

    def hearing_candidate_outcomes(self):
        panel = "heard_candidate_context"
        if not any(row["panel"] == panel for row in self.rows):
            return "<p>No decoded positions could be paired with a local reference in this run.</p>"
        contexts = heard.CONTEXTS[:-1]
        targets = heard.TARGETS[:3]
        labels = ("Seen now", "Unseen 1–2", "Unseen 3–5", "Unseen 6+")
        figure, axes = plt.subplots(1, len(targets), sharex=True, sharey=True,
                                    figsize=(10.4, 3.8), layout="constrained")
        for ax, target in zip(axes, targets):
            groups = [f"{target} / {context}" for context in contexts]
            better = self.measured_series(panel, groups, "heard_better")
            worse = self.measured_series(panel, groups, "heard_worse")
            similar = np.maximum(0, 100 - better - worse)
            ax.barh(labels, better, color=COLORS[0], label="Closer by over 5 cm")
            ax.barh(labels, similar, left=better, color="#bbc8d0", label="Within 5 cm")
            ax.barh(labels, worse, left=better + similar, color=COLORS[2], label="Farther off by over 5 cm")
            ax.set_title(target)
            ax.xaxis.set_major_formatter(PercentFormatter())
            ax.set_xlim(0, 100)
            style(ax, "Share of paired reports", "")
        axes[0].invert_yaxis()
        figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3,
                      fontsize=9, frameon=False)
        return image(figure, "Each report is compared with a local position for the same observer, target and cycle. "
                     "The unseen groups use a prior-cycle baseline. Blank bars have fewer than 100 comparable reports.")

    def hearing_distance_risk(self):
        panel = "heard_candidate_distance"
        if not any(row["panel"] == panel for row in self.rows):
            return ""
        distances = heard.DISTANCES[:-1]
        labels = ("<10", "10–<40", "40+")
        figure, axes = plt.subplots(1, 3, sharey=True, figsize=(10.3, 3.5), layout="constrained")
        for ax, target in zip(axes, heard.TARGETS[:3]):
            for context, label, color in (
                ("Fresh visual", "Seen now", COLORS[2]),
                ("Unseen / 3–5", "Unseen 3–5 cycles", COLORS[0]),
            ):
                groups = [f"{target} / {context} / {distance}" for distance in distances]
                ax.plot(labels, self.measured_series(panel, groups, "heard_worse"), marker="o",
                        color=color, label=label, linewidth=2)
            ax.set_title(target)
            ax.yaxis.set_major_formatter(PercentFormatter())
            style(ax, "True target distance (m)", "Heard position farther off" if ax is axes[0] else "")
        axes[0].set_ylim(0, 105)
        figure.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2,
                      fontsize=9, frameon=False)
        return image(figure, "Share of reports that are more than 5 cm worse than the applicable local reference. "
                     "Each point has at least 100 comparable reports.")

    def hearing_effect_size(self):
        rows = []
        for target in heard.TARGETS[:3]:
            for context, label in (("Fresh visual", "Seen now"), ("Unseen / 3–5", "Unseen 3–5 cycles")):
                group = f"{target} / {context}"
                panel = "heard_candidate_context"
                count = self.result(panel, group, "candidate_pos_error", "n")
                if count is None or count < MIN_PLOT_SAMPLES:
                    continue
                received = self.result(panel, group, "candidate_pos_error")
                local = self.result(panel, group, "reference_pos_error")
                rows.append((target, label, number(local), number(received), f"{local - received:+.2f}"))
        if not rows:
            return ""
        return table(("Target", "Situation", "Local error (m)", "Heard error (m)", "Mean error reduction (m)"), rows)

    def hearing_message_types(self):
        tables = []
        for target in heard.TARGETS:
            rows = []
            for message in heard.MESSAGE_TYPES:
                group = f"{target} / {message}"
                count = self.result("heard_message_payload", group, "message_pos_error", "n")
                if count is None or count < MIN_PLOT_SAMPLES:
                    continue
                error = self.result("heard_message_payload", group, "message_pos_error")
                cells = [escape(message), number(error)]
                if target == "Ball":
                    velocity = self.result("heard_message_payload", group, "message_ball_vel_error")
                    cells.append(number(velocity))
                for reference in heard.REFERENCES:
                    key = f"{target} / {reference} / {message}"
                    n = self.result("heard_candidate_message", key, "heard_better", "n")
                    if n is None or n < MIN_PLOT_SAMPLES:
                        cells.append("—")
                        continue
                    better = self.result("heard_candidate_message", key, "heard_better")
                    worse = self.result("heard_candidate_message", key, "heard_worse")
                    cells.append(f"{better:.0f}% / {worse:.0f}%")
                rows.append((error, cells))
            if rows:
                headers = ("Message type", "Decoded position error (m)")
                if target == "Ball":
                    headers += ("Decoded velocity error (m/cycle)",)
                headers += ("Seen now · better / worse", "Unseen · better / worse")
                tables.append(f"<h3>{escape(target)}</h3>"
                              + table(headers, [cells for _, cells in sorted(rows, key=lambda item: item[0])]))
        if not tables:
            return ""
        return ("<p>Lower decoded error means the payload itself was closer to fullstate. The last two columns "
                "show how often it beat or lost to the local reference by over 5 cm. The rest were within 5 cm. "
                "A dash means fewer than 100 comparable reports. Message types are sent in different situations, "
                "so this comparison includes the sender’s accuracy and choice of when to send.</p>" + "".join(tables))

    def prediction(self):
        labels = []
        reductions = []
        rows = []
        for label, metric, unit, scale in PREDICTIONS:
            p = self.result("next_cycle", "All", f"prediction_{metric}")
            b = self.result("next_cycle", "All", f"persistence_{metric}")
            if p is None or b is None:
                continue
            reduction = 100 * (1 - p / b) if b else np.nan
            labels.append(label)
            reductions.append(reduction)
            improvement = f"{reduction:+.1f}%" if math.isfinite(reduction) else "—"
            rows.append((label, unit, number(p * scale), number(b * scale), improvement))
        if not rows:
            return "<p>No paired next-cycle predictions were available.</p>"
        figure, ax = plt.subplots(figsize=(8.7, 3.2), layout="constrained")
        bars = ax.barh(labels, reductions, color=[COLORS[0] if val >= 0 else COLORS[2] for val in reductions])
        ax.bar_label(bars, labels=[f"{val:+.1f}%" if math.isfinite(val) else "" for val in reductions], padding=5)
        ax.axvline(0, color="#63798c", lw=1.4)
        ax.margins(x=0.15)
        ax.invert_yaxis()
        ax.set_xlabel("Error reduction compared with carrying forward")
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.spines[["top", "right", "left"]].set_visible(False)
        chart = image(figure, "Bars show how much the prediction reduces error compared with carrying forward the current estimate. A negative value means the prediction adds error.")
        return chart + table(("Quantity", "Unit", "Prediction error", "Carry-forward error", "Error reduction"), rows)


def legend(ax, columns):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=columns, frameon=False, fontsize=9)


def expandable(title, content):
    return f"<details><summary>{escape(title)}</summary>{content}</details>"


def style(ax, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", color="#dce6ea")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def kick_sensitivity():
    # Illustrative type-0 player from exp1. The ball is directly ahead.
    ball_distance = 0.65
    ball_speed = 0.50
    target_speed = 2.70
    target_distance = 20.0
    baseline_error = 0.0431
    needed_acceleration = target_speed - ball_speed

    def kick_rate(distance):
        surface_gap = distance - 0.30 - 0.085
        return 0.027 * (1 - 0.25 * surface_gap / 0.70)

    scenarios = (("4.31 cm", baseline_error), ("Half (2.155 cm)", baseline_error / 2), ("0 cm", 0))
    speed_rows = []
    speed_errors = []
    for label, error in scenarios:
        power = needed_acceleration / kick_rate(ball_distance + error)
        speed = ball_speed + power * kick_rate(ball_distance)
        speed_errors.append(speed - target_speed)
        speed_rows.append((label, f"{power:.2f}", f"{speed:.3f}", f"{speed - target_speed:+.3f}"))
    speed_table = table(("Overestimate of ball distance", "Kick power", "First-step speed (m/cycle)",
                         "Speed error (m/cycle)"), speed_rows)

    aim_error = math.atan2(baseline_error, target_distance)
    lateral_error = target_speed * math.sin(aim_error)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for ax, errors, ylabel in (
        (axes[0], speed_errors, "Speed overshoot (m/cycle)"),
        (axes[1], [100 * target_speed * math.sin(math.atan2(error, target_distance)) for _, error in scenarios],
         "First-cycle sideways miss (cm)"),
    ):
        bars = ax.bar([label for label, _ in scenarios], errors, color=COLORS[0], width=.6)
        ax.bar_label(bars, fmt="%.3g", padding=3)
        ax.margins(y=0.2)
        style(ax, "Assumed position error", ylabel)
    chart = image(figure, "Illustrative kick sensitivity. The two panels isolate different ways a position error can affect a kick.")
    return f"""<details><summary>How self-position error can affect a 2.7 m/cycle kick</summary>
<p>The server uses the <strong>true ball-to-player distance and angle</strong> to apply the kick.
Helios chooses its power using the <strong>estimated</strong> distance and angle.</p>
<p><strong>Kick rate</strong> = player kick power rate × (1 − 0.25 × |ball angle from body| ÷ 180°
− 0.25 × surface gap ÷ kickable margin). Surface gap is center-to-center distance minus the player and ball radii.</p>
<p><strong>Requested power</strong> = needed change in ball velocity ÷ estimated kick rate.
<strong>Applied acceleration</strong> = requested power × true kick rate in the kick direction, before server noise and limits.</p>
{chart}
<h3>If the self-position error changes the estimated distance to the ball</h3>
<p>Take a type-0 player with the ball {ball_distance:.2f} m straight ahead and already rolling toward the target
at {ball_speed:.2f} m/cycle. A first-step speed of {target_speed:.2f} m/cycle needs a {needed_acceleration:.2f} m/cycle
velocity increase. Suppose the ball’s estimated field position is correct, and the entire 4.31 cm self-position
error makes the player think the ball is farther away. Helios then asks for too much power.</p>
{speed_table}
<p>In this example, halving the distance error cuts the speed error from about 0.038 to 0.019 m/cycle.
Speeds are for the ball’s first movement, before decay.</p>
<h3>If the estimated player and ball positions shift together</h3>
<p>A fresh visual ball position can inherit the same field-position shift as the player. The player-to-ball
offset then stays the same, so this shift does not change the kick rate. Aiming at a fixed field target can
still change. With a sideways shift of 4.31 cm and a target {target_distance:.0f} m away, the aim changes by
about {math.degrees(aim_error):.2f}°, or {100 * lateral_error:.2f} cm sideways in the first cycle at
{target_speed:.2f} m/cycle. Halving the shift reduces that to about {50 * lateral_error:.2f} cm.</p>
<p class="hint">These examples isolate position error and assume correct ball velocity and body angle.
The 4.31 cm input is the overall exp1 self-position mean. Actual kicks also depend on ball and body-angle
estimates, server noise, and speed and power limits.</p>
<p class="hint">Code references. rcssserver <code>Player::kick</code> and <code>MPObject::_inc</code>,
librcsc <code>PlayerType::kickRate</code> and <code>WorldModel::localizeBall</code>,
and Helios <code>Body_KickOneStep::execute</code>.</p></details>"""


def write_report(output, logs, settings, coverage, rows):
    for row in rows:
        if row["metric"] not in SIGNED_METRICS:
            row["bias"] = None
    report = Report(rows)
    r = report
    name = escape(logs.name)
    side = coverage.get("observer_side", "both")
    scope = "" if side == "both" else f"<p><strong>Observers</strong> {side.capitalize()}-team players only. Both teams are included as observed targets.</p>"
    self_position = r.result("self", "All", "n_est_pos_d")
    self_card = f"{100 * self_position:.2f} cm" if self_position is not None else "—"
    cards = "".join(f"<div class='card'><strong>{value}</strong><span>{label}</span></div>" for value, label in (
        (f'{coverage["games"]:,}', "Games"),
        (f'{coverage["play_on_rows"] - coverage.get("missing_truth", 0):,}', "Agent decisions with truth"),
        (f'{coverage.get("object_rows", 0):,}', "On-field object samples"),
        (self_card, "Mean self position error"),
    ))
    settings_table = table(("Run setting", "Value"), ((escape(key), escape(value)) for key, value in settings.items()))
    head = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Denoise report · {name}</title>
<style>{STYLE}</style></head><body><main><header><div class="kicker">RoboCup 2D · Denoise experiment</div>
<h1>{name} · Play-on report</h1><p class="lede">Sensor error, track freshness, communication, and one-cycle prediction.</p>
{scope}
<div class="cards">{cards}</div>
<nav aria-label="Report sections"><a href="#self">Self</a><a href="#objects">Ball and players</a>
<a href="#view">View and age</a><a href="#hear">Hearing</a><a href="#prediction">Predictions</a>
<a href="#next">What to test next</a><a href="#method">Method</a>
<a href="metrics.csv">Detailed metrics (CSV)</a></nav></header>"""

    self_section = f"""<section id="self"><h2>Self localization</h2>
<p><strong>Fresh visual position</strong> is where the agent thinks it is based on what it just saw, such as flags and field lines.
<strong>Fresh visual looking direction</strong> is where it thinks it is looking, in degrees.
Both are measured before librcsc updates the world model. The paired tables compare the fresh visual and world-model
estimates on the same cycles.</p>
{r.self_detail("Fresh visual and world-model estimates on the same cycles", "self_paired_fresh",
               ("All",), "", VISUAL_COMPARISON, "Lower bars mean smaller error. Each pair uses the same cycles.")}
{r.self_detail("Fresh visual and world-model estimates by view width", "self_paired_by_view",
               VIEWS, "View width", VISUAL_COMPARISON, "Compare fresh visual and final error within each view width.")}
{r.self_detail("Self errors by view width", "self_view", VIEWS, "View width",
               (POSITION_CHART, VELOCITY_CHART, FACE_CHART), "Mean errors by the agent’s current view width. Helios chooses that width based on the game situation.")}
{r.self_detail("Self errors by position count", "self_position_age", AGES[:-1], "Cycles since last position fix",
               (POSITION_CHART,), "The mean position error at each age. View width and game situation can differ between ages.", line=True)}
{r.self_detail("Self velocity by velocity count", "self_velocity_age", AGES[:-1], "Velocity count (cycles)",
               (VELOCITY_CHART,), "How far the world-model velocity is from fullstate at each velocity count.", line=True)}
{r.self_detail("Self orientation by face count", "self_face_age", AGES[:-1], "Face count (cycles)",
               (("Orientation error (°)", 1, (("n_est_face", "Looking direction"), ("n_est_body", "Body angle"))),),
               "Compare looking-direction and body-angle error as the orientation estimate ages.", line=True)}
{r.self_detail("Landmark sensor error by true distance", "landmark_distance", DISTANCES, "True landmark distance (m)",
               (("Distance error (cm)", 100, (("n_dist", "Raw distance"),)),
                ("Bearing error (°)", 1, (("n_dir", "Raw bearing"),))),
               "Raw distance and bearing error for flags and goals, grouped by their true distance from the observer.", line=True)}
{kick_sensitivity()}
{expandable("Overall self metrics", r.show("self"))}</section>"""

    object_raw = r.object_distance_charts(
        (("n_dist", "Mean distance error (m)"), ("n_dir", "Mean bearing error (°)")), TARGETS,
        "Raw readings from see messages. Each point groups matched observations by their true distance.",
    )
    object_paired = r.metric_chart(
        "object_paired_fresh", TARGETS, "",
        (("Mean position error (m)", 1, (("n_ls_pos_d", "Fresh visual"), ("n_trk_pos_d", "World model"))),
         ("Mean velocity error (m/cycle)", 1, (("n_ls_vel_d", "Fresh visual"), ("n_trk_vel_d", "World model")))),
        "Each pair compares the visual estimate and final track against the same fullstate target on the same cycle.",
    )
    object_fields = r.object_distance_charts(
        (("unum_visible", "Uniform number available"), ("raw_velocity_available", "Raw motion fields available")),
        TARGETS[1:], "Percent of seen player observations with each field. A missing field needs to be inferred or left unknown.",
        rates=True,
    )

    objects_section = f"""<section id="objects"><h2>Ball and other players</h2>
<h3>Does the world model improve a fresh measurement?</h3>
{object_paired}
<p>Where the world-model bar is higher, check how the player observation was assigned to a track before changing the sensor model.
This comparison can reflect both player matching and fusion. Position and velocity use separate sets of paired samples.</p>
{expandable("How distance affects the raw measurement", object_raw)}
{expandable("Which player fields disappear with distance", object_fields)}</section>"""

    view_section = f"""<section id="view"><h2>View width and time since observation</h2>
<h3>How much does seeing the same target again help?</h3>
{r.reacquisition()}
<p>The gap between the lines shows how much the estimate changed when the target came back into view.
It shows where a model that predicts through missed observations has room to help.</p>
{expandable("How often each view gives an update and sees a target", r.view_tradeoff())}
<p class="hint">Visual age counts cycles since the last visual position update. A track’s position count can reset after hearing.</p>
</section>"""

    message_types = r.hearing_message_types()
    distance_risk = r.hearing_distance_risk()
    hear_section = f"""<section id="hear"><h2>When does hearing help?</h2>
<p>Compare the decoded position in a teammate message with the receiver’s local estimate. The report calls it
better or worse when the two errors differ by more than 5 cm. Both are scored against the same fullstate position.</p>
{r.hearing_candidate_outcomes()}
<ul>
<li><strong>Seen now.</strong> Compare with the receiver’s fresh visual position, before fusion. Player identity
must be visible for this comparison.</li>
<li><strong>Unseen.</strong> Compare the message with the ball forecast made on the preceding cycle, or with
the previous position of an identified player carried forward one cycle.</li>
</ul>
{r.hearing_effect_size()}
{expandable("Which message types carry useful positions?", message_types) if message_types else ""}
{expandable("When is a heard position more likely to be worse?", distance_risk) if distance_risk else ""}
{expandable("Do the final tracks with fresh reports also look better?", r.hearing()) if message_types else ""}
<p>These results test what would happen if you chose the reported position over the stated reference. Earlier
messages can already be part of that reference. To measure the effect of the actual fusion rule, log the same-cycle
estimate immediately before and after applying the message. Use <code>S0</code> to measure the overall effect of
communication on the run.</p>
</section>"""

    ball_forecast = r.metric_chart(
        "next_cycle_ball_visual_age", AGES[:-1], "Cycles since last visual ball position",
        (("Ball position error (m)", 1, (("prediction_ball_pos", "Prediction"), ("persistence_ball_pos", "Carry forward"))),
         ("Ball velocity error (m/cycle)", 1, (("prediction_ball_vel", "Prediction"), ("persistence_ball_vel", "Carry forward")))),
        "Compare both curves at the same age. A small gap shows where the current motion model adds little over carrying forward the estimate.",
        line=True,
    )

    prediction_section = f"""<section id="prediction"><h2>Predicted state versus the real next cycle</h2>
<p>The prediction is scored against fullstate exactly one cycle later. The carry-forward baseline uses the
current estimate on the same samples. Compare the ball velocity and position rows to see which the current motion model helps more.</p>
{r.prediction()}
{expandable("When does predicting the ball beat carrying its estimate forward?", ball_forecast)}
<p class="hint">Schema v2 supplies numeric next-cycle predictions for self position and body angle and for ball position and velocity. It does not log next self velocity or next opponent/teammate state.</p></section>"""

    next_steps = """<section id="next"><h2>What to test next</h2>
<ul>
<li><strong>Tracking through visual gaps.</strong> Use the before-and-after reacquisition plot to choose ages and
targets for a prediction model. <code>V1</code> tests the combined benefit of a wider cone and a <code>see</code>
every cycle. Separate cone and update-rate experiments are needed to distinguish their effects.</li>
<li><strong>Player matching and missing fields.</strong> Where a fresh player estimate is better than its final
track, test <code>F1</code> for identity and <code>F2</code> for the additional motion and orientation fields.
Compare position error and track availability together.</li>
<li><strong>Teammate fusion.</strong> Run <code>S0</code> against the baseline and compare stale tracks within the
same distance and age ranges. Log before and after heard updates to see which messages help.</li>
<li><strong>Fresh measurement correction.</strong> Use <code>N1</code> to isolate distance noise where the raw
distance plot shows the biggest errors. Direction and motion-field rounding remain.</li>
</ul></section>"""

    checks = table(("Coverage check", "Count"), (
        (label, f"{coverage.get(key, 0):,}") for key, label in (
            ("agent_logs", "Finalized agent logs"),
            ("missing_truth", "Play-on decisions excluded for missing truth"),
            ("skipped_decisions", "Skipped-decision placeholders across all modes"),
            ("seen_objects", "Eligible object rows marked seen"),
            ("ambiguous_raw_matches", "Seen rows excluded from raw error for ambiguous matching"),
            ("next_cycle_pairs", "Eligible consecutive agent decisions"),
            ("hear_logs", "Agent logs with hearing data"),
            ("landmark_logs", "Agent logs with landmark data"),
        )
    ))
    method = f"""<section id="method"><h2>How to read the results</h2>
<p>The report uses <code>play_on</code> samples with valid fullstate and gives every eligible sample equal weight.
Grouped plots omit groups with fewer than {MIN_PLOT_SAMPLES} measurements. The <a href="metrics.csv">detailed CSV</a>
includes sample counts, availability and additional breakdowns.</p>
<details><summary>Dataset and calculation details</summary>
<p>Source <code>{escape(str(logs))}</code>. Self uses one row per observer decision. Ball and players use one
row per observer, on-field object and decision. Missing measurements are excluded from the error average.
Means use absolute or Euclidean errors, and percentiles are approximate. Distance bins include the lower
bound and exclude the upper bound.</p>
<p>Raw player errors exclude ambiguous nearest-direction matches. Fused track results can include heuristic
nearest-truth associations. Ground-truth bins are for evaluation. Train with features the agent can access
and evaluate on held-out games.</p>
{checks}{settings_table}</details>
</section></main></body></html>"""

    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "run.json").write_text(json.dumps({"source": str(logs), "settings": settings, "coverage": coverage},
                                                indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(head + r.highlights() + self_section + objects_section + view_section + hear_section
                                        + prediction_section + next_steps + method, encoding="utf-8")

