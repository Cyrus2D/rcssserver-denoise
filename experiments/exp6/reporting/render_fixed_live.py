#!/usr/bin/env python3
"""Render the corrected Gaussian live batch and the first temporal comparison."""
import argparse
from pathlib import Path
import json
import shutil

import markdown
from render_report import Run, DISTANCES, num, pct, table, plot, plt

CSS = ("body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f7fafc;color:#1e293b}"
       "main{max-width:1100px;margin:auto;padding:28px 24px}img{display:block;width:100%;background:white}"
       "table{border-collapse:collapse;display:block;overflow:auto;background:white;margin:18px 0}"
       "td,th{border:1px solid #d5e2e8;padding:8px 11px}th{background:#e4eff1;text-align:left}"
       "pre{background:#edf4f7;padding:16px;overflow:auto}a{color:#075e71}")
NAMES = ("Exp3 quantized", "Exp6 original Gaussian", "Exp6 corrected Gaussian")
COLORS = ("#64748b", "#c66935", "#007d85")
FILTERS = {
    "visual":"Current visual position",
    "mean_two":"Average two positions",
    "velocity_two":"Average with one-cycle velocity prediction",
    "kf_pos":"Kalman, position only",
    "kf_vel_smooth":"Kalman + velocity, smooth",
    "kf_vel_medium":"Kalman + velocity, medium",
    "kf_vel_quick":"Kalman + velocity, quick",
}


def name(key):
    for suffix, label in (("_cap25"," + 25 cm cap"),("_cap50"," + 50 cm cap")):
        if key.endswith(suffix):
            return FILTERS[key[:-len(suffix)]] + label
    return FILTERS[key]


def write(out, title, text):
    out.mkdir(parents=True, exist_ok=True)
    (out/"README.md").write_text(text,encoding="utf-8")
    html = markdown.markdown(text,extensions=["tables"])
    (out/"report.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+
        '</title><style>'+CSS+'</style></head><body><main>'+html+'</main></body></html>',encoding="utf-8")


def render_live(quantized, original, fixed, fixed_file, out):
    runs = (quantized,original,fixed)
    count = fixed.data["games"]
    cov = fixed.coverage
    assert fixed.data["localizer"]=={"true":22*count} and fixed.data["observers"]==22*count
    assert all(cov[k]==cov["play"] for k in ("truth","one_see","see","full_view","high_quality"))
    assert not cov["teammate_payload"] and not cov["skipped"]
    assert fixed.metric("fresh_wm")[0]==cov["known_ranges"] and fixed.mean("fresh_wm")==1
    scores = fixed.data["scores"]
    assert len(scores)==count and all(s["server_verified"] for s in scores)
    left,right = (sum(s[key] for s in scores) for key in ("left","right"))
    wins,draws = (sum(test(s["left"],s["right"]) for s in scores) for test in (
        lambda x,y:x>y,lambda x,y:x==y))
    results = table(("Mean error",*NAMES), [
        (label,*(num(r.mean(metric,target)*scale,digits) for r in runs))
        for label,target,metric,scale,digits in [
            ("Raw player range MAE (cm)","all","raw_range",100,2),
            ("Raw relative player position (cm)","all","raw_position",100,2),
            ("Signed player range error after localization (cm)","all","signed_localized_range",100,3),
            ("Player WM position (cm)","all","track_position",100,2),
            ("Player WM velocity (m/cycle)","all","track_velocity",1,4),
            ("Ball WM position (cm)","ball","track_position",100,2),
            ("Ball WM velocity (m/cycle)","ball","track_velocity",1,4),
            ("Self WM position (cm)","self","self_position",100,2),
            ("Self WM looking direction (°)","self","self_face",1,3),
        ]])
    ranges = table(("True distance (m)", *NAMES), [
        (d,*(num(r.mean("track_position",distance=d)*100) for r in runs))
        for d in DISTANCES])
    out.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    for ax,title,key in [(axes[0],"Raw range error","raw_range"),(axes[1],"Global WM position error","track_position")]:
        plot(ax,title,"Mean error (cm)",[
            (label,[100*r.mean(key,distance=d) for d in DISTANCES],color)
            for label,r,color in zip(NAMES,runs,COLORS)])
    fig.savefig(out/"three-live-modes.svg")
    fig.savefig(out/"three-live-modes.png",dpi=120)
    plt.close(fig)
    text = f"""# Exp6 · corrected Gaussian in live games

**{count} completed matches** were found in AutoTest2D exp6logs3 and the
matching player-log directory exp6-logs3. The original Exp6 Gaussian run and
the archived Exp3 quantized run each contain 200 games.

Correcting the range lookup lowered player WM error from
**{100*original.mean('track_position'):.2f} to {100*fixed.mean('track_position'):.2f} cm** across the two
Gaussian live batches. The corrected mean is consistent with the separate
fixed-pose replay estimate of 67.48 cm. These are independently played matches.

All {fixed.data['observers']:,} observer logs finalized with corrected Gaussian
localization enabled. Every one of the {cov['play']:,} PlayOn decisions logged a
single 360° visual update and same-cycle reference truth, with no skipped
decision placeholders or teammate payloads. All {cov['known_ranges']:,} identified
player positions were fresh. All {count} match scores agree with AutoTest2D
({left}–{right} goals, {wins}/{draws}/{count-wins-draws} left wins/draws/losses).
Both sides used the same configuration.

{results}

![Raw and WM errors by distance for the three live configurations](three-live-modes.svg)

## Player WM error by true distance, cm

{ranges}

All means use equal weight per eligible observer/object/cycle in PlayOn.
The raw range comparison uses identified seen players. The WM means use
available valid estimates with reference truth. The Exp3 archive supports
comparisons for players. A dash marks an unavailable archival measure.

[Baseline and quantization comparison](../report.html) ·
[Fixed-pose replay](../range-replay/report.html) · [Metrics and match scores](metrics.json) ·
[Temporal denoising study](../temporal/report.html)

    exp=rcssserver-denoise/experiments/exp6
    python3 "$exp/reporting/analyze_logs.py" logs/exp6-logs3 logs/exp6-analysis/fixed-exp6logs3 --require-corrected --autotest AutoTest2D/out/exp6logs3
"""
    write(out,"Exp6 corrected Gaussian live results",text)
    shutil.copy2(fixed_file,out/"metrics.json")


def render_temporal(source,out,confirmation):
    result = json.loads(source.read_text())
    rows = result["test"]
    winners = result["selected"]
    ci = result["test_change95_cm"]
    nvalid = len(result["split"]["validation"])
    ntest = len(result["split"]["test"])

    def lookup(target,band,method):
        return next(r for r in rows if (r["target"],r["band"],r["method"]) == (target,band,method))

    def candidate_rows(target):
        candidates = [r for r in rows if r["target"]==target and r["band"]=="under50"]
        assert len({r["n"] for r in candidates})==1
        return table(("Method","Mean cm","Gain cm","Worsened >25 cm","Worsened >50 cm","Max added cm"), [
            (name(r["method"]),num(r["error_cm"]),num(r["baseline_cm"]-r["error_cm"]),
             pct(r["worsened_25cm"]),pct(r["worsened_50cm"]),num(r["max_added_cm"]))
            for r in sorted(candidates,key=lambda row:(row["method"]!="visual",row["error_cm"]))])

    choices=[]
    for target in ("player","ball"):
        for criterion,method in winners[target].items():
            r=lookup(target,"under50",method)
            low,high=ci[target][criterion]
            choices.append((target.title(), "Lowest mean" if criterion=="mean" else "25 cm cap",
                            name(method),num(r["baseline_cm"]),num(r["error_cm"]),
                            f"[{low:+.2f}, {high:+.2f}]",pct(r["worsened_25cm"])))
    selected=table(("Target","Selection","Validation choice","Visual cm","Test cm",
                    "95% change interval, cm","Worsened >25 cm"),choices)

    player=winners["player"]
    ball=winners["ball"]
    eval_bands=("0-3","3-10","10-20","20-30","30-40","40-60","60+")
    out.mkdir(parents=True,exist_ok=True)
    fig, axes=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    for ax,target,methods in zip(axes,("player","ball"),(player,ball)):
        keys=["visual",methods["mean"],methods["capped25"]]
        plot(ax,target.title()+" position","Mean error (cm)",[
            (label,[lookup(target,d,key)["error_cm"] for d in eval_bands],color)
            for label,key,color in zip(("Visual","Validation pick","25 cm cap"),keys,COLORS)])
        ax.set_xlabel("True observer-to-target distance (m)")
    fig.savefig(out/"temporal-error.svg")
    plt.close(fig)
    close=table(("Target below 3 m","Visual cm","Best mean filter cm","25 cm cap cm",
                 "Best mean worsened >25 cm"), [
        (target.title(),num(lookup(target,"0-3","visual")["error_cm"]),
         num(lookup(target,"0-3",winners[target]["mean"])["error_cm"]),
         num(lookup(target,"0-3",winners[target]["capped25"])["error_cm"]),
         pct(lookup(target,"0-3",winners[target]["mean"])["worsened_25cm"]))
        for target in ("player","ball")])

    followup = ""
    if confirmation:
        confirmed = json.loads(confirmation.read_text())
        previous_test = {r["game"] for r in result["split"]["test"]}
        new_test = {r["game"] for r in confirmed["split"]["test"]}
        validation = {r["game"] for r in confirmed["split"]["validation"]}
        assert len(new_test)==40 and previous_test.isdisjoint(new_test) and validation.isdisjoint(new_test)
        comparison, nearby = [], []
        for target in ("player","ball"):
            method = confirmed["selected"][target]["near_safe"]
            by_band = {r["band"]:r for r in confirmed["test"] if r["target"]==target and r["method"]==method}
            row, close_row = by_band["under50"], by_band["0-3"]
            assert row["worsened_25cm"]==0 and row["max_added_cm"]<=25.+1e-7
            low, high = confirmed["test_change95_cm"][target]["near_safe"]
            comparison.append((target.title(), num(row["baseline_cm"]), num(row["error_cm"]),
                               f"[{low:+.2f}, {high:+.2f}]", pct(row["worsened_25cm"])))
            nearby.append((target.title(), f"{close_row['n']:,}", num(close_row["baseline_cm"]),
                           num(close_row["error_cm"]), num(close_row["max_added_cm"])))
        followup = f"""## Confirmation on 40 additional games

The first test found that filtering could worsen close positions, especially
for the ball. The follow-up uses the current visual position when the measured
range is below 5 m and caps position corrections at 25 cm elsewhere.
Validation chose the position-only filter for players and the medium
position-and-velocity filter for the ball. The same validation games were
used for selection. The 40 confirmation games differ from the first 40 test games.

{table(("Target under 50 m","Visual cm","Filtered cm","95% change interval, cm",
        "Worsened >25 cm"),comparison)}

{table(("Target under 3 m","Rows","Visual cm","Filtered cm","Max added cm"),nearby)}

Distance slices use true distance for evaluation. The gate uses measured
Gaussian range. The 25 cm cap bounds added position error relative to the
visual estimate for each sample.

[Confirmation split and metrics](confirmation-metrics.json)

"""
    text=f"""# Exp6 · first Gaussian temporal denoising study

The corrected visual position is the baseline. We compared two-frame
averages with Kalman filters that combine previous motion and the current
measured position. Player and ball candidates are selected separately.
The evaluation covers {ntest} held-out games after choosing settings on
{nvalid} other games. Each game contributes four observers with rotating
uniform numbers, balanced between the two sides.

{selected}

A negative change in the interval means lower position error than the visual
baseline. The intervals resample whole matches. A 25 cm cap limits how far
the output can move from the visual position in each cycle. By the triangle
inequality, it also limits added position error to 25 cm.

![Held-out accuracy by distance for the frozen picks](temporal-error.svg)

## Close targets below 3 m

{close}

Distance slices use reference truth for analysis. These slices show how the
overall improvements vary near the observer. The filters use measured range
and pose during inference.

## All candidates under 50 m on the held-out games

### Players

{candidate_rows("player")}

### Ball

{candidate_rows("ball")}

“Worsened” measures the increase in Euclidean error compared with the current
visual position for the same sample.

## How the filters work

- **Two-frame averages.** Average the previous and current observed positions.
  The velocity version advances the previous position by the previous direct
  visual velocity first, and uses the current fix when that velocity is missing.
- **Kalman filters.** Predict the current position with a constant-velocity
  state. Trust the current measurement less along the viewing ray when
  estimated Gaussian range noise is large. Including direct visual velocity
  offers another update when both motion fields are present. Filters differ
  in how quickly they allow motion to change. The ball uses larger process
  and velocity uncertainty.
- **Caps.** Limit the output correction to 25 or 50 cm. Prediction history
  is updated from the underlying filter state. The cap applies to evaluation
  output each cycle.

Inputs use current and past observations only. Track history restarts when
a consecutive PlayOn observation is missing. Truth is read for scoring after
calculating the outputs, and supplies the distance slices. Validation
chooses the lowest mean error under 50 m, once among all candidates and once
among the 25 cm capped candidates. Test trajectories are the recorded games
played with corrected Gaussian localization.

The reported variance settings are working estimates for filtering. They
approximate correlated position and velocity measurements. This study
measures current position error on recorded trajectories. Live decisions
and gameplay need a separate evaluation.

{followup}[Live corrected Gaussian audit](../fixed-live/report.html) ·
[Split and numerical results](metrics.json)

    exp=rcssserver-denoise/experiments/exp6
    .venv-exp5-s1/bin/python "$exp/reporting/analyze_temporal.py" logs/exp6-logs3 logs/exp6-analysis/temporal-exp6logs3
"""
    write(out,"Exp6 first Gaussian temporal denoising study",text)
    shutil.copy2(source,out/"metrics.json")
    if confirmation:
        shutil.copy2(confirmation,out/"confirmation-metrics.json")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("quantized","original","fixed","temporal","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--confirmation", type=Path)
    a=p.parse_args()
    render_live(Run(a.quantized),Run(a.original),Run(a.fixed),a.fixed,a.output/"fixed-live")
    render_temporal(a.temporal,a.output/"temporal",a.confirmation)
