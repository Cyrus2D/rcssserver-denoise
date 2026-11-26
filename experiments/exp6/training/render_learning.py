#!/usr/bin/env python3
"""Report held-out results of the first Exp6 residual GRUs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"reporting"))
from render_fixed_live import write, table, num, pct, plt

NAMES = {
    "visual":"Current visual position",
    "kalman":"Kalman",
    "kalman_guard":"Kalman with 5 m gate and 25 cm cap",
    "gru":"Residual GRU",
    "gru_guard":"Residual GRU with 5 m gate and 25 cm cap",
}
SET_NAMES = {"fresh_test":"New 20-game test", "previous_test":"Earlier 40-game benchmark"}


def render(source,out):
    result = json.loads(source.read_text())
    assert set(result["targets"]) == {"player","ball"}
    split = result["split"]
    cohorts = [set(games) for games in split.values()]
    assert sum(map(len,cohorts))==len(set.union(*cohorts))
    counts = result["data_counts"]
    lots = table(("Cohort","Games","Player windows","Ball windows"),[
        (label,len(split[key]),f"{counts[key+'_player']:,}",f"{counts[key+'_ball']:,}")
        for key,label in [("train","Training"),("validation","Validation"),
                          ("fresh_test","New test"),("previous_test","Earlier benchmark")]])
    out.mkdir(parents=True,exist_ok=True)
    evaluated = []
    distances = ("under3","3-10","10-20","20-35","35-50")
    distance_labels = ("0–3","3–10","10–20","20–35","35–50")
    fig, axes = plt.subplots(1,2,figsize=(11,4.1),layout="constrained")
    for ax,target in zip(axes,("player","ball")):
        rows = {(r["distance"],r["method"]):r for r in result["targets"][target]["fresh_test"]["rows"]}
        for method,color in (("visual","#64748b"),("kalman_guard","#c66935"),
                             ("gru_guard","#007d85"),("gru","#235aa8")):
            ax.plot(distance_labels,[rows[d,method]["error_cm"] for d in distances],
                    "o-",color=color,label=NAMES[method])
        ax.set(xlabel="True observer-to-target distance (m)",ylabel="Mean position error (cm)",
               title=target.title()+" on new test games",ylim=(0,None))
        ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    fig.savefig(out/"learned-by-distance.svg")
    fig.savefig(out/"learned-by-distance.png",dpi=120)
    plt.close(fig)
    for split_name,designation in SET_NAMES.items():
        sections = []
        for target in ("player","ball"):
            data = result["targets"][target][split_name]
            metrics = {r["method"]:r for r in data["rows"] if r["distance"]=="under50"}
            count = metrics["visual"]["n"]
            assert len({r["n"] for r in metrics.values()})==1
            assert all(metrics[k]["worsened_25cm"]==0 and metrics[k]["max_added_cm"]<=25.001
                       for k in ("kalman_guard","gru_guard"))
            short = {r["method"]:r for r in data["rows"] if r["distance"]=="under3"}
            ci = data["guarded_gru_minus_guarded_kalman95_cm"]
            sections.append(f"""### {target.title()}

{count:,} paired samples below 50 m.

{table(("Method","Mean cm","Gain cm","Worsened >25 cm","Worsened >50 cm","Max added cm"),[
    (NAMES[key],num(metrics[key]["error_cm"]),num(metrics[key]["baseline_cm"]-metrics[key]["error_cm"]),
     pct(metrics[key]["worsened_25cm"]),pct(metrics[key]["worsened_50cm"]),num(metrics[key]["max_added_cm"]))
    for key in NAMES])}

Difference in mean between guarded GRU and guarded Kalman, with a whole-game
95% bootstrap interval **[{ci[0]:+.2f}, {ci[1]:+.2f}] cm**. Negative favors the GRU.

Under 3 m, the mean was {short["visual"]["error_cm"]:.2f} cm for the current
visual position, {short["gru"]["error_cm"]:.2f} cm for the uncapped GRU and
{short["gru_guard"]["error_cm"]:.2f} cm with the gate and cap.
""")
        evaluated.append(f"## {designation}\n\n"+"\n".join(sections))

    configs = result["config"]
    training = table(("Target","Parameters","Selected epoch","Windows per epoch",
                      "Validation capped error (cm)","Validation uncapped error (cm)"),[
        (t.title(),f"{r['training']['parameters']:,}",r["training"]["best_epoch"],
         f"{epoch['train_windows']:,}",num(epoch["guard_cm"]),num(epoch["raw_cm"]))
        for t,r in result["targets"].items() for epoch in r["training"]["history"]
        if epoch["epoch"]==r["training"]["best_epoch"]])
    text = f"""# Exp6 · Learn a correction from Gaussian observation history

The dataset now combines 100 previously logged corrected-Gaussian games and
200 new games. The new set supplied all training and validation rows, plus
a separate 20-game test. We also evaluated on the frozen 40-game benchmark
from the earlier physical-filter experiment.

{lots}

## Model and inputs

We trained separate {configs['hidden']}-unit PyTorch GRUs for players and the
ball on {result['gpu'] or result['device']}. Every example has up to
{result['history']} consecutive PlayOn observations ending at the current
cycle. A missing cycle or an interruption resets history. Before the first
eight observations the window is padded and carries a validity flag.

The visible inputs describe changes in visual position, the measured range
and estimated noise, the current and past viewing rays, measured velocity
when available, teammate flag, and the Kalman position correction. Vectors are
expressed along and across the current viewing direction. The GRU learns
how much of the Kalman correction to trust and adds a residual in both axes.
Fullstate provides the training target and the evaluation distance slices.

Each epoch randomly samples windows within each selected observer stream.
Observer numbers rotate across games. Validation chose one checkpoint per target by guarded
mean error plus 0.20 times uncapped mean error. The training loss emphasizes
close and medium distances and includes both guarded and uncapped position
error. The checkpoint can correct a position only. The ground truth of the
target velocity or body is outside this experiment.

{training}

![Position errors by distance on the new test](learned-by-distance.svg)

The score tables use the same sampled endpoints for all estimates. Test
endpoints use fixed cycle/identity strides, so physical-reference numbers
can differ from the earlier full-rate report. Each sample has equal weight.
The physical comparisons use the validated position-only Kalman for players
and the velocity-informed Kalman for the ball. Guarded variants keep visual
position below **5 m measured range** and cap other corrections at **25 cm**.
The cap bounds the added position error relative to the same visual estimate.

{"".join(evaluated)}

The evaluation measures position quality on recorded PlayOn observations.
Match outcomes require a separate test on live trajectories.

[Full metrics, validation curves and game split](metrics.json) ·
[Physical temporal comparison](../temporal/report.html) ·
[Corrected Gaussian live audit](../fixed-live/report.html)

    exp=rcssserver-denoise/experiments/exp6
    data=logs/exp6-analysis/learning-300/data
    run=logs/exp6-analysis/learning-300/gru40
    .venv-exp5-s1/bin/python "$exp/training/build_data.py" logs/exp6-logs3 "$exp/reports/fixed-live/metrics.json" "$exp/reports/temporal/confirmation-metrics.json" AutoTest2D/out/exp6logs3 "$data"
    .venv-exp5-s1/bin/python "$exp/training/train.py" "$data" "$run"
    .venv-exp5-s1/bin/python "$exp/training/render_learning.py" "$run/metrics.json" "$exp/reports/learning"
"""
    write(out,"Exp6 Gaussian residual GRU",text)
    shutil.copy2(source,out/"metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    render(args.metrics,args.output)
