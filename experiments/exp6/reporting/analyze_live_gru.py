#!/usr/bin/env python3
"""Audit the live Exp6 player denoiser, pooling paired errors over PlayOn rows."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from analyze_logs import RATES, SERVER, SLOTS


def read_metadata(game, model_side=None):
    files = sorted(game.glob("*/meta.json"))
    if {path.parent.name for path in files} != SLOTS:
        raise ValueError(f"{game}: expected all 22 observer streams")
    scores = set()
    for path in files:
        meta = json.loads(path.read_bytes())
        cfg = meta["config"]
        side = path.parent.name[0]
        if (not meta["final"] or meta["status"] != "ok" or meta["game_id"] != game.name
                or meta["side"] != side or not cfg["gaussian_range_localization"]
                or not cfg["gaussian_see"] or cfg["use_fullstate"] or not cfg["debug_fullstate"]):
            raise ValueError(f"{path}: invalid Gaussian reference log")
        if bool(cfg.get("player_position_corrector")) != (side == model_side):
            raise ValueError(f"{path}: unexpected model configuration")
        params = meta["server_param"]
        for key, value in {**SERVER, "randomize_player_commands": "false"}.items():
            if params.get(key) != value:
                raise ValueError(f"{path}: unexpected server setting {key}={params.get(key)}")
        for key, value in RATES.items():
            if float(params[key]) != value:
                raise ValueError(f"{path}: unexpected noise rate {key}={params.get(key)}")
        scores.add((meta["score_left"], meta["score_right"]))
    if len(scores) != 1:
        raise ValueError(f"{game}: observer scores disagree")
    return scores.pop()


def audit_game(game, server_dir, scanner, output, build_hash):
    score = read_metadata(game, "l")
    server_file = server_dir / game.name
    matches = re.findall(r"Score:\s*(\d+)\s*-\s*(\d+)", server_file.read_text(errors="replace"))
    if not matches or tuple(map(int, matches[-1])) != score:
        raise ValueError(f"{server_file}: server result differs from observer scores")
    key = hashlib.sha256(f"{build_hash}:{game}".encode())
    for slot in sorted(SLOTS):
        for name in ("objects.csv.gz", "agent_cycle.csv.gz"):
            path = game / slot / name
            st = path.stat()
            key.update(f"{slot}:{name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    digest = key.hexdigest()
    cache = output / f"{game.name}.json"
    if cache.is_file():
        record = json.loads(cache.read_bytes())
        if record["fingerprint"] == digest and record["score"] == list(score):
            return record

    raw = subprocess.check_output([scanner, str(game)], text=True)
    audits = {}
    metrics = []
    for line in raw.splitlines():
        kind, side, *fields = line.split("\t")
        if kind == "audit":
            audits[side] = [*map(int, fields[:-1]), float(fields[-1])]
        else:
            target, band, n, applied, stock, decision, better1, worse1, worse5, worse10, worse25, max_added = fields
            metrics.append([side, target, band, int(n), int(applied), float(stock), float(decision),
                            int(better1), int(worse1), int(worse5), int(worse10), int(worse25), float(max_added)])
    record = dict(game=game.name, fingerprint=digest, score=list(score), audit=audits, metrics=metrics)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(cache)
    return record


def match_stats(scores):
    n = len(scores)
    goals = [l-r for l, r in scores]
    decisions = Counter((diff > 0) - (diff < 0) for diff in goals)
    wins, draws, losses = (decisions[k] for k in (1,0,-1))
    played = wins + losses
    z = 1.95996398454
    p, win_ci, exact = None, None, None
    if played:
        p = wins / played
        center = (p + z*z/(2*played)) / (1 + z*z/played)
        margin = z * math.sqrt(p*(1-p)/played + z*z/(4*played*played)) / (1 + z*z/played)
        win_ci = [center-margin, center+margin]
        exact = min(1., 2 * sum(math.comb(played,k) for k in range(min(wins,losses)+1)) / 2**played)
    points = [3 if diff > 0 else -3 if diff < 0 else 0 for diff in goals]

    def mean_and_ci(values):
        mean = statistics.mean(values)
        if n == 1:
            return [mean, None, None]
        se = statistics.stdev(values) / math.sqrt(n)
        return [mean, mean-z*se, mean+z*se]

    return dict(games=n, wins=wins, draws=draws, losses=losses, decisive_win_rate=p,
                decisive_win_rate_wilson95=win_ci, exact_50pct_p=exact,
                goals=[sum(x[i] for x in scores) for i in (0,1)],
                points=[3*wins+draws, 3*losses+draws], goal_difference_mean_ci95=mean_and_ci(goals),
                point_difference_mean_ci95=mean_and_ci(points))


def summarize(results):
    metrics = {}
    audit = {}
    improved = 0
    for record in results:
        for side, fields in record["audit"].items():
            total = audit.setdefault(side, [0,0,0,0,0,0.])
            for i, value in enumerate(fields[:-1]):
                total[i] += value
            total[-1] = max(total[-1], fields[-1])
        for side, target, band, *fields in record["metrics"]:
            total = metrics.setdefault((side,target,band), [0,0,0.,0.,0,0,0,0,0,0.])
            for i, value in enumerate(fields[:-1]):
                total[i] += value
            total[-1] = max(total[-1], fields[-1])
            improved += side == "l" and target == "player" and band == "under50" and fields[3] < fields[2]
    fields = ("corrections","out_of_policy","over_cap","at_least_5m_not_applied","under_5m_not_applied","max_shift_m")
    audit = {side:dict(zip(fields, t)) for side,t in audit.items()}
    rows = []
    for (side,target,band), values in sorted(metrics.items()):
        n,applied,before,after,better1,worse1,worse5,worse10,worse25,max_added = values
        rows.append(dict(side=side,target=target,band=band,n=n,applied=applied,
                         stock_cm=100*before/n,decision_cm=100*after/n,improvement_pct=100*(1-after/before) if before else 0.,
                         better1=better1,worse1=worse1,worse5=worse5,worse10=worse10,worse25=worse25,
                         max_added_cm=100*max_added))
    return audit, rows, improved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("autotest_result_dir", type=Path, help="AutoTest2D result.d")
    parser.add_argument("baseline_logs", type=Path, help="Gaussian stock-vs-stock batch")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    logs = args.logs.resolve()
    games = sorted({p.parent.parent for p in logs.glob("*/*/meta.json")})
    baseline = sorted({p.parent.parent for p in args.baseline_logs.glob("*/*/meta.json")})
    if not games or not baseline:
        parser.error("both the live and baseline batches must contain games")
    output = args.output.resolve()
    cache = output/"games"
    cache.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).with_name("scan_live_gru.cpp")
    build_hash = hashlib.sha256(code.read_bytes() + code.with_name("csv_reader.h").read_bytes()).hexdigest()
    scanner = output/"scan_live_gru"
    subprocess.run(["g++","-O3","-std=c++17","-Wall","-Wextra",str(code),"-lz","-o",str(scanner)], check=True)
    print(f"Scanning {len(games)} games and {22*len(games):,} observers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_game, g, args.autotest_result_dir, scanner, cache, build_hash)
                   for g in games]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) == 1 or len(results) % 10 == 0:
                print(f"{len(results)}/{len(games)} games checked",flush=True)

    results.sort(key=lambda record:record["game"])
    audit, metrics, improved = summarize(results)
    report = dict(source=str(logs), baseline_source=str(args.baseline_logs.resolve()),
                  live_match=match_stats([r["score"] for r in results]),
                  baseline_match=match_stats([read_metadata(g) for g in baseline]),
                  games_with_paired_gain_under50=improved, audit=audit, metrics=metrics,
                  scores=[dict(game=r["game"],left=r["score"][0],right=r["score"][1]) for r in results])
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
