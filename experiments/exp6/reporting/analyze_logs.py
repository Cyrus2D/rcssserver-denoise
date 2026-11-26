#!/usr/bin/env python3
"""Pool Exp6 logs using the same player masks and distance bins as Exp3."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess

DISTANCES = ("0–3", "3–10", "10–20", "20–30", "30–40", "40–60", "60+")
SERVER = dict(synch_mode="true", full_view="true", see_no_fade_unum="true",
              see_no_fade_all="false", see_noise_free="false", see_no_quantization="false",
              say_msg_size="0", fullstate_l="true", fullstate_r="true")
RATES = dict(dist_noise_rate=.0125, focus_dist_noise_rate=.0125,
             land_dist_noise_rate=.00125, land_focus_dist_noise_rate=.00125)
TABLES = ("agent_cycle", "objects", "landmarks", "hear")
SLOTS = {f"{side}_{n}" for side in "lr" for n in range(1, 12)}


def collect(game, output, scanner, exe_hash, score_dirs, require_corrected):
    files = sorted(game.glob("*/meta.json"))
    if {m.parent.name for m in files} != SLOTS:
        raise ValueError(f"{game}: expected 22 player logs")
    fingerprint = hashlib.sha256((str(game) + exe_hash).encode())
    scores = set()
    order, localizer = Counter(), Counter()
    for file in files:
        raw = file.read_bytes()
        fingerprint.update(raw)
        meta = json.loads(raw)
        cfg = meta["config"]
        if (meta["schema_version"] != 2 or not meta["final"] or meta["status"] != "ok"
                or cfg["use_fullstate"] or not cfg["debug_fullstate"] or not cfg["gaussian_see"]
                or cfg["version"] != 19 or not cfg["denoise_log_landmarks"]):
            raise ValueError(f"{file}: expected complete Gaussian schema 2 logs with reference fullstate")
        active = cfg.get("gaussian_range_localization")
        if require_corrected and active is not True:
            raise ValueError(f"{file}: corrected Gaussian localizer was not active")
        localizer["legacy" if active is None else str(active).lower()] += 1
        params = meta["server_param"]
        for key, expected in SERVER.items():
            if params.get(key) != expected:
                raise ValueError(f"{file}: {key}={params.get(key)}, expected {expected}")
        for key, expected in RATES.items():
            if float(params[key]) != expected:
                raise ValueError(f"{file}: unexpected {key}={params[key]}")
        order[params["randomize_player_commands"]] += 1
        scores.add((meta["score_left"], meta["score_right"]))
        for name in TABLES:
            stat = file.with_name(name + ".csv.gz").stat()
            fingerprint.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    if len(scores) != 1:
        raise ValueError(f"{game}: observer scores disagree: {scores}")
    score = list(scores.pop())
    server_score = None
    for folder in score_dirs:
        file = folder / "result.d" / game.name
        if file.is_file():
            scores = re.findall(r"Score:\s*(\d+)\s*-\s*(\d+)", file.read_text(errors="replace"))
            if not scores:
                raise ValueError(f"{file}: missing server score")
            server_score = list(map(int, scores[-1]))
            if server_score != score:
                raise ValueError(f"{file}: server and observer scores disagree")
            break
    if score_dirs and server_score is None:
        raise ValueError(f"{game}: missing AutoTest2D server result")

    cached = output / (game.name + ".json")
    digest = fingerprint.hexdigest()
    if cached.is_file():
        previous = json.loads(cached.read_text())
        if previous.get("source_hash") == digest:
            previous["server_score_verified"] = server_score is not None
            previous["localizer"] = dict(localizer)
            return previous
    result = subprocess.run([str(scanner), str(game), *map(str, RATES.values())],
                            check=True, stdout=subprocess.PIPE, text=True)
    coverage, metrics = {}, []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "coverage":
            coverage[fields[1]] = int(fields[2])
        else:
            target, index, metric, count, total, squares = fields
            metrics.append((target, DISTANCES[int(index)], metric, int(count), float(total), float(squares)))
    record = dict(game=game.name, source_hash=digest, coverage=coverage, metrics=metrics,
                  score=score, order=dict(order), localizer=dict(localizer), server_score_verified=server_score is not None)
    temp = cached.with_suffix(".tmp")
    temp.write_text(json.dumps(record))
    temp.replace(cached)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--autotest", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-corrected", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit nonnegative")
    logs = args.logs.resolve()
    games = sorted({p.parent.parent for p in logs.glob("*/*/meta.json")})
    if args.limit:
        games = games[:args.limit]
    if not games:
        raise ValueError(f"{logs}: no games")
    output = args.output.resolve()
    cache = output / "games"
    cache.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("scan_logs.cpp")
    scanner = output / "scan_logs"
    subprocess.run(["g++", "-O3", "-std=c++17", str(source), "-lz", "-o", str(scanner)], check=True)
    exe_hash = hashlib.sha256(source.read_bytes() + source.with_name("csv_reader.h").read_bytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect, g, cache, scanner, exe_hash, args.autotest, args.require_corrected) for g in games]
        results = []
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i == 1 or i % 10 == 0 or i == len(games):
                print(f"Exp6: {i}/{len(games)} games scanned", flush=True)
    coverage = Counter(teammate_payload=0, skipped=0, focus_nonzero=0)
    order, localizer = Counter(), Counter()
    pooled = defaultdict(lambda: [0, 0., 0.])
    scores = []
    for r in sorted(results, key=lambda item: item["game"]):
        coverage.update(r["coverage"])
        order.update(r["order"])
        localizer.update(r["localizer"])
        for target, distance, metric, n, total, squares in r["metrics"]:
            row = pooled[target, distance, metric]
            row[0] += n
            row[1] += total
            row[2] += squares
        scores.append(dict(game=r["game"], left=r["score"][0], right=r["score"][1],
                           server_verified=r["server_score_verified"]))
    metrics = [dict(target=t, distance=d, metric=m, n=n, mean=total/n, mean_square=squares/n)
               for (t,d,m),(n,total,squares) in sorted(pooled.items())]
    summary = dict(source=str(logs), games=len(results), observers=len(results)*22, settings=SERVER,
                   rates=RATES, command_order=dict(order), localizer=dict(localizer), coverage=dict(coverage), metrics=metrics, scores=scores)
    (output/"metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()

