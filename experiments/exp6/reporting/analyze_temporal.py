#!/usr/bin/env python3
"""Select and score causal temporal range/position baselines on separate games."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np

SEED = 60623


def scan(game, scanner, cache, observer_slots, near_safe):
    inputs = [game/slot for slot in observer_slots]
    digest = hashlib.sha256((str(game)+str(near_safe)).encode())
    for path in inputs:
        meta = json.loads((path/"meta.json").read_text())
        if not meta["final"] or meta["status"] != "ok" or not meta["config"].get("gaussian_range_localization"):
            raise ValueError(f"{path}: need finalized corrected Gaussian metadata")
        for name in ("objects.csv.gz","agent_cycle.csv.gz"):
            stat=(path/name).stat()
            digest.update(f"{path.name}/{name}/{stat.st_size}/{stat.st_mtime_ns}".encode())
    cache_path = cache/(game.name+".json")
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if saved["input"] == digest.hexdigest():
            return saved
    options = ["--near-safe"] if near_safe else []
    text = subprocess.check_output([str(scanner),*options,*map(str,inputs)],text=True)
    rows = []
    for line in text.splitlines():
        target,band,method,count,raw,error,better,worse25,worse50,max_added=line.split("\t")
        rows.append([target,band,method,int(count),float(raw),float(error),
                     int(better),int(worse25),int(worse50),float(max_added)])
    result = dict(input=digest.hexdigest(),game=game.name,observers=observer_slots,rows=rows)
    cache_path.write_text(json.dumps(result))
    return result


def aggregate(games):
    result = defaultdict(lambda:[0,0.,0.,0,0,0,0.])
    for game in games:
        for target,band,method,*values in game["rows"]:
            row = result[target,band,method]
            for i in range(6):
                row[i] += values[i]
            row[-1] = max(row[-1],values[-1])
    output = []
    for (target,band,method),(count,raw,error,better,worse25,worse50,max_added) in sorted(result.items()):
        output.append(dict(target=target,band=band,method=method,n=count,
            baseline_cm=100*raw/count,error_cm=100*error/count,
            improved_1cm=better/count,worsened_25cm=worse25/count,
            worsened_50cm=worse50/count,max_added_cm=100*max_added))
    return output


def select(rows,near_safe):
    winners = {}
    for target in ("player","ball"):
        common = [r for r in rows if r["band"]=="under50" and r["target"]==target]
        winner = min(common,key=lambda r:r["error_cm"])
        bounded = min([r for r in common if r["method"].endswith("_cap25")],key=lambda r:r["error_cm"])
        winners[target] = dict(mean=winner["method"],capped25=bounded["method"])
        if near_safe:
            gated = min([r for r in common if r["method"].endswith("_far5_cap25")],
                        key=lambda r:r["error_cm"])
            winners[target]["near_safe"] = gated["method"]
    return winners


def uncertainty(games,winners):
    rng = np.random.default_rng(SEED)
    result = {}
    for target,candidates in winners.items():
        result[target] = {}
        for selection,method in candidates.items():
            rows = np.array([row[3:6] for g in games for row in g["rows"]
                             if row[:3]==[target,"under50",method]],dtype=float)
            selected = rows[rng.integers(len(rows),size=(5000,len(rows)))].sum(axis=1)
            change = 100*(selected[:,2]-selected[:,1])/selected[:,0]
            result[target][selection] = np.quantile(change,[.025,.975]).tolist()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--workers",type=int,default=16)
    p.add_argument("--near-safe",action="store_true",help="confirm 5 m gate in previously unused games")
    args = p.parse_args()
    games = sorted({m.parent.parent for m in args.logs.resolve().glob("*/*/meta.json")})
    required = 100 if args.near_safe else 60
    if len(games)<required:
        p.error(f"need at least {required} games for the requested split")
    # Hash ordering fixes the split without consulting any sensor or match result.
    rng = random.Random(SEED)
    rng.shuffle(games)
    test_start = 60 if args.near_safe else 20
    subsets = {"validation":games[:20],"test":games[test_start:test_start+40]}
    chosen = subsets["validation"]+subsets["test"]
    slots = {}
    for index,game in enumerate(games[:test_start+40]):
        uniforms = (index%11+1,(index+5)%11+1)
        slots[game.name]=[f"{side}_{number}" for side in "lr" for number in uniforms]
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    source = Path(__file__).with_name("scan_temporal.cpp")
    signature = hashlib.sha256(source.read_bytes()+source.with_name("csv_reader.h").read_bytes()
                               +source.with_name("kalman_state.h").read_bytes()).hexdigest()[:12]
    cache = output/("games-"+signature)
    cache.mkdir(exist_ok=True)
    scanner = output/("temporal-"+signature)
    if not scanner.is_file():
        subprocess.run(["g++","-O3","-std=c++17",str(source),"-lz","-o",str(scanner)],check=True)
    split = {name:[{"game":g.name,"observers":slots[g.name]} for g in cohort] for name,cohort in subsets.items()}
    (output/"split.json").write_text(json.dumps(split,indent=2)+"\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = {}
        jobs = [pool.submit(scan,g,scanner,cache,slots[g.name],args.near_safe) for g in chosen]
        for i,future in enumerate(as_completed(jobs),1):
            result = future.result()
            results[result["game"]] = result
            if i==1 or i%10==0:
                print(f"Temporal replay: {i}/60 games",flush=True)
    collected = {name:[results[g.name] for g in cohort] for name,cohort in subsets.items()}
    validation = aggregate(collected["validation"])
    winners = select(validation,args.near_safe)
    test = aggregate(collected["test"])
    (output/"metrics.json").write_text(json.dumps(dict(source=str(args.logs.resolve()),
        seed=SEED,validation=validation,selected=winners,test=test,
        test_change95_cm=uncertainty(collected["test"],winners),split=split),indent=2)+"\n")


if __name__ == "__main__":
    main()
