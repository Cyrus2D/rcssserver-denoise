#!/usr/bin/env python3
"""Build a game-split Exp6 training set from corrected Gaussian observer logs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

SEED = 60624
HISTORY = 8
FEATURES = 14
LABELS = 6
WIDTH = HISTORY * FEATURES + LABELS
SLOTS = {f"{side}_{number}" for side in "lr" for number in range(1, 12)}
EXPECTED = dict(full_view="true",synch_mode="true",see_no_quantization="false",see_noise_free="false",
                say_msg_size="0",fullstate_l="true",fullstate_r="true",
                see_no_fade_unum="true",see_no_fade_all="false",dist_noise_rate="0.0125",
                focus_dist_noise_rate="0.0125")
FEATURE_NAMES = ["shift_radial_div3","shift_tangential_div3","velocity_radial","velocity_tangential",
    "direct_velocity","observed_distance_div50","estimated_range_sigma","ray_dot","ray_cross",
    "kalman_delta_radial","kalman_delta_tangential","teammate","valid_timestep","lookback_div8"]
LABEL_NAMES = ["truth_delta_radial","truth_delta_tangential","observed_distance","true_distance",
               "kalman_delta_radial","kalman_delta_tangential"]


def verify_game(game, autotest):
    files = {p.parent.name:p for p in game.glob("*/meta.json")}
    if files.keys() != SLOTS:
        raise ValueError(f"{game}: expected 22 observer logs")
    scores = set()
    for path in files.values():
        m = json.loads(path.read_text())
        cfg = m["config"]
        if (not m["final"] or m["status"]!="ok" or not cfg.get("gaussian_range_localization")
                or not cfg["gaussian_see"] or not cfg["debug_fullstate"] or cfg["use_fullstate"]):
            raise ValueError(f"{path}: need corrected Gaussian and reference truth")
        for flag,expected in EXPECTED.items():
            if m["server_param"].get(flag)!=expected:
                raise ValueError(f"{path}: {flag}={m['server_param'].get(flag)}, expected {expected}")
        scores.add((m["score_left"],m["score_right"]))
    if len(scores)!=1:
        raise ValueError(f"{game}: observer scores disagree")
    left,right = scores.pop()
    result = autotest/"result.d"/game.name
    server = re.findall(r"Score:\s*(\d+)\s*-\s*(\d+)",result.read_text(errors="replace"))
    if not server or tuple(map(int,server[-1]))!=(left,right):
        raise ValueError(f"{result}: score mismatch")
    return {"game":game.name,"left":left,"right":right}


def extract(job, scanner, output):
    game, slot, split = job
    strides = (12,3) if split=="train" else (10,2)
    path = game/slot
    folder = output/split
    name = game.name+"-"+slot
    prefix = folder/name
    temporary = folder/("."+name)
    meta = (path/"meta.json").read_bytes()
    digest = hashlib.sha256(meta)
    for kind in ("agent_cycle","objects"):
        stat = (path/(kind+".csv.gz")).stat()
        digest.update(f"{stat.st_size}/{stat.st_mtime_ns}".encode())
    marker = prefix.with_suffix(".json")
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous["input_hash"]==digest.hexdigest() and all((output/v["path"]).is_file() for v in previous["data"].values()):
            return previous
    subprocess.run([str(scanner),str(path),str(temporary),*map(str,strides)],check=True)
    datasets = {}
    for target in ("player","ball"):
        src = Path(str(temporary)+f".{target}.bin")
        count,remainder = divmod(src.stat().st_size,WIDTH*4)
        if remainder or not count:
            raise ValueError(f"{src}: invalid float32 records")
        target_file = Path(str(prefix)+f".{target}.bin")
        src.replace(target_file)
        datasets[target] = {"path":str(target_file.relative_to(output)),"windows":count}
    result = dict(game=game.name,observer=slot,split=split,strides=list(strides),data=datasets,input_hash=digest.hexdigest())
    marker.write_text(json.dumps(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("logs","old_metrics","confirmation_metrics","fresh_autotest","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--workers",type=int,default=22)
    args = p.parse_args()
    logs = args.logs.resolve()
    all_games = {m.parent.parent.name:m.parent.parent for m in logs.glob("*/*/meta.json")}
    old = {s["game"] for s in json.loads(args.old_metrics.read_text())["scores"]}
    confirmation = json.loads(args.confirmation_metrics.read_text())["split"]["test"]
    testing = {g["game"] for g in confirmation}
    if len(old)!=100 or not testing<=old or len(testing)!=40 or not old<=all_games.keys():
        p.error("expect the original 100 games and their frozen 40-game confirmation split")
    fresh = sorted(all_games.keys()-old)
    if len(fresh)!=200:
        p.error(f"expected 200 additional games, found {len(fresh)}")
    rng = random.Random(SEED)
    rng.shuffle(fresh)
    subsets = {"train":fresh[:160],"validation":fresh[160:180],"fresh_test":fresh[180:],
               "previous_test":[g["game"] for g in confirmation]}
    assert sum(map(len,subsets.values()))==len({g for games in subsets.values() for g in games})
    previous_observers = {g["game"]:g["observers"] for g in confirmation}
    jobs = []
    for split,games in subsets.items():
        for i,game in enumerate(games):
            if split=="previous_test":
                slots = previous_observers[game]
            elif split=="train":
                slots = [f"l_{i%11+1}", f"r_{(i+4)%11+1}"]
            else:
                slots = [f"{side}_{number}" for side in "lr" for number in (i%11+1,(i+5)%11+1)]
            jobs.extend((all_games[game],slot,split) for slot in slots)
    source = Path(__file__).with_name("extract_windows.cpp")
    headers = [source.parent.parent/"reporting"/file for file in ("csv_reader.h","kalman_state.h")]
    signature = hashlib.sha256(b"".join(f.read_bytes() for f in [source,*headers])).hexdigest()[:12]
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    scanner = output/f"extract-{signature}"
    if not scanner.is_file():
        subprocess.run(["g++","-O3","-std=c++17",str(source),"-lz","-o",str(scanner)],check=True)
    shard_dir = output/("shards-"+signature)
    for split in subsets:
        (shard_dir/split).mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scores = list(pool.map(lambda game: verify_game(all_games[game],args.fresh_autotest),fresh))
        print(f"Verified {len(scores)} new match scores and {len(scores)*22} finalized Gaussian observer logs.",flush=True)
        futures = [pool.submit(extract,job,scanner,shard_dir) for job in jobs]
        records = []
        for i,job in enumerate(as_completed(futures),1):
            row = job.result()
            for info in row["data"].values():
                info["path"] = str(Path(shard_dir.name)/info["path"])
            records.append(row)
            if i==1 or i%40==0 or i==len(jobs):
                print(f"Built {i}/{len(jobs)} observer shards",flush=True)
    counts = Counter()
    for row in records:
        for target,info in row["data"].items():
            counts[(row["split"],target)] += info["windows"]
    result = dict(source=str(logs),seed=SEED,history=HISTORY,features=FEATURE_NAMES,labels=LABEL_NAMES,
        width=WIDTH,split=subsets,records=sorted(records,key=lambda x:(x["split"],x["game"],x["observer"])),
        counts={f"{split}_{target}":n for (split,target),n in counts.items()},fresh_scores=scores)
    (output/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result["counts"],flush=True)


if __name__ == "__main__":
    main()
