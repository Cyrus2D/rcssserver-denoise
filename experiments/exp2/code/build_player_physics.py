#!/usr/bin/env python3
"""Index the player type known before each Exp2 motion transition."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from player_physics import MOTION_FIELDS, MOTION_ROW, MOTION_VERSION, TYPE_FIELDS

DATASET = Path("/home/nader/workspace/denoise/logs/exp2")
TYPE_CHANGE = re.compile(r"\(change_player_type\s+(\d+)\s+(\d+)\)(?:\\[rn])*")


def types_at_commands(path, unum, valid_types):
    """Read receive/send order without using the final type or fullstate."""
    opener = gzip.open if path.suffix == ".gz" else open
    current = 0
    commands = {}
    with opener(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            message = row["msg"]
            if row["dir"] == "recv" and message.startswith("(change_player_type "):
                change = TYPE_CHANGE.fullmatch(message)
                if change and int(change[1]) == unum:
                    current = int(change[2])
                    if current not in valid_types:
                        raise ValueError(f"{path}: unknown own player type {current}")
            elif row["dir"] == "send" and row["stopped"] == "0" and "(done)" in message:
                # Keep the state at the first decision send if a packet repeats.
                commands.setdefault(int(row["cycle"]), current)
    return commands


def build_game(task):
    dataset, source, target, game = task
    original = np.load(dataset / game["file"], mmap_mode="r", allow_pickle=False)
    result = np.zeros(len(original), dtype=MOTION_ROW)
    agent_data = {}
    totals = {"transitions": 0, "nondefault": 0, "agents_with_midgame_change": 0}
    for segment in game["segments"]:
        agent = segment["agent"]
        if agent not in agent_data:
            directory = source / game["id"] / agent
            meta = json.loads((directory / "meta.json").read_text())
            if not meta["config"]["synch_see"] or str(meta["server_param"]["synch_mode"]).lower() != "true":
                raise ValueError(f"{directory}: the index expects synchronous decision packets")
            if f'{meta["side"]}_{meta["unum"]}' != agent:
                raise ValueError(f"{directory}: log identity differs from the dataset")
            parameters = {int(entry["id"]): [entry[field] for field in TYPE_FIELDS]
                          for entry in meta["player_types"]}
            if 0 not in parameters or not np.isfinite(list(parameters.values())).all():
                raise ValueError(f"{directory}: invalid player type catalogue")
            raw = directory / "raw_stream.csv.gz"
            if not raw.is_file():
                raw = directory / "raw_stream.csv"
            commands = types_at_commands(raw, int(meta["unum"]), parameters)
            agent_data[agent] = commands, parameters

        commands, parameters = agent_data[agent]
        start, end = segment["start"], segment["end"]
        cycles = original["cycle"][start:end]
        # A seed is already a complete state. A motion row at t uses what
        # the agent knew before sending the command that moves it to t.
        try:
            ids = [commands[int(cycle) - 1] for cycle in cycles[1:]]
        except KeyError as error:
            raise ValueError(f"{game['id']}/{agent}: no decision packet for cycle {error.args[0]}") from error
        result["type_id"][start] = 0
        result["values"][start] = parameters[0]
        result["type_id"][start + 1:end] = ids
        result["values"][start + 1:end] = [parameters[id_] for id_ in ids]
        totals["transitions"] += len(ids)
        totals["nondefault"] += sum(id_ != 0 for id_ in ids)
    totals["agents_with_midgame_change"] = sum(
        len({id_ for cycle, id_ in commands.items() if cycle > 0}) > 1
        for commands, _ in agent_data.values()
    )
    np.save(target / game["file"], result, allow_pickle=False)
    return {"file": game["file"], "rows": len(result), **totals}


def build(args):
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new --output")
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    source = Path(manifest["source"])
    games = [entry for entry in manifest["games"] if entry["split"] == args.split]
    if not games:
        raise ValueError(f"No games in split {args.split}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "index"
        staging.mkdir()
        tasks = [(dataset, source, staging, game) for game in games]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(games))) as pool:
            entries = []
            for entry in pool.map(build_game, tasks):
                entries.append(entry)
                print(f"Indexed {len(entries)}/{len(games)} {args.split} games", flush=True)
        index = {"schema_version": MOTION_VERSION, "fields": MOTION_FIELDS, "split": args.split,
                 "dataset_fingerprint": hashlib.sha256(manifest_bytes).hexdigest(), "games": entries}
        (staging / "manifest.json").write_text(json.dumps(index, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"{output} appeared during the build")
        staging.rename(output)
    print(f"Player physics index: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output is None:
        args.output = args.dataset / f"player-physics-{args.split}"
    try:
        build(args)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
