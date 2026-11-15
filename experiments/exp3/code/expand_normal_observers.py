#!/usr/bin/env python3
"""Add a second observer per side to training games only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from build_normal_events import make_game
from replay_normal import build_game


def extra(job):
    source, events, known, game = job
    name = game["name"]
    number = game["observer_unum"] % 11 + 1
    record = make_game((source / name, "train", number, events))
    _, audit = build_game((events, known, name, False))
    # Store uses the original naming convention and keeps the two sides
    # together. The synthetic name uniquely identifies this observer pair.
    label = f"{name}-extra-{number}"
    for side in "lr":
        for root, extension in ((events, "npz"), (known, "npy")):
            (root / f"{name}-{side}.{extension}").rename(root / f"{label}-{side}.{extension}")
    return {**record, "name": label, "source_game": name, "identity_audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "events", "known", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    events, known = args.output / "events", args.output / "known"
    args.output.mkdir(parents=True, exist_ok=False)
    for root in (events, known):
        root.mkdir()
    manifest = json.loads((args.known / "manifest.json").read_text())
    games = manifest["games"]
    jobs = [(args.source, events, known, game) for game in games if game["split"] == "train"]
    # Add symlinks after extraction so the original data cannot be overwritten.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        additions = []
        for i, record in enumerate(pool.map(extra, jobs), 1):
            additions.append(record)
            if i == 1 or i % 10 == 0 or i == len(jobs):
                print(f"Built extra observers for {i}/{len(jobs)} training games", flush=True)
    for target, original, extension in ((events, args.events, "npz"), (known, args.known, "npy")):
        for game in games:
            for side in "lr":
                name = f"{game['name']}-{side}.{extension}"
                (target / name).symlink_to((original / name).resolve(strict=True))
        (target / "manifest.json").write_text(json.dumps({**manifest, "games": games + additions},
                                                        indent=2) + "\n")


if __name__ == "__main__":
    main()
