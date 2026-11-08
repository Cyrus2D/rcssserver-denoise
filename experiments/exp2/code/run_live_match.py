#!/usr/bin/env python3
"""Run one local Exp2 match with reference fullstate and independent team modes."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

WORKSPACE = Path(__file__).resolve().parents[4]
MODES = ("off", "shadow", "apply")


def run(args):
    server = args.server.resolve()
    start = args.start.resolve()
    model = args.model.resolve()
    if not all(os.access(path, os.X_OK) and path.is_file() for path in (server, start)):
        raise ValueError("build rcssserver and Helios or provide --server and --start")
    if (args.left != "off" or args.right != "off") and not model.is_file():
        raise ValueError(f"missing exported weights: {model}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.game_id):
        raise ValueError("game ID may contain letters, digits, hyphens and underscores")

    output = args.output.resolve() / args.game_id
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "dataset"

    def team(name, mode):
        command = [
            str(start), "-h", "127.0.0.1", "-p", str(args.port), "-t", name,
            "--fullstate", "reference", "--denoise-log-dir", str(dataset),
            "--denoise-game-id", args.game_id, "--no-denoise-raw",
            "--self-localization", mode,
        ]
        if mode != "off":
            command += ["--self-localization-model", str(model)]
        return shlex.join(command)

    options = {
        "port": args.port, "coach_port": args.port + 1, "olcoach_port": args.port + 2,
        "synch_mode": "true", "auto_mode": "true", "fullstate_l": "true", "fullstate_r": "true",
        "full_view": "false", "see_noise_free": "false", "see_no_fade_unum": "false",
        "see_no_fade_all": "false", "half_time": args.half_time, "nr_normal_halfs": 2,
        "nr_extra_halfs": 0, "penalty_shoot_outs": "false",
        "connect_wait": 100, "kick_off_wait": 10, "game_over_wait": 10,
        "game_log_dir": str(output), "text_log_dir": str(output),
        "team_l_start": team("Exp2Left", args.left), "team_r_start": team("Exp2Right", args.right),
    }
    config = output / "match.conf"
    string_options = {"game_log_dir", "text_log_dir", "team_l_start", "team_r_start"}
    with config.open("w") as conf:
        for key, value in options.items():
            literal = json.dumps(value, ensure_ascii=False) if key in string_options else value
            conf.write(f"server::{key} = {literal}\n")
    command = [str(server), f"include={config.name}"]
    log_path = output / "match.log"
    print(f"Running {args.game_id} ({args.left} vs {args.right}). Match log: {log_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            status = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
        finally:
            # Allow players to close their dataset files after the final referee message.
            time.sleep(1)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if status:
        raise RuntimeError(f"rcssserver exited with {status}. See {log_path}")
    dataset = dataset / args.game_id
    files = list(dataset.glob("*/meta.json"))
    if len(files) != 22 or any(not json.loads(path.read_text()).get("final") for path in files):
        raise RuntimeError(f"expected 22 finalized player logs. See {log_path}")
    for side, mode in (("l", args.left), ("r", args.right)):
        count = len(list(dataset.glob(f"{side}_*/self_localization.csv")))
        if count != (0 if mode == "off" else 11):
            raise RuntimeError(f"unexpected self localization log count for {side}: {count}. See {log_path}")
    print(f"Dataset: {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_id")
    parser.add_argument("--left", choices=MODES, default="shadow")
    parser.add_argument("--right", choices=MODES, default="shadow")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--half-time", type=int, default=300, help="simulated seconds per half")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "logs/exp2/live")
    parser.add_argument("--server", type=Path, default=WORKSPACE / "rcssserver-denoise/build/rcssserver")
    parser.add_argument("--start", type=Path, default=WORKSPACE / "helios-base-denoise/build/bin/start.sh")
    parser.add_argument("--model", type=Path,
                        default=WORKSPACE / "logs/exp2/runs/step05-circular/self-position-gru.weights")
    args = parser.parse_args()
    if args.half_time < 1 or not 1024 <= args.port <= 65533:
        parser.error("half-time must be positive and port must be in 1024..65533")
    run(args)
