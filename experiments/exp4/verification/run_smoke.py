#!/usr/bin/env python3
"""Run a short isolated 360-degree match to verify Exp4 live wiring."""
import argparse
import json
from pathlib import Path
import shlex
import signal
import subprocess
import time
import socket
import threading

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--half-time", type=int, default=60)
    parser.add_argument("--port", type=int, default=16300)
    parser.add_argument("--left", choices=["apply", "shadow", "off"], default="apply")
    parser.add_argument("--right", choices=["apply", "shadow", "off"], default="off")
    parser.add_argument("--player-state", choices=["apply", "shadow", "off"], default=None)
    parser.add_argument("--randomize-commands", action="store_true")
    parser.add_argument("--order-seed", type=int, default=-1)
    parser.add_argument("--server", type=Path, default=ROOT / "rcssserver-denoise/build/rcssserver")
    parser.add_argument("--base", type=Path, default=ROOT/"helios-base-denoise/build/bin")
    parser.add_argument("--debug-overlay", choices=["off","focus","overview"], default="off")
    parser.add_argument("--debug-target", default="nearest")
    parser.add_argument("--debug-observer", type=int, default=6)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.half_time < 1 or not 1024 <= args.port <= 65533:
        parser.error("invalid name, half time or port")
    output = ROOT / "logs/exp4/live-f4" / args.name
    output.mkdir(parents=True, exist_ok=False)
    start = args.base.resolve() / "start.sh"
    def team(name, mode):
        command = [str(start), "-h", "127.0.0.1", "-p", str(args.port), "-t", name, "-C",
                   "--fullstate", "reference", "--denoise-log-dir", str(output/"dataset"),
                   "--denoise-game-id", args.name, "--no-denoise-raw", "--no-denoise-landmarks"]
        if args.debug_overlay != "off" and mode != "off":
            command[0] = str(start.with_name("start-debug.sh"))
            debug_dir = output/("debug-"+name)
            debug_dir.mkdir()
            command += ["--denoise-debug", args.debug_overlay, "--denoise-debug-target", args.debug_target,
                        "--denoise-debug-observer", str(args.debug_observer),
                        "--debug-server-port", str(args.port+32), "--log-dir", str(debug_dir)]
        if mode != "off":
            command += ["--player-position-denoise", "apply",
                        "--player-position-model", str(ROOT/"logs/exp3-context-runs/step26-polar/player-position.weights"),
                        "--player-position-routing", "corrected", "--exp4-state", mode,
                        "--exp4-model-dir", str(ROOT/"logs/exp4/weights")]
            if args.player_state is not None:
                command += ["--exp4-player-state", args.player_state]
        return shlex.join(command)
    options = dict(port=args.port, coach_port=args.port+1, olcoach_port=args.port+2,
                   synch_mode=True, auto_mode=True, fullstate_l=True, fullstate_r=True,
                   full_view=True, see_no_fade_unum=True, see_no_fade_all=False, see_noise_free=False,
                   say_msg_size=0, quantize_step=.1,
                   half_time=args.half_time, nr_normal_halfs=2, nr_extra_halfs=0,
                   penalty_shoot_outs=False, connect_wait=100, kick_off_wait=10, game_over_wait=10,
                   game_log_dir=str(output), text_log_dir=str(output),
                   team_l_start=team("F4Left",args.left), team_r_start=team("F4Right",args.right))
    if args.randomize_commands:
        options.update(randomize_player_commands=True, command_order_seed=args.order_seed,
                       command_order_log=True, game_log_fixed=True, game_log_fixed_name=args.name)
    with (output/"match.conf").open("w") as f:
        for k,v in options.items():
            f.write(f"server::{k} = {json.dumps(v) if isinstance(v,(str,bool)) else v}\n")
    print("Running",output,flush=True)
    receiver = None
    stop = threading.Event()
    def receive_debug():
        with (output/"debug-packets.txt").open("w") as packets:
            while not stop.is_set():
                try: data,_ = receiver.recvfrom(65535)
                except socket.timeout: continue
                if b"DENOISE" in data:
                    packets.write(data.rstrip(b"\0").decode()+"\n")
    if args.debug_overlay != "off":
        receiver = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1",args.port+32))
        receiver.settimeout(.1)
        thread = threading.Thread(target=receive_debug)
        thread.start()
    with (output/"match.log").open("w") as log:
        process = subprocess.Popen([str(args.server.resolve()),"include=match.conf"],
                                   cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            status = process.wait(timeout=300)
            time.sleep(1)
        finally:
            try: subprocess.run(["kill","-TERM","--",str(-process.pid)],check=False,capture_output=True)
            except ProcessLookupError: pass
            if receiver:
                stop.set(); thread.join(); receiver.close()
    if status: raise RuntimeError(f"server failed: {status}")
    files = list((output/"dataset"/args.name).glob("*/meta.json"))
    assert len(files)==22
    assert all(json.loads(p.read_text())["final"] for p in files)
    print("PASS: 22 finalized agents.",flush=True)

if __name__ == "__main__":
    main()
