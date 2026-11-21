#!/usr/bin/env python3
"""Short baseline match: exact visible measurements and ordinary finite views."""
import argparse,json,shlex,subprocess,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("name");p.add_argument("--view-policy",choices=["adaptive","narrow","normal","wide"],default="adaptive")
    p.add_argument("--half-time",type=int,default=30);p.add_argument("--port",type=int,default=16600)
    p.add_argument("--quantized",action="store_true",help="legacy-mode negative control")
    p.add_argument("--server",type=Path,default=ROOT/"rcssserver-denoise/build/rcssserver")
    a=p.parse_args()
    if Path(a.name).name!=a.name or a.half_time<1 or not 1024<=a.port<=65533:p.error("invalid name, time or port")
    output=ROOT/"logs/exp5/smoke"/a.name;output.mkdir(parents=True,exist_ok=False)
    def team(name):
        return shlex.join([str(ROOT/"helios-base-denoise/build/bin/start.sh"),"-h","127.0.0.1","-p",str(a.port),
          "-t",name,"-C","--fullstate","reference","--view-policy",a.view_policy,
          "--denoise-log-dir",str(output/"player"),"--denoise-game-id",a.name])
    params=dict(port=a.port,coach_port=a.port+1,olcoach_port=a.port+2,
       synch_mode=True,auto_mode=True,full_view=False,see_no_quantization=not a.quantized,
       see_noise_free=False,see_no_fade_all=True,see_no_fade_unum=True,say_msg_size=0,
       fullstate_l=True,fullstate_r=True,randomize_player_commands=True,command_order_seed=42,random_seed=42,
       half_time=a.half_time,nr_normal_halfs=2,nr_extra_halfs=0,penalty_shoot_outs=False,
       connect_wait=100,kick_off_wait=10,game_over_wait=10,game_logging=True,text_logging=False,
       game_log_fixed=True,game_log_fixed_name=a.name,game_log_dir=str(output),
       team_l_start=team("GapLeft"),team_r_start=team("GapRight"))
    with (output/"match.conf").open("w")as f:
        for k,v in params.items():f.write(f"server::{k} = {json.dumps(v)}\n")
        f.write("player::random_seed = 43\n")
    print("Running",output,flush=True)
    with (output/"server.log").open("w")as log:
        proc=subprocess.Popen([str(a.server.resolve()),"include=match.conf"],
          cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:status=proc.wait(timeout=300)
        finally:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    assert status==0,("server exit",status)
    meta=[json.loads(m.read_text())for m in (output/"player"/a.name).glob("*/meta.json")]
    assert len(meta)==22 and all(m["final"]and m["status"]=="ok"for m in meta)
    assert all(m["run_options"]["view_policy"]==a.view_policy for m in meta)
    assert all(not m["config"]["use_fullstate"]for m in meta)
    print("PASS: 22 finalized baseline agents; reference-only fullstate; policy",a.view_policy,flush=True)
if __name__=="__main__":main()
