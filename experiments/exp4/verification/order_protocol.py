#!/usr/bin/env python3
"""Exercise the real UDP done barrier and strict failure paths."""
import argparse,json,os,re,socket,subprocess,tempfile,time
from pathlib import Path

def startup_failure(server,root,name,options,expected):
    directory=root/name;directory.mkdir()
    env={**os.environ,"RCSS_CONF_DIR":str(directory)}
    result=subprocess.run([str(server),*options],cwd=directory,env=env,capture_output=True,text=True,timeout=5)
    output=result.stdout+result.stderr
    assert result.returncode!=0 and expected in output and "Score:"not in output,output[-2000:]
    return dict(exit=result.returncode,expected=expected)

def timeout_case(server,root):
    directory=root/"timeout";directory.mkdir()
    env={**os.environ,"RCSS_CONF_DIR":str(directory)}
    options=["server::port=16490","server::coach_port=16491","server::olcoach_port=16492",
             "server::synch_mode=true","server::randomize_player_commands=true","server::command_order_seed=7",
             "server::game_logging=false","server::text_logging=false","server::game_log_fixed=true",
             "server::game_log_fixed_name=timeout","server::game_log_dir="+str(directory)]
    log=directory/"server.log"
    with log.open("w")as f:
        process=subprocess.Popen([str(server),*options],cwd=directory,env=env,stdout=f,stderr=subprocess.STDOUT)
        client=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);client.bind(("127.0.0.1",0));client.settimeout(.2)
        peer=("127.0.0.1",16490);initialized=False;sent=False;query_count=None
        started=time.monotonic()
        try:
            while time.monotonic()-started<5 and process.poll()is None:
                if not initialized:client.sendto(b"(init TimeoutTeam (version 19))\0",peer)
                try:packet,address=client.recvfrom(65535)
                except socket.timeout:continue
                message=packet.rstrip(b"\0").decode()
                if message.startswith("(init "):peer=address;initialized=True
                elif message=="(think)"and not sent:
                    # No done: an action plus a query must leave the turn counter at zero.
                    client.sendto(b"(turn 10)(sense_body)\0",peer);sent=True
                elif message.startswith("(sense_body")and sent:
                    query_count=int(re.search(r"\(turn (\d+)\)",message)[1])
            status=process.wait(timeout=3)
        finally:
            client.close()
            if process.poll()is None:process.kill();process.wait()
    output=log.read_text()
    assert initialized and sent and query_count==0
    assert status==1 and "discarding all actions and aborting match"in output and "Score:"not in output,output[-2000:]
    return dict(exit=status,turn_counter_before_done=query_count,scored_result=False)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("server",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();server=a.server.resolve()
    with tempfile.TemporaryDirectory(prefix="server-order-protocol-")as temp:
        root=Path(temp)
        checks={"async_rejected":startup_failure(server,root,"async",["server::synch_mode=false","server::randomize_player_commands=true"],"requires synch_mode=true"),
                "bad_seed_rejected":startup_failure(server,root,"seed",["server::synch_mode=true","server::command_order_seed=-2"],"must be -1 or nonnegative"),
                "timeout_aborted":timeout_case(server,root)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(checks,indent=2))
