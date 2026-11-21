"""Collect untouched baseline games for frozen-policy confirmation only."""
from pathlib import Path
import argparse,json,os,signal,subprocess,shlex,socket,time
from concurrent.futures import ThreadPoolExecutor
from refinement_models import atomic_json

ROOT=Path('/home/nader/workspace/denoise')

def run_game(task):
    i,out=task;name=f'fresh-{i+1:02d}';seed=50601+i;port=21600+i*10;game=out/name
    final=game/'completed.json'
    if final.exists():return json.loads(final.read_text())
    game.mkdir(parents=True,exist_ok=False)
    # Check all three UDP ports before starting; never stop an unrelated server.
    for p in range(port,port+3):
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:s.bind(('127.0.0.1',p))
    def team(side):
        return shlex.join([str(ROOT/'helios-base-denoise/build/bin/start.sh'),'-h','127.0.0.1','-p',str(port),'-P',str(port+2),
          '-t',side,'--fullstate','reference','--view-policy','adaptive','--denoise-log-dir',str(out/'player'),
          '--denoise-game-id',name,'--no-denoise-raw','--no-denoise-landmarks'])
    params=dict(port=port,coach_port=port+1,olcoach_port=port+2,synch_mode=True,auto_mode=True,
        full_view=False,see_no_quantization=True,see_noise_free=False,see_no_fade_all=True,see_no_fade_unum=True,
        say_msg_size=0,fullstate_l=True,fullstate_r=True,randomize_player_commands=True,command_order_seed=seed,
        command_order_log=True,random_seed=seed,half_time=300,nr_normal_halfs=2,nr_extra_halfs=0,
        penalty_shoot_outs=False,connect_wait=100,kick_off_wait=10,game_over_wait=10,
        game_logging=True,text_logging=False,game_log_fixed=True,game_log_fixed_name=name,game_log_dir=str(game),
        team_l_start=team('lt'),team_r_start=team('rt'))
    config=game/'match.conf'
    config.write_text(''.join(f'server::{k} = {json.dumps(v)}\n' for k,v in params.items())+f'player::random_seed = {seed+100}\n')
    started=time.monotonic();print('New untouched game started',name,flush=True)
    with (game/'server.log').open('w') as log:
        proc=subprocess.Popen([str(ROOT/'rcssserver-denoise/build/rcssserver'),'include=match.conf'],cwd=game,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:status=proc.wait(timeout=900)
        finally:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    metas=[json.loads(f.read_text()) for f in (out/'player'/name).glob('*/meta.json')]
    if status or len(metas)!=22 or not all(m['final'] and m['status']=='ok' and not m['config']['use_fullstate'] for m in metas):
        raise RuntimeError(f'Incomplete new game {name}: exit={status}, observers={len(metas)}')
    result=dict(game=name,seed=seed,seconds=time.monotonic()-started,observers=len(metas),config=params,
                use='Independent confirmation only; never used in fitting, threshold selection or model selection.')
    atomic_json(final,result);print('New game completed',name,round(result['seconds']),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--games',type=int,default=10);p.add_argument('--jobs',type=int,default=2);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    atomic_json(a.out/'protocol.json',dict(games=a.games,jobs=a.jobs,seeds=list(range(50601,50601+a.games)),
      prohibition='Do not inspect prediction performance until all policies and routing choices have been frozen on the original validation games.'))
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:results=list(pool.map(run_game,[(i,a.out) for i in range(a.games)]))
    atomic_json(a.out/'collection-results.json',results)
