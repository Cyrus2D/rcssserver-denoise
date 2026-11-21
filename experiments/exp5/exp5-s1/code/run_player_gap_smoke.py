"""Isolated short native-model checks; real batches still use start.sh."""
import argparse,json,shlex,subprocess,signal,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
def main(a):
    if Path(a.name).name!=a.name:raise ValueError('Use a simple run name')
    output=ROOT/'logs/exp5-s1/live-smoke'/a.name;output.mkdir(parents=True,exist_ok=False)
    model=ROOT/'logs/exp5-s1/live-model'
    def team(name,mode):
        args=[str(ROOT/'helios-base-denoise/build/bin/start.sh'),'-h','127.0.0.1','-p',str(a.port),'-P',str(a.port+2),'-t',name,'-C',
              '--fullstate','reference','--denoise-log-dir',str(output/'player'),'--denoise-game-id',a.name,'--no-denoise-raw','--no-denoise-landmarks']
        if mode!='off':args+=['--player-gap',mode,'--player-gap-model-dir',str(model),'--player-gap-pos-count',a.count]
        if a.features and mode!='off':args+=['--player-gap-features']
        if a.debug_overlays:
            directory=output/'debug'/name;directory.mkdir(parents=True,exist_ok=True)
            args+=['--debug','--debug-server-logging','--log-dir',str(directory),
                   '--player-gap-debug','both','--player-gap-debug-target',a.debug_target]

        return shlex.join(args)
    params=dict(port=a.port,coach_port=a.port+1,olcoach_port=a.port+2,synch_mode=True,auto_mode=True,full_view=False,see_no_quantization=not a.quantized,
       see_noise_free=False,see_no_fade_all=True,see_no_fade_unum=True,say_msg_size=0,fullstate_l=True,fullstate_r=True,randomize_player_commands=True,
       command_order_seed=50701,random_seed=50701,half_time=a.half_time,nr_normal_halfs=2,nr_extra_halfs=0,penalty_shoot_outs=False,
       connect_wait=100,kick_off_wait=10,game_over_wait=10,game_logging=False,text_logging=False,game_log_dir=str(output),
       team_l_start=team('lt',a.left),team_r_start=team('rt',a.right))
    (output/'match.conf').write_text(''.join(f'server::{k} = {json.dumps(v)}\n' for k,v in params.items())+'player::random_seed=50702\n')
    with (output/'server.log').open('w') as log:
        process=subprocess.Popen([str(ROOT/'rcssserver-denoise/build/rcssserver'),'include=match.conf'],cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:status=process.wait(timeout=600)
        finally:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    meta=[json.loads(p.read_text()) for p in (output/'player'/a.name).glob('*/meta.json')]
    assert status==0 and len(meta)==22 and all(m['final'] and m['status']=='ok' for m in meta),(status,len(meta))
    print('PASS',a.name,'22 finalized agents',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--left',choices=['off','shadow','apply'],default='shadow');p.add_argument('--right',choices=['off','shadow','apply'],default='off');p.add_argument('--count',choices=['preserve','zero','half'],default='preserve');p.add_argument('--half-time',type=int,default=20);p.add_argument('--port',type=int,default=22600);p.add_argument('--features',action='store_true');p.add_argument('--quantized',action='store_true');p.add_argument('--debug-overlays',action='store_true');p.add_argument('--debug-target',default='all');main(p.parse_args())
