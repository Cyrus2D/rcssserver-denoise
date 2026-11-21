#!/usr/bin/env python3
"""Reconstruct a labelled RCG clip and dlog overlay from recorded truth/prediction CSVs.

This is a visualization export, not the original server RCG and not a new run.
Command counters and unrecorded action-state bits are placeholders.
"""
import argparse,json,math
import numpy as np
from pathlib import Path
import pandas as pd


def value(v,default=0.):
    return float(v) if pd.notna(v) and math.isfinite(float(v)) else default


def export(a):
    game=a.game;observer=game/a.observer
    m=json.loads((observer/'meta.json').read_text());assert m['final'] and m['status']=='ok'
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    lo,hi=a.cycle-a.radius,a.cycle+a.radius
    agent_cols=['cycle','stopped','decision_made','game_mode','score_l','score_r','view_width','sb_view_width',
                'tru_pos_x','tru_pos_y','tru_vel_x','tru_vel_y','tru_body','tru_neck','tru_stamina','tru_effort','tru_recovery','tru_player_type','tru_goalie']
    agents={};names={}
    for file in sorted(game.glob('*/meta.json')):
        meta=json.loads(file.read_text());oid=meta['unum']+(0 if meta['side']=='l' else 11)
        data=pd.read_csv(file.parent/'agent_cycle.csv.gz',usecols=agent_cols)
        data=data[data.stopped.eq(0)&data.decision_made.eq(1)&data.cycle.between(lo,hi)].set_index('cycle')
        data=data[np.isfinite(data[['tru_pos_x','tru_pos_y','tru_vel_x','tru_vel_y','tru_body','tru_neck']].to_numpy(float)).all(1)]
        agents[oid]=data;names[meta['side']]=meta['team_name']
    assert len(agents)==22
    o=pd.read_csv(observer/'objects.csv.gz',usecols=['cycle','stopped','object_id','tru_pos_x','tru_pos_y','tru_vel_x','tru_vel_y'])
    ball=o[o.object_id.eq(0)&o.stopped.eq(0)&o.cycle.between(lo,hi)].set_index('cycle')
    ball=ball[np.isfinite(ball[['tru_pos_x','tru_pos_y','tru_vel_x','tru_vel_y']].to_numpy(float)).all(1)]
    own_id=m['unum']+(0 if m['side']=='l' else 11)
    available=set(ball.index)
    for data in agents.values():available &= set(data.index)
    cycles=sorted(available);assert a.cycle in cycles
    rcg=out/'reconstructed.rcg';dlog=out/f"{m['team_name']}-{m['unum']}.log"
    pred=pd.read_csv(observer/'player_prediction.csv.gz')
    pred=pred[pred.object_id.eq(a.object_id)&pred.stopped.eq(0)&pred.cycle.isin(cycles)].set_index('cycle')
    records=[]
    with rcg.open('w') as f:
        f.write('ULG4\n')
        # Physical/view defaults are for display; all shown positions come from CSV truth.
        for cycle in cycles:
            r=agents[own_id].loc[cycle]
            f.write(f"(playmode {cycle} {r.game_mode})\n")
            f.write(f"(team {cycle} {names['l']} {names['r']} {int(r.score_l)} {int(r.score_r)})\n")
            b=ball.loc[cycle]
            text=f"(show {cycle} ((b) {value(b.tru_pos_x):.6f} {value(b.tru_pos_y):.6f} {value(b.tru_vel_x):.6f} {value(b.tru_vel_y):.6f})"
            for oid,data in sorted(agents.items()):
                p=data.loc[cycle];side='l' if oid<=11 else 'r';unum=(oid-1)%11+1
                state=1|(8 if value(p.tru_goalie) else 0)
                width={'narrow':60,'normal':120,'wide':180}.get(p.sb_view_width,120)
                text+=f" (({side} {unum}) {int(value(p.tru_player_type))} {state:#x}"
                for key in ['tru_pos_x','tru_pos_y','tru_vel_x','tru_vel_y','tru_body','tru_neck']:text+=f' {value(p[key]):.6f}'
                text+=f" (v h {width}) (s {value(p.tru_stamina,8000):.3f} {value(p.tru_effort,1):.6f} {value(p.tru_recovery,1):.6f}) (c 0 0 0 0 0 0 0 0 0 0 0))"
            f.write(text+')\n')
    sign=1 if m['side']=='l' else -1
    with dlog.open('w') as f:
        for cycle,r in pred.iterrows():
            if pd.isna(r.stock_x) or pd.isna(r.used_x):continue
            raw=(r.stock_x*sign,r.stock_y*sign);used=(r.used_x*sign,r.used_y*sign)
            prefix=f'{cycle},0 4 '
            status='APPLY' if r.applied==1 else 'WM'
            label=f"recorded object{a.object_id} {status} pc={int(r.stock_count)}>{int(r.decision_count)} originalErr={r.stock_error:.3f}m usedErr={r.used_error:.3f}m"
            f.write(prefix+f'c {raw[0]:.6f} {raw[1]:.6f} 0.3 #ff9f1c\n')
            f.write(prefix+f'r {used[0]-.3:.6f} {used[1]-.3:.6f} .6 .6 #00d5ff\n')
            f.write(prefix+f'l {raw[0]:.6f} {raw[1]:.6f} {used[0]:.6f} {used[1]:.6f} #00d5ff\n')
            f.write(prefix+f'm {used[0]+.45:.6f} {used[1]-.45:.6f} (c #00d5ff) {label}\n')
            f.write(prefix+'M [RECORDED-GAP] '+label+'\n')
            records.append(dict(cycle=int(cycle),stock_error_m=float(r.stock_error),used_error_m=float(r.used_error),stock=[float(r.stock_x),float(r.stock_y)],used=[float(r.used_x),float(r.used_y)],truth=[float(r.true_x),float(r.true_y)]))
    info=dict(source=str(game.resolve()),observer=a.observer,target_object_id=a.object_id,focus_cycle=a.cycle,
              cycles=cycles,rcg=str(rcg.resolve()),dlog=str(dlog.resolve()),records=records,
              caveat='Reconstructed visualization from all 22 agents same-cycle truth; not original RCG. Player types/views copied from self logs; action counters and unrecorded state bits are placeholders. No model/decision data was changed.')
    (out/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
    (out/'README.md').write_text(f'''# Recorded position-error example

Open `reconstructed.rcg` in SoccerWindow, then open this directory in Debug Message
for side {m['side']}, observer {m['unum']}, with WM enabled. Seek cycle {a.cycle}.
Orange circle is original WM; cyan square is the recorded used position. The
standard monitor players are truth reconstructed from recorded fullstate columns.

This is a reconstructed clip, not the original server log. Action counters and
unrecorded action-state bits are placeholders; do not use it for action analysis.
The model was not rerun to produce these overlays. See provenance.json.
''')
    print(json.dumps({k:v for k,v in info.items() if k not in ['cycles','records']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game',type=Path);p.add_argument('observer');p.add_argument('object_id',type=int);p.add_argument('cycle',type=int);p.add_argument('output',type=Path);p.add_argument('--radius',type=int,default=20);export(p.parse_args())
