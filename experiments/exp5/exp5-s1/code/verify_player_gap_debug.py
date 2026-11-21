#!/usr/bin/env python3
"""Check all observers' colored debug geometry against decision CSV positions."""
import argparse,json,re
from pathlib import Path
import numpy as np
import pandas as pd

def balanced(text):
    depth=0;quoted=False
    for c in text:
        if c=='"':quoted=not quoted
        elif not quoted:
            depth+=int(c=='(')-int(c==')')
            assert depth>=0
    assert depth==0 and not quoted

def verify(root,target='all'):
    result=dict(observers=0,packets=0,max_packet_bytes=0,position_pairs=0,applied=0,clipped_packets=0)
    for meta_file in sorted((root/'player'/root.name).glob('*/meta.json')):
        m=json.loads(meta_file.read_text());options=m['run_options'];team=m['team_name'];unum=m['unum']
        assert m['final'] and m['status']=='ok' and options['player_gap_debug']=='both'
        directory=root/'debug'/team;log=directory/f'{team}-{unum}.log';dcl=directory/f'{team}-{unum}.dcl'
        assert log.exists() and dcl.exists()
        text=log.read_text();assert 'cache=' in text and '#ff9f1c' in text and '#00d5ff' in text
        cache_counts=re.findall(r'cache=(\d+)/(\d+)',text)
        assert cache_counts and all(int(total)==int(unique) and int(total)<=26 for total,unique in cache_counts)
        assert re.findall(r'cache_ok=(\d+)',text) and all(v=='1' for v in re.findall(r'cache_ok=(\d+)',text))
        frames={}
        for line in dcl.read_text().splitlines():
            if not line.startswith('%% debug ['):continue
            message=line[len('%% debug ['):-1]
            balanced(message);size=len(message.encode())+1
            assert size<=7800
            result['max_packet_bytes']=max(result['max_packet_bytes'],size);result['packets']+=1
            result['clipped_packets']+=int('OVERLAY CLIPPED' in message)
            key=tuple(map(int,re.search(r'\(time (\d+),(\d+)\)',message).groups()))
            circles=[];squares=[]
            for kind,body,color in re.findall(r'\((circle|rect) ([^()]*) "(#[a-fA-F0-9]+)"\)',message):
                values=list(map(float,body.split()))
                if kind=='circle' and color=='#ff9f1c':circles.append(values[:2]);assert abs(values[2]-.3)<1e-8
                if kind=='rect' and color=='#00d5ff':
                    squares.append([(values[0]+values[2])/2,(values[1]+values[3])/2])
                    assert abs(values[2]-values[0]-.6)<.002 and abs(values[3]-values[1]-.6)<.002
            if target!='all':assert len(circles)<=1 and len(squares)<=1
            frames[key]=(np.array(circles),np.array(squares))
        pred=pd.read_csv(meta_file.parent/'player_prediction.csv.gz')
        if not len(pred):
            o=pd.read_csv(meta_file.parent/'objects.csv.gz',usecols=['cycle','stopped','object_id','track_match','trk_stock_pos_x','trk_stock_pos_y','trk_pos_x','trk_pos_y','trk_prediction_active'])
            pred=o[o.object_id.gt(0)&o.track_match.eq('unum')].rename(columns={'trk_stock_pos_x':'stock_x','trk_stock_pos_y':'stock_y','trk_pos_x':'used_x','trk_pos_y':'used_y','trk_prediction_active':'applied'})
        valid=pred.stock_x.notna()&pred.used_x.notna()
        pred=pred[valid]
        if target!='all':
            kind,n=target.split(':');side=m['side'] if kind=='teammate' else ('r' if m['side']=='l' else 'l')
            oid=int(n)+(0 if side=='l' else 11);pred=pred[pred.object_id.eq(oid)]
        sign=1 if m['side']=='l' else -1
        for key,group in pred.groupby(['cycle','stopped']):
            if key not in frames:continue # BeforeKickOff/TimeOver are not saved in .dcl.
            circles,squares=frames[key]
            assert len(circles) and len(squares),(team,unum,key)
            for columns,shapes in [(['stock_x','stock_y'],circles),(['used_x','used_y'],squares)]:
                points=group[columns].to_numpy()*sign
                distance=np.linalg.norm(points[:,None,:]-shapes[None,:,:],axis=-1).min(1)
                assert distance.max()<.002,(team,unum,key,columns,float(distance.max()))
            result['position_pairs']+=len(group);result['applied']+=int(group.applied.eq(1).sum())
        result['observers']+=1
    assert result['observers']==22 and result['position_pairs']>0 and result['clipped_packets']==0
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--target',default='all');p.add_argument('--output',type=Path);a=p.parse_args()
    r=verify(a.run,a.target)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
