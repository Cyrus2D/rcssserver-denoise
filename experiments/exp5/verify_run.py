#!/usr/bin/env python3
"""Verify finalized metadata and actual hearing payload absence."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import gzip
import json
from live_audit import select_games

def verify(games):
    counts=Counter(hear_rows=0,content_rows=0)
    metas=[]
    for game in games:
        for p in game.glob('*/meta.json'):
            metas.append(json.loads(p.read_text()))
            path=p.parent/'hear.csv.gz'
            counts['hear_files']+=1
            with gzip.open(path,'rt') as f:
                for row in csv.DictReader(f):
                    counts['hear_rows']+=1
                    counts['sender/'+row['sender_kind']]+=1
                    counts['content_rows']+=int(row.get('has_content')=='1')
    counts['metadata_files']=len(metas)
    counts['status_ok_final']=sum(m.get('final') and m.get('status')=='ok' for m in metas)
    bad=[m['game_id']+'/'+m['side']+'_'+str(m['unum']) for m in metas if m.get('config',{}).get('use_fullstate')]
    return dict(games=[g.name for g in games],counts=counts,fullstate_as_input_violations=bad,
                schema_versions=sorted({m['schema_version'] for m in metas}),
                view_policies=sorted({m.get('run_options',{}).get('view_policy','unknown') for m in metas}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--expected-games',type=int,default=200)
    args=p.parse_args();games,skipped=select_games(args.logs)
    if len(games)!=args.expected_games or skipped:
        raise ValueError(f'Expected {args.expected_games} complete games; got {len(games)}, skipped {len(skipped)}')
    data=verify(games);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    if data['fullstate_as_input_violations'] or data['counts']['content_rows']:
        raise ValueError('Unexpected fullstate input or hearing payloads; inspect output')
    print(f'PASS: {len(games)} games, {data["counts"]["metadata_files"]} finalized observers, {data["counts"]["content_rows"]} hearing payload rows')

if __name__=='__main__':main()
