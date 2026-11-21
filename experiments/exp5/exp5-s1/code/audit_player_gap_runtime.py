#!/usr/bin/env python3
"""Check live console/order logs and compare game outcomes with prior batches."""
import argparse,csv,json,re
from collections import Counter
from pathlib import Path
import numpy as np


def load_batch(logs):
    rows=[]
    for file in sorted(logs.glob('*/l_1/meta.json')):
        m=json.loads(file.read_text())
        if not m.get('final') or m.get('status')!='ok':continue
        s=m['server_param']
        rows.append(dict(game=m['game_id'],left=m['score_left'],right=m['score_right'],
                         random_seed=s.get('random_seed'),order_seed=s.get('command_order_seed')))
    return rows


def outcomes(rows):
    a=np.array([v['left']-v['right'] for v in rows],float)
    return dict(games=len(rows),goals_left=sum(v['left'] for v in rows),goals_right=sum(v['right'] for v in rows),
                wins=int((a>0).sum()),draws=int((a==0).sum()),losses=int((a<0).sum()),mean_goal_difference=float(a.mean()))


def audit(a):
    rows=load_batch(a.logs);counts=Counter();errors=[];missing=[];score_mismatch=[];hist=Counter()
    for m in rows:
        gid=m['game']
        for side in ['L','R']:
            file=a.autotest/'result.d'/(gid+'_'+side)
            if not file.exists():missing.append(str(file));continue
            for line in file.read_text(errors='replace').splitlines():
                if re.search(r'inference_error|Cannot load player-gap|Player gap prediction disabled|unknown column name|segmentation fault|terminate called|unknown parameter|\*\*\*ERROR',line,re.I):
                    errors.append(dict(file=file.name,text=line[:300]))
        file=a.autotest/'result.d'/gid
        if file.exists():
            found=re.findall(r'Score:\s*(\d+)\s*-\s*(\d+)',file.read_text(errors='replace'))
            if not found or tuple(map(int,found[-1]))!=(m['left'],m['right']):score_mismatch.append(gid)
        else:missing.append(str(file))
        file=a.autotest/'log.d'/(gid+'.command-order.csv')
        if not file.exists():missing.append(str(file));continue
        for record in csv.DictReader(file.open()):
            entries=record['order'].split()
            if not entries:continue
            counts['nonempty_rounds']+=1
            counts['left_first']+=entries[0].startswith('l')
            counts['right_first']+=entries[0].startswith('r')
            hist[len(entries)]+=1
    rng=np.random.default_rng(50703)
    d=np.array([r['left']-r['right'] for r in rows],float)
    bootstrap=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
    result=dict(current=outcomes(rows),command_order=dict(counts),command_batch_sizes=dict(hist),
                runtime_errors=errors,missing_files=missing,score_mismatches=score_mismatch,
                goal_diff_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),references={})
    for label,path in a.reference:
        prior=load_batch(Path(path));pd=np.array([r['left']-r['right'] for r in prior],float)
        pb=pd[rng.integers(len(pd),size=(20000,len(pd)))].mean(1)
        overlap=set((r['random_seed'],r['order_seed']) for r in rows)&set((r['random_seed'],r['order_seed']) for r in prior)
        result['references'][label]=dict(**outcomes(prior),change_in_mean_goal_difference=float(d.mean()-pd.mean()),
            independent_game_bootstrap95_for_change=np.quantile(bootstrap-pb,[.025,.975]).tolist(),
            overlapping_simulator_and_order_seed_pairs=len(overlap),paired=False,
            goal_diff_bootstrap95=np.quantile(pb,[.025,.975]).tolist())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('logs',type=Path);p.add_argument('autotest',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--reference',nargs=2,action='append',default=[],metavar=('LABEL','LOGS'))
    audit(p.parse_args())
