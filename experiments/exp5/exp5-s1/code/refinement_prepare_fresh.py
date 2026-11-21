"""Build an evaluation-only manifest from the separately collected new games."""
from pathlib import Path
import argparse,json,hashlib
from concurrent.futures import ProcessPoolExecutor
from prepare import build_observer
from refinement_models import atomic_json

def job(task):return build_observer(*task)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fresh',type=Path);p.add_argument('original_dataset',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    completed=json.loads((a.fresh/'collection-results.json').read_text());games=[a.fresh/'player'/v['game'] for v in completed]
    paths=[p.parent for g in games for p in sorted(g.glob('*/meta.json'))]
    assert len(paths)==22*len(completed)
    a.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as pool:records=list(pool.map(job,[(p,a.out) for p in paths]))
    manifest=json.loads((a.original_dataset/'manifest.json').read_text())
    manifest.update(logs=str((a.fresh/'player').resolve()),split={g.name:'test' for g in games},split_seed=None,
                    observers=[dict(r,split='test') for r in records],
                    provenance='New games generated only for confirmation; never included in fitting or threshold/model selection.')
    atomic_json(a.out/'manifest.json',manifest)
    print('Prepared independent confirmation:',len(games),'games',len(records),'observers',sum(r['eligible'] for r in records),'eligible endpoints',flush=True)
