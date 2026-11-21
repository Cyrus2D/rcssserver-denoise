"""Publish saved offline studies without rerunning model inference."""
from pathlib import Path
import json,re,shutil,hashlib
from followup_reporting import embed
from report import write_html

def publish(step,logs):
    out=step/'reports'
    sources={'confidence-study/results.json':'followup-confidence.json',
             'scaling-study/results.json':'followup-capacity.json',
             'selector-study/results.json':'followup-selectors.json',
             'selector-study/multi-expert/results.json':'followup-multi-expert.json',
             'refinement/distance-results.json':'refinement-distance.json',
             'refinement/routing-results.json':'refinement-routing.json',
             'refinement/blend-results.json':'refinement-blend.json',
             'refinement/final-results.json':'refinement-final.json',
             'refinement/fresh-results.json':'refinement-fresh.json',
             'refinement/frozen-policies.json':'refinement-frozen-policies.json',
             'refinement/protocol.json':'refinement-protocol.json',
             'refinement/safety-results.json':'refinement-safety.json',
             'refinement/fresh-verification.json':'refinement-fresh-verification.json',
             'refinement/study-requirements.json':'refinement-study-requirements.json',
             'refinement/training-results.json':'refinement-training.json',
             'refinement/tree-controls.json':'refinement-tree-controls.json',
             'refinement/tree-controls-fresh.json':'refinement-tree-fresh.json',
             'refinement/pruned-trees.json':'refinement-pruned-trees.json',
             'refinement/runtime.json':'refinement-runtime.json'}
    for source,target in sources.items():
        p=logs/source
        if p.exists():shutil.copyfile(p,out/target)
    d=json.loads((out/'metrics.json').read_text())
    plots={p.stem:embed(p,p.stem) for p in out.glob('*.svg')}
    write_html(out,d,plots)
    latest=out/'refinement-fresh.json'
    if not latest.exists():latest=out/'refinement-final.json'
    if latest.exists():
        study=json.loads(latest.read_text());label='New independent10 games' if study['cohort']=='fresh' else 'Original20 test games (exploratory reuse)'
        with (out/'report.md').open('a') as f:
            f.write('\n## Latest refinement — '+label+'\n\n| Policy | All mean cm | Under50m mean cm | Under50m made >25cm worse % |\n|---|---:|---:|---:|\n')
            for key,v in study['policies'].items():
                a=v['metrics']['all'];n=v['metrics']['under50'];f.write(f'| {key} | {100*a["mean"]:.2f} | {100*n["mean"]:.2f} | {n["worse"]["25"]:.2f} |\n')

    p=out/'provenance.json';provenance=json.loads(p.read_text())
    provenance['followup_code_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (step/'code').glob('*.py')}
    provenance['followup_result_sha256']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in sources.values() if (out/name).exists()}
    provenance['followup_evaluation']='Exploratory reuse of previously inspected test games; no live code changes.'
    p.write_text(json.dumps(provenance,indent=2)+'\n')
    print('Published',out/'report.html')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('step',type=Path);p.add_argument('logs',type=Path);a=p.parse_args();publish(a.step,a.logs)
