"""Reproduce the complete staged offline refinement and new-game check."""
from pathlib import Path
import argparse,subprocess,sys,json

def main(a):
    step=Path(__file__).resolve().parent.parent;code=step/'code';out=a.logs/'refinement';fresh=a.logs/'refinement-fresh';freshdata=a.logs/'refinement-fresh-dataset'
    def run(script,*args):subprocess.run([sys.executable,str(code/script),*map(str,args)],check=True)
    def publish():run('publish_followups.py',step,a.logs)
    complete=['final-results.json','fresh-results.json','safety-results.json','tree-controls.json','tree-controls-fresh.json','pruned-trees.json','frozen-policies.json']
    if all((out/name).exists() for name in complete):
        import hashlib
        frozen=json.loads((out/'frozen-policies.json').read_text())
        frozen_hash=hashlib.sha256((out/'frozen-policies.json').read_bytes()).hexdigest()
        protocol=json.loads((out/'protocol.json').read_text())
        if hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest()!=protocol['dataset_sha256']:
            raise ValueError('Dataset changed; use a new study directory')
        for name,expected in frozen['provenance']['checkpoint_sha256'].items():
            path=a.logs/'scaling-study/wide128/best.pt' if name=='wide128' else out/name/'best.pt'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen checkpoint changed')
        for file in ['final-results.json','fresh-results.json']:
            if json.loads((out/file).read_text())['frozen_policy_sha256']!=frozen_hash:raise ValueError('Frozen-policy result mismatch')
        publish();print('Completed study verified; refreshed report without refitting or inference.');return
    run('refinement_models.py','prepare',a.dataset,a.logs,out)
    run('refinement_select.py','fit',out,'--model','wide128');run('refinement_select.py','aggregate',out);publish()
    run('refinement_distance.py',a.logs,out/'distance-results.json');publish()
    run('refinement_models.py','train',a.dataset,a.logs,out)
    run('refinement_models.py','cache',a.dataset,a.logs,out)
    names=json.loads((out/'training-results.json').read_text())
    for name in names:
        run('refinement_select.py','fit',out,'--model',name);run('refinement_select.py','aggregate',out);publish()
    run('refinement_select.py','blend',out);publish()
    run('refinement_final.py','freeze',a.dataset,out)
    run('refinement_final.py','evaluate',a.dataset,out);publish()
    run('refinement_safety.py','test',out);publish()
    run('refinement_collect.py',fresh,'--games','10','--jobs','2')
    run('refinement_prepare_fresh.py',fresh,a.dataset,freshdata)
    run('refinement_final.py','fresh',a.dataset,out,'--fresh-dataset',freshdata);publish()
    run('refinement_safety.py','fresh',out,'--fresh-dataset',freshdata);publish()
    # Extra controls are explicitly exploratory after the primary confirmation.
    run('refinement_tree_control.py',out);publish()
    run('refinement_tree_fresh.py',out,freshdata)
    run('refinement_export_tree.py',out);publish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('logs',type=Path);main(p.parse_args())
