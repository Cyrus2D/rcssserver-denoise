#!/usr/bin/env python3
"""Summarize the fixed exploratory scaling study without choosing on test."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'baseline64':'Original 64×1','continued64':'64×1, 20 epochs','wide128':'128×1, 10 epochs','deep64x2':'64×2, 10 epochs','data40':'64×1, 40 train games'}

def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('report',type=Path);args=p.parse_args()
    data=json.loads(args.results.read_text());models=data['models'];out=args.report.parent;out.mkdir(parents=True,exist_ok=True)
    names=list(models)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8));x=np.arange(len(names));width=.25
    for axis,field,label,offset in [(axes[0],'train_probe_mean_m','Train-pool probe',-.25),(axes[0],'selection_mean_m','Selection10',0.)]:
        axis.bar(x+offset,[100*models[k][field] for k in names],width,label=label)
    axes[0].bar(x+.25,[100*models[k]['test']['mean_m'] for k in names],width,label='Old-test subset')
    axes[0].set(ylabel='Mean position error (cm)',xticks=x,xticklabels=[LABELS[k].replace(', ','\n') for k in names]);axes[0].tick_params(axis='x',rotation=15);axes[0].legend(fontsize=8)
    axes[1].bar(x,[models[k]['test']['worse_over_25cm_pct'] for k in names],color='#be6a44')
    axes[1].set(ylabel='Samples made >25 cm worse than WM (%)',xticks=x,xticklabels=[LABELS[k].replace(', ','\n') for k in names]);axes[1].tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(out/'scaling-comparison.svg');fig.savefig(out/'scaling-comparison.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for name in names:
        f=args.results.parent/name/'training.json'
        if not f.exists():continue
        rows=json.loads(f.read_text());offset=10 if name=='continued64' else 0
        ax.plot([r['epoch']+offset for r in rows],[100*r['selection_mean_m'] for r in rows],marker='o',label=LABELS[name])
    ax.set(xlabel='Epoch (continuation starts after original epoch10)',ylabel='Selection10 mean error (cm)');ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/'scaling-learning-curves.svg');fig.savefig(out/'scaling-learning-curves.png',dpi=140);plt.close(fig)
    best=min(models,key=lambda k:models[k]['selection_mean_m'])
    lines=['# Exp5-S1 capacity, depth, training and data-diversity study','',
           '**Exploratory reuse of previously inspected test games; one seed.** All model settings were fixed before this study’s test scoring. Checkpoints were selected only on the original ten selection games.','',
           'The original training curve was still improving at epoch10 (selection mean70.82→56.41 cm). That motivates an actual comparison rather than assuming more capacity/data will help.','',
           '| Variant | Parameters | Train-pool probe cm | Selection cm | Old-test subset cm | p95 cm | Worse >0 cm % | Worse >25 cm % | Max added error m |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in names:
        m=models[k];t=m['test'];lines.append(f'| {LABELS[k]} | {m["parameters"]:,} | {100*m["train_probe_mean_m"]:.2f} | {100*m["selection_mean_m"]:.2f} | {100*t["mean_m"]:.2f} | {100*t["p95_m"]:.2f} | {t["worse_over_0cm_pct"]:.2f} | {t["worse_over_25cm_pct"]:.2f} | {t["max_added_error_m"]:.2f} |')
    lines += ['',f'Checkpoint-selection winner: **{LABELS[best]}**, chosen from selection error, not test error. Test subset contains {models[names[0]]["test"]["n"]:,} identical, uniformly sampled eligible target-cycles for every variant. Main scores are sample-weighted.','',
              '![Training, selection/test and regression tradeoff](scaling-comparison.svg)','',
              '![Selection learning curves](scaling-learning-curves.svg)','',
              '## Controlled questions','',
              '- **More training:** continue the saved64-unit weights for ten additional epochs at lower learning rate3e-4→5e-5. AdamW is restarted because the original optimizer state was not saved. Total sampled endpoints become4.096M.',
              '- **More width:**128 hidden units, one GRU layer; fresh ten-epoch training with the same2.048M endpoint budget and endpoint RNG sequence as the original160-game model.',
              '- **More depth:**64 hidden units and two GRU layers; same fresh-training budget. No extra dropout is introduced, so this isolates layer/capacity changes under the fixed recipe.',
              '- **More source-game diversity:**40 versus160 training games at the same2.048M endpoint budget. The40 games are a fixed seeded subset of the original160, with all their observers; this is not a comparison of40 versus160 fully enumerated datasets.',
              '', '## Interpretation and limits','',
              'Compare the train/selection errors and curves together. Falling train and selection error supports remaining optimization/capacity headroom. A wider/deeper model is not automatically better, and lower average error does not imply fewer harmful replacements of already-accurate WM.',
              '', 'The50k train-pool probe is shared across all variants and drawn from the original160 training games. For the40-game model it therefore includes games outside its fit subset: call that column a common training-pool diagnostic, not its in-sample training error. The160-game variants provide the clean train-versus-selection comparison.',
              '', 'More raw games, more sampled endpoints and more optimizer updates are different interventions. Only a small part of the227M eligible training endpoints was sampled in the original run. If extra data are collected, target distinct difficult situations rather than merely duplicating common trajectories; choose sampling/weights on training and validation, then retain natural sample weights at evaluation.',
              '', 'This study does not establish new-team generalization or live-game benefit. The old20 test games have already been inspected in earlier analyses, so these results are exploratory; confirm any chosen recipe on a fresh independent batch before deployment. All raw metrics, exact40-game subset, indices and configs are stored beside results.json.']
    args.report.write_text('\n'.join(lines)+'\n')
    print('Wrote',args.report)

if __name__=='__main__':main()
