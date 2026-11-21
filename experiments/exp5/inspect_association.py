#!/usr/bin/env python3
"""Plot a named track against same-cycle truth, without changing live state."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('observer',type=Path);p.add_argument('cycle',type=int);p.add_argument('target',type=int);p.add_argument('output',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    a=pd.read_csv(args.observer/'agent_cycle.csv.gz',low_memory=False)
    a=a[a.cycle.eq(args.cycle)&a.stopped.eq(0)].iloc[0]
    o=pd.read_csv(args.observer/'objects.csv.gz',low_memory=False)
    o=o[o.cycle.eq(args.cycle)&o.stopped.eq(0)].copy()
    t=o[o.object_id.eq(args.target)].iloc[0]
    observer=np.array([a.tru_pos_x,a.tru_pos_y]);truth=np.array([t.tru_pos_x,t.tru_pos_y]);track=np.array([t.trk_pos_x,t.trk_pos_y])
    distances=np.hypot(o.tru_pos_x-track[0],o.tru_pos_y-track[1]);other=o.loc[distances.idxmin()]
    width=dict(narrow=60,normal=120,wide=180)[a.sb_view_width if pd.notna(a.sb_view_width) else a.view_width]
    face=a.tru_body+a.tru_neck
    r=max(np.linalg.norm(truth-observer),np.linalg.norm(track-observer))+3
    fig,ax=plt.subplots(figsize=(8,6))
    ax.add_patch(Wedge(observer,r,face-width/2,face+width/2,facecolor='#e1eeee',edgecolor='#8bb6b8',alpha=.7))
    ax.scatter(*observer,c='#087e8b',s=130,label='Observer true position',zorder=4)
    ax.annotate('Observer '+args.observer.name,observer,xytext=(10,-20),textcoords='offset points')
    ax.scatter(*truth,c='#5179b0',s=100,label='Target true position',zorder=4)
    label=lambda n: 'Ball' if n==0 else ('L'+str(n) if n<=11 else 'R'+str(n-11))
    ax.annotate('Target truth: '+label(args.target),truth,xytext=(-12,12),textcoords='offset points',ha='right')
    ax.scatter(*track,c='#cf6840',s=180,marker='x',lw=3,label='Stored target track',zorder=5)
    ax.annotate('Stored '+label(args.target)+' at true '+label(int(other.object_id))+' position\n(exact anonymous observation)',track,xytext=(15,8),textcoords='offset points')
    ax.annotate('',xy=track,xytext=truth,arrowprops=dict(arrowstyle='<->',color='#cf6840',lw=2))
    midpoint=(track+truth)/2
    ax.text(*midpoint,f'  {t.n_trk_pos_d:.2f} m identity error',rotation=-60,color='#a34b2d')
    points=np.vstack([observer,truth,track]);low=points.min(axis=0)-4;high=points.max(axis=0)+4
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal',xlabel='Global x (m)',ylabel='Global y (m)',
           title=f'Exact measurement, wrong retained identity\n{args.observer.parent.name}, {args.observer.name}, cycle {args.cycle}')
    ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=9);fig.tight_layout()
    fig.savefig(args.output/'identity-case.svg');fig.savefig(args.output/'identity-case.png',dpi=140);plt.close(fig)
    details=dict(game=args.observer.parent.name,observer=args.observer.name,cycle=args.cycle,target=args.target,
                 target_error_m=float(t.n_trk_pos_d),target_position_count=int(t.trk_pos_count),target_seen=bool(t.seen==1),
                 nearest_other_truth_id=int(other.object_id),other_form=str(other.form),other_unum_visible=bool(other.unum_visible==1),
                 other_localization_error_m=float(other.n_ls_pos_d),distance_between_stored_track_and_other_truth_m=float(distances.min()),
                 actual_view_degrees=width,mode=str(a.game_mode))
    (args.output/'identity-case.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps(details,indent=2))

if __name__=='__main__':main()
