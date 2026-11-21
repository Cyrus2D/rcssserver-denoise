#!/usr/bin/env python3
"""Visual comparison of position predictors, including observer regressions."""
import argparse,csv,html,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

KINDS=['point','gaussian','mixture']
LABELS={'point':'Single position','gaussian':'Position + uncertainty','mixture':'Three possible positions'}
POLICY_LABELS={'raw':'Use full model prediction','guarded':'Move at most 25 cm from librcsc (with gates)'}
COLORS=['#087e8b','#527bb3','#ca7041']
AGES=['1','2','3','4','5','6-10','11-20','21-30']
# Allow for float32 distance rounding; this is 1/1000 of the smallest displayed band.
CHANGE_TOLERANCE_M=1e-5
CHANGE_BUCKETS=[('better','Better (any decrease)','#087e8b'),
                ('unchanged','No change','#d6dfe3'),
                ('worse_0_1cm','Worse by up to 1 cm','#fff0ba'),
                ('worse_1_5cm','Worse by >1 to 5 cm','#f3d15c'),
                ('worse_5_25cm','Worse by >5 to 25 cm','#efa04a'),
                ('worse_25_50cm','Worse by >25 to 50 cm','#da703a'),
                ('worse_over_50cm','Worse by >50 cm','#b74643')]

def error_change_breakdown(error,baseline):
    """Disjoint intervals; all increases are additional error, not total error."""
    error=np.asarray(error,dtype=np.float64);baseline=np.asarray(baseline,dtype=np.float64)
    if error.shape!=baseline.shape or not np.isfinite(error).all() or not np.isfinite(baseline).all():
        raise ValueError('Error arrays must be paired, finite and equally shaped')
    delta=error-baseline;n=len(delta);tol=CHANGE_TOLERANCE_M
    masks=[delta < -tol,np.abs(delta)<=tol,(delta>tol)&(delta<=.01+tol),
           (delta>.01+tol)&(delta<=.05+tol),(delta>.05+tol)&(delta<=.25+tol),
           (delta>.25+tol)&(delta<=.5+tol),delta>.5+tol]
    counts=[int(mask.sum()) for mask in masks]
    assert sum(counts)==n
    return {key:dict(n=count,pct=100*count/n if n else 0.)
            for (key,_,_),count in zip(CHANGE_BUCKETS,counts)}

def build_omission_study(source):
    """Post-hoc scoring/filter diagnostic using unchanged cached predictions."""
    with np.load(source/'cohort.npz') as cohort:
        anonymous=(cohort['flags']&8)!=0
        unnamed=cohort['count']==0
        baseline=cohort['baseline']
    excluded=anonymous|unnamed;keep=~excluded;n=len(baseline)
    def summarize(error,stock):
        error=np.asarray(error,dtype=np.float64);stock=np.asarray(stock,dtype=np.float64)
        gain=stock-error
        return dict(n=len(error),mean=float(error.mean()),baseline_mean=float(stock.mean()),
                    gain=float(gain.mean()),gain_pct=float(100*gain.mean()/stock.mean()),
                    hurt_over_1cm_pct=float(100*(gain<-.01).mean()),max_added_error=float(max(0,-gain.min())))
    results={}
    for kind in KINDS:
        with np.load(source/(kind+'-test.npz')) as prediction:error=prediction['error']
        results[kind]=dict(all_samples=summarize(error,baseline),
                           exclude_both=summarize(error[keep],baseline[keep]),
                           keep_wm_for_both=summarize(np.where(excluded,baseline,error),baseline))
    return dict(total=n,anonymous_current=int(anonymous.sum()),unnamed_wm_refresh=int(unnamed.sum()),
                overlap=int((anonymous&unnamed).sum()),excluded_union=int(excluded.sum()),
                excluded_pct=float(100*excluded.mean()),remaining=int(keep.sum()),models=results,
                training_changed=False,classification='Post-hoc diagnostic; no retraining or test-driven weight/gate tuning.',
                filter_definition='Union of anonymous_current and unnamed_wm_refresh. In this paired no-current-named-sighting cohort, unnamed_wm_refresh equals WM position count 0.',
                limitation='Anonymous entries are attached to true target IDs by the evaluation logger. Exact target-level union fallback is a diagnostic, not a directly deployable identity rule. WM position count is observable.')

def render(source,out):
    out.mkdir(parents=True,exist_ok=True)
    d=json.loads((source/'metrics.json').read_text());models=d['models'];plots={}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(name,fig):
        fig.tight_layout(pad=2);fig.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
        raw=(out/(name+'.svg')).read_text();raw=raw[raw.index('<svg'):]
        for ident in set(re.findall(r'\bid="([^"]+)"',raw)):
            raw=raw.replace('id="'+ident+'"','id="'+name+'-'+ident+'"').replace('#'+ident+'"','#'+name+'-'+ident+'"').replace('#'+ident+')','#'+name+'-'+ident+')')
        plots[name]=raw
    def value(kind,group='all',policy='raw'):return models[kind]['groups'][policy].get(group)
    base=value('point')['baseline_mean']
    cohort=np.load(source/'cohort.npz');baseline=cohort['baseline']
    breakdown={}
    for kind in KINDS:
        with np.load(source/(kind+'-test.npz')) as prediction:
            raw_error=prediction['error']
        guarded_error=np.load(source/(kind+'-guarded-error.npy'),mmap_mode='r')
        breakdown[kind]={'raw':error_change_breakdown(raw_error,baseline),
                         'guarded':error_change_breakdown(guarded_error,baseline)}
    d['error_change_buckets']=breakdown
    d['error_change_bucket_definition']=dict(tolerance_m=CHANGE_TOLERANCE_M,
        change='model error minus librcsc error; positive means worse',
        intervals='better < -tolerance; no change within tolerance; worse up to .01, .05, .25, .50 meters and above; tolerance also applies at boundaries')
    (out/'error-change-buckets.json').write_text(json.dumps(dict(definition=d['error_change_bucket_definition'],models=breakdown),indent=2)+'\n')
    with (out/'error-change-buckets.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','policy','change','samples','percentage'])
        for kind,policies in breakdown.items():
            for policy,groups in policies.items():
                for key,label,_ in CHANGE_BUCKETS:writer.writerow([LABELS[kind],POLICY_LABELS[policy],label,groups[key]['n'],groups[key]['pct']])
    fig,axes=plt.subplots(1,2,figsize=(15,6.6));x=np.arange(3)
    for policy,offset,label in [('raw',-.18,'Use full model prediction'),('guarded',.18,'Move at most 25 cm from librcsc\n(with validation-selected gates)')]:
        axes[0].bar(x+offset,[100*value(k,policy=policy)['mean'] for k in KINDS],.34,label=label)
    axes[0].axhline(base*100,c='#8c394b',ls='--',label='Original librcsc');axes[0].set(title='Average position error: lower is better',ylabel='Mean position error (cm)',xticks=x,xticklabels=['Single','Uncertainty','Mixture']);axes[0].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False)
    left=np.zeros(3)
    for key,label,color in CHANGE_BUCKETS:
        widths=np.array([breakdown[k]['raw'][key]['pct'] for k in KINDS])
        axes[1].barh(x,widths,left=left,color=color,label=label,height=.65)
        for i,width in enumerate(widths):
            if width>=4:axes[1].text(left[i]+width/2,i,f'{width:.1f}%',ha='center',va='center',fontsize=9,color='white' if key in ['better','worse_over_50cm'] else '#263541')
        left+=widths
    axes[1].set(title='Full model: how much does each sample change?',yticks=x,yticklabels=['Single','Uncertainty','Mixture'],xlabel='Share of paired samples (%)',xlim=(0,100))
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=9,frameon=False)
    save('overall-benefit-harm',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    baseline=[100*value('point','gap/'+age)['baseline_mean'] for age in AGES]
    axes[0].plot(AGES,baseline,c='#8c394b',ls='--',marker='o',label='Stock librcsc')
    for k,col in zip(KINDS,COLORS):
        axes[0].plot(AGES,[100*value(k,'gap/'+age)['mean'] for age in AGES],marker='o',color=col,label=LABELS[k])
        axes[1].plot(AGES,[value(k,'gap/'+age)['hurt_pct'] for age in AGES],marker='o',color=col,label=LABELS[k])
    axes[0].set(xlabel='Cycles since confirmed named sighting',ylabel='Mean position error (cm)');axes[1].set(xlabel='Cycles since confirmed named sighting',ylabel='Rows worsened by >1 cm (%)')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);save('age-comparison',fig)
    fig,ax=plt.subplots(figsize=(9,4.6))
    age_regression_rows=[]
    for k,col in zip(KINDS,COLORS):
        values=[value(k,'gap/'+age)['harm_over_25cm_pct'] for age in AGES]
        ax.plot(AGES,values,marker='o',color=col,label=LABELS[k])
        for age,pct in zip(AGES,values):
            group=value(k,'gap/'+age)
            age_regression_rows.append([LABELS[k],age,group['n'],group['hurt_pct'],pct])
    ax.set(title='Position error increased by more than 25 cm versus librcsc',
           xlabel='Cycles since confirmed named sighting',ylabel='Rows worsened by >25 cm (%)',ylim=(0,None))
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    save('age-worsened-over-25cm',fig)
    with (out/'age-regressions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','sighting_age','samples','worsened_over_1cm_pct','worsened_over_25cm_pct'])
        writer.writerows(age_regression_rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));labels=['0-10cm','10-25cm','25-50cm','50-100cm','100cm+']
    for k,col in zip(KINDS,COLORS):
        axes[0].plot(labels,[100*value(k,'baseline_error/'+g)['gain'] for g in labels],marker='o',color=col,label=LABELS[k])
        axes[1].plot(labels,[value(k,'baseline_error/'+g)['hurt_pct'] for g in labels],marker='o',color=col)
    axes[0].axhline(0,c='#8c394b',ls='--');axes[0].set(xlabel='How accurate stock WM already was',ylabel='Mean gain (cm); negative = worse');axes[1].set(xlabel='Stock position error',ylabel='Rows worsened by >1 cm (%)')
    for ax in axes:ax.tick_params(axis='x',rotation=20);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);save('already-accurate',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for policy,off,label in [('raw',-.18,'Use full prediction'),('guarded',.18,'Move at most 25 cm from librcsc (with gates)')]:
        axes[0].bar(x+off,[value(k,policy=policy)['harm_over_25cm_pct'] for k in KINDS],.34,label=label)
        axes[1].bar(x+off,[100*value(k,policy=policy)['p95'] for k in KINDS],.34,label=label)
    axes[0].set(ylabel='Rows made >25 cm worse (%)',xticks=x,xticklabels=['Single','Uncertainty','Mixture']);axes[1].set(ylabel='95th percentile position error (cm)',xticks=x,xticklabels=['Single','Uncertainty','Mixture'])
    axes[0].legend();save('large-regressions',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for k,col in zip(['gaussian','mixture'],COLORS[1:]):
        prob=models[k]['probability'];nominal=[.5,.8,.9,.95]
        axes[0].plot(np.array(nominal)*100,[prob[str(v)]['coverage']*100 for v in nominal],marker='o',label=LABELS[k],color=col)
        axes[1].plot(AGES,[100*prob['by_age'][a]['mean_radius'] for a in AGES],marker='o',label=LABELS[k],color=col)
    axes[0].plot([50,95],[50,95],ls='--',c='#8899a5');axes[0].set(xlabel='Claimed coverage (%)',ylabel='Actual test coverage (%)')
    axes[1].set(xlabel='Sighting age',ylabel='90% ellipse equivalent radius (cm)')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=9);save('confidence-calibration',fig)
    fig,ax=plt.subplots(figsize=(8,4.4))
    for k,col in zip(['gaussian','mixture'],COLORS[1:]):
        p=models[k]['probability']['uncertainty_ranking']
        ax.plot([10,25,50,100],[100*p[str(v)]['mean_error'] for v in [.1,.25,.5,1.]],marker='o',label=LABELS[k],color=col)
    ax.set(xlabel='Keep lowest-uncertainty fraction of test rows (%)',ylabel='Mean position error of retained rows (cm)');ax.grid(alpha=.2);ax.legend();save('confidence-ranking',fig)
    # Full test errors are exact; only this distribution plot is subsampled.
    cohort=np.load(source/'cohort.npz');baseline=cohort['baseline'];rng=np.random.default_rng(50503)
    from evaluate import metrics
    wm_stale=cohort['count']>0;strict_stale=wm_stale&((cohort['flags']&8)==0)
    additional={}
    for k in KINDS:
        pred=np.load(source/(k+'-test.npz'));error=pred['error']
        additional[k]={name:metrics(error[mask],baseline[mask]) for name,mask in [('wm_stale',wm_stale),('strict_unseen_and_wm_stale',strict_stale)]}
    d['additional_cohorts']=additional
    (out/'additional-cohorts.json').write_text(json.dumps(additional,indent=2)+'\n')
    d['omission_study']=build_omission_study(source)
    (out/'group-omission-study.json').write_text(json.dumps(d['omission_study'],indent=2)+'\n')
    sample=np.sort(rng.choice(len(baseline),min(200000,len(baseline)),replace=False))
    fig,ax=plt.subplots(figsize=(9,4.4))
    for k,col in zip(KINDS,COLORS):
        pred=np.load(source/(k+'-test.npz'));gain=baseline[sample]-pred['error'][sample]
        ax.hist(np.clip(gain,-1,1)*100,bins=np.linspace(-100,100,81),density=True,histtype='step',lw=2,label=LABELS[k],color=col)
    ax.axvline(0,c='#8c394b',ls='--');ax.set(xlabel='Position-error gain (cm): left hurts, right helps; tails clipped at ±100 cm',ylabel='Density');ax.legend();save('gain-distribution',fig)
    # Diagnostic cases use one common coordinate frame: displacement from last sighting.
    truth=cohort['truth'];wm=cohort['wm'];age=cohort['age'];mix=np.load(source/'mixture-test.npz');mixerror=mix['error']
    layout_path=source/'observer-layout.json';layout=json.loads(layout_path.read_text()) if layout_path.exists() else []
    ends=np.cumsum([r['rows']*21 for r in layout])
    def identity(idx):
        if not layout:return {}
        flat=int(cohort['indices'][idx]);which=int(np.searchsorted(ends,flat,side='right'));r=layout[which]
        begin=int(ends[which-1]) if which else 0;time_index,slot=divmod(flat-begin,21);p=Path(r['path'])
        return dict(game=r['game'],observer=r['observer'],object_id=int(np.load(p/'ids.npy')[slot]),
                    cycle=int(np.load(p/'cycle.npy')[time_index]),wm_position_count=int(cohort['count'][idx]))
    harm=int(np.argmax(mixerror-baseline));helped=int(np.argmax(baseline-mixerror))
    weights=mix['weight'][sample];means=mix['means'][sample].reshape(-1,3,2)
    mean=(weights[:,:,None]*means).sum(1);spread=(weights*((means-mean[:,None,:])**2).sum(-1)).sum(-1)
    ambiguous=int(sample[np.argmax(spread)])
    fig,axes=plt.subplots(1,3,figsize=(14,4.7));cases=[]
    for ax,idx,title in zip(axes,[helped,harm,ambiguous],['Largest mixture improvement','Largest mixture regression','Separated possible positions']):
        mu=mix['means'][idx].reshape(3,2);scale=mix['scale'][idx].reshape(3,2);rho=mix['rho'][idx];w=mix['weight'][idx]
        ax.scatter(0,0,c='#9aabba',marker='+',s=80,label='Last sighting')
        ax.scatter(*wm[idx],c='#8c394b',marker='x',s=70,label='Stock WM')
        ax.scatter(*truth[idx],c='#202c36',marker='*',s=120,label='Truth (scoring only)')
        for component,col in enumerate(COLORS):
            cov=np.array([[scale[component,0]**2,rho[component]*scale[component].prod()],[rho[component]*scale[component].prod(),scale[component,1]**2]])
            eig,vec=np.linalg.eigh(cov);angle=np.degrees(np.arctan2(vec[1,1],vec[0,1]))
            ax.add_patch(Ellipse(mu[component],2*np.sqrt(eig[1]),2*np.sqrt(eig[0]),angle=angle,color=col,alpha=.15))
            ax.scatter(*mu[component],c=col,s=25+100*w[component],label=f'{100*w[component]:.1f}%')
        ax.set(title=title+'\n'+f'Gap {int(age[idx])}; stock {baseline[idx]:.2f} m / model {mixerror[idx]:.2f} m',xlabel='Δx from last sighting (m)',ylabel='Δy (m)',aspect='equal');ax.autoscale();ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        cases.append(dict(test_row=idx,age=int(age[idx]),baseline_error=float(baseline[idx]),mixture_error=float(mixerror[idx]),weights=w.tolist(),positions=mu.tolist(),**identity(idx)))
    for k in KINDS:
        pred=np.load(source/(k+'-test.npz'));errors=pred['error'];idx=int(np.argmax(errors-baseline))
        cases.append(dict(model=k,diagnostic='largest_regression',test_row=idx,age=int(age[idx]),baseline_error=float(baseline[idx]),
                          model_error=float(errors[idx]),predicted_position=pred['pred'][idx].tolist() if 'pred' in pred.files else [],**identity(idx)))
    save('possible-positions-and-failures',fig);(out/'diagnostic-cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    write_html(out,d,plots)

def write_html(out,d,plots):
    from followup_reporting import confidence_section,capacity_section,selector_section,summary_section
    models=d['models']
    def table(headers,rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    def number(x):return f'{x:,.2f}'
    def comparison_number(value,gain_cm,signed=False):
        # Match the displayed precision so a printed 0.00 is neutral.
        change=round(gain_cm,2)
        style='better' if change>0 else 'worse' if change<0 else 'neutral'
        meaning={'better':'Lower mean error than librcsc (better)',
                 'worse':'Higher mean error than librcsc (worse)',
                 'neutral':'No difference at the displayed precision'}[style]
        shown=number(value)
        if signed:
            shown=number(0.) if style=='neutral' else ('+'+shown if change>0 else shown)
        return f'<span class="comparison-{style}" title="{meaning}">{shown}</span>'
    def score(k,policy='raw',group='all'):return models[k]['groups'][policy][group]
    best=min(KINDS,key=lambda k:score(k)['mean']);baseline=score(best)['baseline_mean']
    rows=[]
    for k in KINDS:
        for policy in ['raw','guarded']:
            v=score(k,policy)
            groups=d['error_change_buckets'][k][policy]
            better=groups['better']['pct'];worse=sum(groups[key]['pct'] for key,_,_ in CHANGE_BUCKETS if key.startswith('worse_'))
            rows.append([LABELS[k],POLICY_LABELS[policy],number(100*v['mean']),number(v['gain_pct']),number(100*v['p95']),number(better),number(worse),number(100*v['max_harm'])])
    body=[f'<p class="eyebrow">EXP5-S1 · POSITION ONLY · {len(d["test_games"])} HELD-OUT GAMES</p>',
          '<h1>Predict unseen players: how much do we help—and how much do we hurt?</h1>',
          '<p class="lead">Three matched recurrent models estimate current position from old observations. Outputs are one position, a position distribution, or three possible position distributions. All scores below use the same paired player cycles.</p>',
          '<div class="cards">'+f'<div><strong>{d["test_samples"]:,}</strong><span>Original comparison: paired test samples</span></div><div><strong>{100*baseline:.2f} cm</strong><span>Original comparison: WM mean error</span></div><div><strong>{100*score(best)["mean"]:.2f} cm</strong><span>Lowest original-model mean: {LABELS[best]}</span></div></div>',
          '<nav><a href="#overall">Overall</a><a href="#age">Gap age</a><a href="#harm">Regressions</a><a href="#confidence">Confidence</a><a href="#examples">Examples</a><a href="#method">Method</a><a href="#capacity">Training/data</a><a href="#selection">WM/model selection</a><a href="#refinement">Refinement + under50m</a></nav>',
          '<div class="note"><b>What “beneficial” means here:</b> smaller current position distance error on the same cycle. Positive gain = stock error − model error. This experiment does not measure action quality, opponent avoidance, possession or match wins. Every valid paired sample has equal weight.</div>',
          '<p><code>e(t) = sqrt((x_pred(t) − x_true(t))² + (y_pred(t) − y_true(t))²)</code>. Both positions refer to the current cycle. Network coordinates are displacement from the last confirmed sighting; this translation leaves the error distance unchanged.</p>',
          summary_section(out,d),
          '<h2 id="overall">1. Mean position error and how often predictions help or hurt</h2>',
          '<h3>What are the three models?</h3>',
          table(['Plot name','Simple meaning'],[
              ['Single','The model gives one predicted player position: one dot.'],
              ['Uncertainty','The model gives one predicted position and a likely area around it. A larger area means more uncertainty.'],
              ['Mixture','The model gives three possible areas. Each has its own center, spread and probability. The three probabilities add up to 100%.']]),
          '<p><b>Mixture example:</b> area A has 60%, area B has 30%, and area C has 10% of the model probability. These are possible position areas, not action labels or accuracy percentages for exact points. Areas may overlap. The values here illustrate the format.</p>',
          ('<p>For the position-error comparison, we use the probability-weighted average of the three centers as one estimated position. We do not choose the center closest to the true player.</p>' if models['mixture']['readout']=='mean' else '<p>For the position-error comparison, we use the center of the most probable component as one estimated position. This rule was chosen on validation data, never using test truth.</p>'),
          plots['overall-benefit-harm'],
          '<div class="note"><b>Orange bars: move at most 25 cm from librcsc.</b> Start at the original librcsc position and move toward the model prediction, by no more than 25 cm. Validation-selected gates may also keep the original position. For example, if the model suggests a position 2 m to the right, this policy moves at most 0.25 m to the right. This limits the size of the correction; it does not intentionally add 25 cm of error, and the final error can still exceed 25 cm.</div>',
          '<p><b>Right chart:</b> full model predictions only. Green means the position error decreased. The warm-colored sections show how much extra error was introduced: up to 1 cm, >1–5 cm, >5–25 cm, >25–50 cm, or >50 cm. Gray means no change. The groups do not overlap and together cover all samples. The limits refer to the <b>change in error</b>, not the final total error.</p>',
          table(['Model','How the prediction is used','Mean cm','Gain %','p95 cm','Better %','Worse %','Max added error cm'],rows),
          '<p>Better/Worse in this section count every change beyond a tiny numerical tolerance of 0.001 cm. The same tolerance is used at interval boundaries. Later tables explicitly labelled >1 cm retain the original 1 cm reporting threshold.</p>']
    bucket_rows=[]
    for kind in KINDS:
        for policy in ['raw','guarded']:
            groups=d['error_change_buckets'][kind][policy]
            bucket_rows.append([LABELS[kind],POLICY_LABELS[policy]]+[number(groups[key]['pct']) for key,_,_ in CHANGE_BUCKETS])
    body += ['<h3>Exactly how much better or worse?</h3>',
             table(['Model','How the prediction is used']+[label+' (%)' for _,label,_ in CHANGE_BUCKETS],bucket_rows),
             '<p>Example: if stock error is 20 cm and model error is 24 cm, that sample belongs to “Worse by >1 to 5 cm.” It does not belong to the larger-increase groups. <a href="error-change-buckets.csv">Download these counts and percentages</a>.</p>']
    rows=[]
    for name,groups in d['baselines'].items():
        v=groups['all'];rows.append([name.replace('_',' '),number(100*v['mean']),number(100*v['p95']),number(v['gain_pct'])])
    body+=['<h3>Does the network beat simple physics?</h3>',table(['Baseline','Mean cm','p95 cm','Gain versus stock %'],rows),
           '<p>Hold uses the last confirmed position. Constant velocity uses the last observed velocity, with a two-sighting displacement fallback when unavailable. Decay uses the server default 0.4. Each baseline uses exactly the same test rows as the networks. Stock means can differ from the original Exp5 report because this comparison requires identified history inside the current PlayOn segment and a 1–30-cycle supported gap.</p>',
           '<h2 id="age">2. How quickly do predictions become unreliable?</h2>',plots['age-comparison'],
           '<h3>How often does the model make position error more than 25 cm worse?</h3>',plots['age-worsened-over-25cm'],
           '<p>This additional chart uses full model predictions on the same samples as the >1 cm chart. A sample is counted when <b>model position error − librcsc position error &gt;25 cm</b>. For example, stock error 10 cm and model error 40 cm means 30 cm extra error, so it is counted. Stock error 30 cm and model error 40 cm means only 10 cm extra error, so it is not counted. Lower percentages are better. <a href="age-regressions.csv">Download the age-by-age percentages for both thresholds</a>.</p>']
    rows=[]
    for age in AGES:
        v=score('point',group='gap/'+age)
        rows.append([age,f'{v["n"]:,}',number(100*v['baseline_mean'])]+[number(100*score(k,group='gap/'+age)['mean']) for k in KINDS])
    body += [table(['Sighting age','Samples','Stock cm','Single cm','Uncertainty cm','Mixture cm'],rows),
             '<p>Age is time since the last confirmed named visual localization. This differs from WM position count, which can reset after an anonymous association. Confirmed fresh positions are preserved by the proposed application rule; this comparison focuses on missing identified observations.</p>',
             '<h2 id="harm">3. When stock WM was already good</h2>',plots['already-accurate'],
             '<p>This is the key regression check: a lower overall mean can hide damage to accurate estimates. The groups use true stock error only for evaluation; this quantity is unavailable to the runtime model or gate.</p>']
    rows=[]
    for k in KINDS:
        for label in ['0-10cm','10-25cm','25-50cm','50-100cm','100cm+']:
            v=score(k,group='baseline_error/'+label)
            rows.append([label,LABELS[k],number(100*v['baseline_mean']),number(100*v['mean']),number(v['hurt_pct']),number(100*v['mean_harm'])])
    body += [table(['Stock error','Model','Stock mean cm','Model mean cm','Worsened >1 cm %','Mean loss when worse >1 cm'],rows),
             '<h3>Severe regressions and a conservative alternative</h3>',plots['large-regressions'],plots['gain-distribution']]
    rows=[]
    for k in KINDS:
        for policy in ['raw','guarded']:
            v=score(k,policy)
            rows.append([LABELS[k],POLICY_LABELS[policy],number(100*v['mean_help']),number(100*v['mean_harm']),number(100*v['p95_harm']),number(v['harm_over_25cm_pct']),number(v['harm_over_50cm_pct']),number(v['harm_over_1m_pct'])])
    body += [table(['Model','How the prediction is used','Mean gain when helped >1 cm','Mean loss when hurt >1 cm','p95 loss among hurt cm','Loss >25 cm %','Loss >50 cm %','Loss >1 m %'],rows),
             '<p>The conservative policy caps the position change from stock at 25 cm, then uses validation-selected age/confidence gates. Rejected predictions retain stock WM. The triangle inequality bounds added position error by the correction length; it does not guarantee a safe action or improved match result. Full gate settings are preserved in metrics.json.</p>',
             '<h2 id="situations">4. Which observer situations benefit?</h2>',
             '<p><b>Reading the colors:</b> <span class="comparison-better">green = better than librcsc</span>; <span class="comparison-worse">red = worse than librcsc</span>; <span class="comparison-neutral">gray = no difference at the displayed precision</span>. Gain = stock error − model error, so a positive gain is an improvement.</p>']
    groups=['strict_unseen','anonymous_current','teammate','opponent','unnamed_wm_refresh','estimated_near_ball_2m','view/60','view/120','view/180','distance/0-10','distance/10-20','distance/20-40','distance/40+']
    rows=[]
    for g in groups:
        v=models['point']['groups']['raw'].get(g)
        if not v:continue
        rows.append([g.replace('_',' '),f'{v["n"]:,}',number(100*v['baseline_mean'])]+[comparison_number(100*score(k,group=g)['gain'],100*score(k,group=g)['gain'],signed=True) for k in KINDS])
    body += [table(['Group','Samples','Stock error cm','Single gain cm','Uncertainty gain cm','Mixture gain cm'],rows),
             '<p>Positive gains help; negative gains hurt. View/distance groups describe the available test situations, rather than a randomized view-policy comparison. “Unnamed WM refresh” means the stored track was refreshed without a named sighting. “Estimated near ball” uses available estimated states, not true possession.</p>',
             '<p><b>Observation-preservation check:</b> the main stress-test cohort includes anonymous current sightings because they do not confirm a named target. These are different from a completely unseen object. Read the strict-unseen and unnamed-refresh rows separately. A model that helps long visual gaps can still damage an accurate anonymous WM update; resolving identity is a separate problem.</p>',
             '<h3 id="exclude-anonymous">What if we leave out these two problematic groups?</h3>']
    study=d['omission_study']
    body += [f'<p>Anonymous current: <b>{study["anonymous_current"]:,}</b> rows. Unnamed WM refresh: <b>{study["unnamed_wm_refresh"]:,}</b> rows. They overlap on <b>{study["overlap"]:,}</b> rows, so removing both excludes <b>{study["excluded_union"]:,} unique rows ({study["excluded_pct"]:.3f}%)</b> and leaves {study["remaining"]:,}. The counts must not be added without subtracting the overlap.</p>',
             '<p>The previously reported “strict unseen and WM stale” subset already excludes both. The following comparison makes that filter explicit. These are unchanged predictions from the original trained models; no model was retrained.</p>']
    omission_rows=[]
    omission_labels={'all_samples':'Use model on all eligible samples',
                     'exclude_both':'Exclude both groups from evaluation',
                     'keep_wm_for_both':'Keep librcsc for both groups (diagnostic)'}
    for kind in KINDS:
        for mode,label in omission_labels.items():
            v=study['models'][kind][mode]
            omission_rows.append([LABELS[kind],label,f'{v["n"]:,}',number(100*v['baseline_mean']),
                                  comparison_number(100*v['mean'],100*v['gain']),number(v['hurt_over_1cm_pct'])])
    body += [table(['Model','Comparison','Samples','Stock mean cm','Model / used-state mean cm','Worsened >1 cm %'],omission_rows),
             '<p>Excluding rows changes the evaluation population and its stock baseline. Keeping librcsc on those rows retains all samples and uses stock errors there. The latter is a post-hoc diagnostic: anonymous observations are labeled with target identity by the logger, so the exact target-level union is not automatically available as a live rule. Removing evaluation rows is also different from excluding training data and retraining. <a href="group-omission-study.json">Exact counts, definitions and results</a>.</p>',
             '<h3>Restricting the comparison to positions that WM has not refreshed</h3>']
    rows=[]
    for name in ['wm_stale','strict_unseen_and_wm_stale']:
        for k in KINDS:
            v=d['additional_cohorts'][k][name]
            name_label={'wm_stale':'Without unnamed WM refresh (position count >0)',
                        'strict_unseen_and_wm_stale':'Without both anonymous current and unnamed WM refresh'}[name]
            rows.append([name_label,LABELS[k],f'{v["n"]:,}',number(100*v['baseline_mean']),comparison_number(100*v['mean'],100*v['gain']),number(v['hurt_pct']),number(100*v['max_harm'])])
    body += [table(['Diagnostic subset','Model','Samples','Stock cm','Model cm','Worsened >1 cm %','Max added error cm'],rows),
             '<p>These additional descriptive subsets exclude every current count-0 position update. They do not alter the primary comparison or retrain the models. The largest single-position regression in the primary cohort occurred on a count-0 anonymous update that stock had localized almost exactly; its saved case identifies the game, observer, target and cycle.</p>',
             '<details><summary>Position-count comparison, separate from identified-sighting age</summary>']
    rows=[]
    for label in ['0','1','2','3','4','5','>5']:
        g='position_count/'+label;v=models['point']['groups']['raw'].get(g)
        if v:rows.append([html.escape(label),f'{v["n"]:,}',number(100*v['baseline_mean'])]+[comparison_number(100*score(k,group=g)['mean'],100*score(k,group=g)['gain']) for k in KINDS])
    body += [table(['WM count','Samples','Stock cm','Single cm','Uncertainty cm','Mixture cm'],rows),'</details>',
             '<h2 id="confidence">5. Can we trust predicted uncertainty?</h2>',plots['confidence-calibration'],plots['confidence-ranking']]
    rows=[]
    for k in ['gaussian','mixture']:
        p=models[k]['probability']
        for level in [.5,.8,.9,.95]:
            v=p[str(level)];rows.append([LABELS[k],f'{level*100:.0f}%',number(v['coverage']*100)+'%',number(100*v['mean_equivalent_radius_m'])])
    body += [table(['Model','Claimed region coverage','Actual test coverage','Mean equivalent radius cm'],rows),
             '<p>Calibration uses ten separate validation games, with age-specific thresholds, before test evaluation. These are empirically calibrated ellipses from total predictive covariance. A mixture ellipse summarizes within-component and between-component uncertainty; it is not the full multimodal density, and individual component weights are not independently calibrated accuracy claims. Region size matters: a huge region can cover the truth without helping decisions.</p>']
    rows=[[LABELS[k],number(models[k]['probability']['nll_mean'])] for k in ['gaussian','mixture']]
    body += [table(['Distribution model','Mean negative log density (lower better)'],rows),
             '<p>NLL evaluates the actual Gaussian or mixture density at the held-out true position in meters. Uncertainty-ranking curves check whether low predicted uncertainty also identifies lower realized error. Neither metric alone proves that confidence gating improves decisions.</p>',
             confidence_section(out),
             '<h2 id="examples">6. Possible positions and concrete failures</h2>',plots['possible-positions-and-failures'],
             '<p>Coordinates are displacement from the last confirmed sighting, aligned with the observer-normalized field axes. Stars are truth labels used only for scoring. Colored centers and percentages are mixture outputs; shaded component ellipses show one standard-deviation axes, not calibrated 90% regions. The two extreme cases intentionally show both the biggest improvement and the biggest regression. The separated-position case is selected from a fixed random diagnostic subset and is an illustration, not a representative accuracy claim.</p>',
             '<h2>7. Consistency across games</h2>']
    rows=[]
    for k in KINDS:
        games=models[k]['per_game'];wins=sum(v['gain']>0 for v in games.values())
        rows.append([LABELS[k],f'{wins}/{len(games)}',number(100*min(v['gain'] for v in games.values())),number(100*max(v['gain'] for v in games.values()))])
    body += [table(['Model','Test games with lower mean error','Worst game gain cm','Best game gain cm'],rows),
             '<p>Main averages remain sample-weighted. This game table checks whether a pooled gain is driven by only a few games; it does not give each game equal weight in the main score.</p>',
             '<h2>What to keep, and what to investigate next</h2>',
             f'<p>In the original three-model comparison, the lowest raw mean error is <b>{LABELS[best]}</b>, at {100*score(best)["mean"]:.2f} cm versus {100*baseline:.2f} cm for stock WM. This identifies a candidate for further validation, not a ready live replacement.</p>',
             '<ul><li>Preserve trustworthy current observations. Investigate anonymous-update reliability separately before changing those tracks.</li><li>Use the already-accurate-stock and near-ball rows to judge practical risk, rather than ranking by the pooled mean alone.</li><li>Compare the capped policy when limiting severe regressions matters. It trades much of the possible correction for a bound on added position error.</li><li>If a distribution model has worse point error but useful confidence, a next experiment can estimate uncertainty around a frozen accurate point model. That was not trained here.</li><li>Repeat promising approaches with additional seeds and inspect failure trajectories before any live integration.</li></ul>',
             '<h2 id="method">8. Method and limits</h2>',
             '<p>'+html.escape(d['cohort'])+'</p>',
             '<ul><li>All 200 source games and 4,400 observers are included in the dataset. Whole-game split: 160 train, ten selection, ten calibration, 20 test.</li><li>Same 51 observation features, uniform-number embedding, 64-unit GRU, 32-cycle history, sampled training endpoints and ten-epoch budget. Point output uses a distance loss; probabilistic outputs use NLL plus a distance term. One seed is tested.</li><li>Features retain old observed velocity/body with masks and ages even when stock WM declares its corresponding field invalid. This experiment predicts position only.</li><li>No fullstate, true actions, true distances or future observations enter features. Stops, setplays and cycle discontinuities reset history.</li><li>Checkpoint and mixture readout are selected before testing. Never choose the component closest to true test position.</li><li>Unknown proximity-matched tracks and rows without valid stock positions are excluded from paired gains. Results do not establish performance for gaps above 30 cycles or different teams/configurations.</li><li>Raw outputs and capped/gated outputs are offline comparisons. Integration consistency, decision timing and match benefit still require a later shadow/live experiment.</li></ul>',
             '<p><a href="metrics.json">All exact aggregates, gates and split</a> · <a href="tables.csv">Comparison tables</a> · <a href="diagnostic-cases.json">Saved diagnostic cases</a> · <a href="../README.md">Training and reproduction</a></p>']
    from refinement_report import refinement_section
    body += [capacity_section(out),selector_section(out),refinement_section(out)]
    style='''body{margin:0;background:#f4f6f6;color:#203447;font:16px/1.6 system-ui}main{max-width:1240px;margin:auto;padding:38px 25px 80px}h1{font-size:39px;line-height:1.2}h2{border-top:1px solid #cad8dc;padding-top:24px;margin-top:50px;line-height:1.3}.eyebrow{color:#087e8b;font-weight:700;letter-spacing:.1em}.lead{font-size:20px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}.cards div{padding:20px;background:white;border-top:4px solid #087e8b}.cards strong{display:block;font-size:28px}.cards span{font-size:14px;color:#566d78}.note{background:#e0efed;padding:20px;border-left:4px solid #087e8b}nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#006d82}.scroll{overflow:auto;background:white}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #e1e8eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e5ecee}svg{width:100%;height:auto;background:white;margin:16px 0}summary{cursor:pointer;color:#006d82}@media(max-width:700px){main{padding:20px 12px}.cards{grid-template-columns:1fr}h1{font-size:30px}}'''
    style+=' .comparison-better{color:#137333;font-weight:700}.comparison-worse{color:#b3261e;font-weight:700}.comparison-neutral{color:#61727d}'
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp5-S1: position prediction benefits and regressions</title><style>'+style+'</style><main>'+'\n'.join(body)+'</main></html>')
    (out/'metrics.json').write_text(json.dumps(d,indent=2)+'\n')
    with (out/'tables.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','policy','group','samples','stock_mean_m','model_mean_m','gain_m','gain_pct','help_over_1cm_pct','hurt_over_1cm_pct','mean_harm_m','p95_error_m','max_added_error_m'])
        for k in KINDS:
            for policy,groups in models[k]['groups'].items():
                for group,v in groups.items():writer.writerow([k,POLICY_LABELS[policy],group,v['n'],v['baseline_mean'],v['mean'],v['gain'],v['gain_pct'],v['help_pct'],v['hurt_pct'],v['mean_harm'],v['p95'],v['max_harm']])
    lines=['# Exp5-S1: three position prediction approaches','','[Open the visual report](report.html).','',f'{d["test_samples"]:,} held-out paired target-cycles in {len(d["test_games"])} games. Every sample has equal weight.','',
           '| Model | Stock cm | Full prediction cm | Gain | Improved >1 cm | Worsened >1 cm | Max added error cm |','|---|---:|---:|---:|---:|---:|---:|']
    for k in KINDS:
        v=score(k);lines.append(f'| {LABELS[k]} | {100*v["baseline_mean"]:.2f} | {100*v["mean"]:.2f} | {v["gain_pct"]:.2f}% | {v["help_pct"]:.2f}% | {v["hurt_pct"]:.2f}% | {100*v["max_harm"]:.2f} |')
    lines += ['', 'The original comparison is followed by completed confidence gates (section5.1), controlled training/capacity/data experiments (section9), and observation-only WM/model selectors (section10).', '', 'These follow-ups reuse previously inspected test games and are exploratory. See the HTML report for full denominators, harm thresholds, validation protocol and limitations.', '', '[Verified selector research](selector-research.md).']
    (out/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evaluation',type=Path);p.add_argument('output',type=Path);args=p.parse_args();render(args.evaluation,args.output)

