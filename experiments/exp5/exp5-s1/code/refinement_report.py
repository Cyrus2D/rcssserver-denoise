"""Concise methods and staged, paired refinement results for the HTML report."""
from pathlib import Path
import json,html,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from followup_reporting import table,fmt,savefig

LABELS={'wm':'Stock WM','point':'Original Single64','gaussian':'Original uncertainty','mixture':'Original mixture','wide128':'GRU128',
 'old_selector_mean':'GRU64 + mean selector','old_selector_risk':'GRU64 + risk selector','continue128':'GRU128 + more training',
 'harm128_l1':'GRU128 + harm penalty1','harm128_l3':'GRU128 + harm penalty3','hard128_l1':'GRU128 + harm penalty1 + WM-good weighting',
 'wmresidual128_l1':'WM-centered residual GRU128 + harm penalty1'}
MODES={'mean':'Mean focused','risk':'Harm penalized','near50_mean':'Mean focused, estimated <50 m','near50_risk':'Harm penalized, estimated <50 m',
       'harm1':'Validation >25 cm-harm budget1%','harm3':'Validation >25 cm-harm budget3%'}
MODES.update(cap25='Correction capped at25cm',cap50='Correction capped at50cm')

def read(out,name):
    p=Path(out)/name;return json.loads(p.read_text()) if p.exists() else None

def method_label(key):
    if key=='count':return 'posCount threshold'
    if key=='count_speed':return 'posCount + speed'
    if key=='count_speed_correction':return 'posCount + speed + disagreement'
    if key.startswith('tree_gain'):return 'Gain tree, depth'+key.removeprefix('tree_gain')
    if key.startswith('tree_cost'):return 'Weighted winner tree, depth'+key.removeprefix('tree_cost')
    return {'strong_gain':'Boosted gain selector','strong_cost':'Boosted weighted winner'}.get(key,key)

def rows_metrics(records):
    rows=[]
    for name,metrics in records:
        for group,label in [('all','All distances'),('under50','True distance <50 m')]:
            m=metrics[group]
            rows.append([name,label,fmt(100*m['mean']),fmt(m['use_pct']),fmt(100*m['p95'])]+[fmt(m['worse'][str(t)]) for t in [0,5,10,25,50,100]]+[fmt(100*m['max_added']),fmt(100*m['max_error'])])
    return rows

HEADERS=['Method','Distance group','Mean error cm','Use model %','p95 cm','Worse >0 cm %','>5 cm %','>10 cm %','>25 cm %','>50 cm %','>100 cm %','Max added error cm','Max total error cm']

def refinement_section(out):
    out=Path(out);distance=read(out,'refinement-distance.json');routing=read(out,'refinement-routing.json');blend=read(out,'refinement-blend.json');final=read(out,'refinement-final.json');fresh=read(out,'refinement-fresh.json');safety=read(out,'refinement-safety.json')
    if not distance and not routing:return ''
    body=['<h2 id="refinement">11. Final refinement: focus on useful distances and harmful replacements</h2>',
          '<p><b>Scope:</b> position only, observation gaps in the same unquantized-vision adaptive-view configuration. No changes are applied to live agents. Each stage below has a separate result. The original 20 test games have already been inspected; these are exploratory improvements, not new independent confirmation.</p>',
          '<p><b>Distance rule:</b> “under 50 m” means the server’s true observer–target distance at the current cycle is strictly below 50 m. This is a reporting label only. Runtime selection and distance-focused tuning use the distance estimated from the unmodified WM and observer position. The two masks are deliberately different.</p>']
    if distance:
        d=distance;rows=[]
        for name,v in d['models'].items():
            a=v['all'];n=v['under50'];rows.append([LABELS[name],fmt(100*a['mean']),fmt(100*n['mean']),fmt(a['worse']['25']),fmt(n['worse']['25']),fmt(100*n['p95'])])
        body += [f'<p>Full original cohort: {d["n"]:,} samples; distance below 50 m: <b>{d["under50_n"]:,}</b>. Missing/invalid true distance: {d["missing_distance"]:,}. All averages remain sample-weighted.</p>',
                 table(['Frozen method','All mean cm','Under 50 m mean cm','All worsened >25 cm %','Under 50 m worsened >25 cm %','Under 50 m p95 cm'],rows)]
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for key in ['wm','point','wide128','old_selector_risk']:
            by=d['models'][key]['age_under50'];x=np.arange(len(by))
            axes[0].plot(x,[100*v['mean'] for v in by.values()],marker='o',label=LABELS[key])
            axes[1].plot(x,[v['worse']['25'] for v in by.values()],marker='o',label=LABELS[key])
        for ax in axes:ax.set(xticks=x,xticklabels=list(by),xlabel='Cycles since confirmed named sighting');ax.grid(alpha=.2)
        axes[0].set_ylabel('Mean position error below 50 m (cm)');axes[1].set_ylabel('Made >25 cm worse than WM below 50 m (%)');axes[0].legend(fontsize=8)
        body.append(savefig(out,'refinement-distance-age',fig))
        body.append('<p><b>Read this first:</b> restricting distance can change the difficulty mix. A higher error below 50 m does not mean the filter caused errors. Compare methods on the same retained rows.</p>')
        confidence=[]
        for v in d['confidence']:
            if v['gate']=='calibrated_probability_within_radius' and v['radius_m']==.5:continue
            label={'always':'Always use','max_component_weight':'Use if largest component weight','calibrated_probability_within_radius':'Use if P(error within 25 cm)'}[v['gate']]
            if v['threshold'] is not None:label+=f' ≥{100*v["threshold"]:.0f}%'
            label+=' / '+v['readout']
            m=v['metrics'];confidence.append([LABELS[v['kind']],label,fmt(m['use_pct']),fmt(100*m['mean']),fmt(m['worse']['0']),fmt(m['worse']['25']),fmt(100*m['max_added'])])
        body.append('<details><summary>Section5.1 confidence gates, rescored only below 50 m</summary>'+table(['Model','Rule','Use model %','Mean cm','Worse >0 cm %','Worse >25 cm %','Max extra cm'],confidence)+'</details>')
    if routing:
        body += ['<h2 id="small-trees">12. Can a small readable decision tree choose well?</h2>',
                 '<p>We fitted depth 2/3/4/5 trees using either expected gain or severity-weighted winner labels. Simple rules use position count, then speed, then disagreement distance. Strong comparators use 180 boosted trees. Tree inputs are 14 named observable features; boosted models add the existing observer/context and uniform-number features. Fit: 250k selection rows. Thresholds/objectives: 500k separate calibration rows. Every selector preserves WM when posCount is zero. Additional-error thresholds allow0.001cm numerical tolerance.</p>',
                 '<p><b>Short method guide:</b> gain tree predicts how many meters the model saves. Weighted winner tree learns which estimate wins, giving larger mistakes more weight. A classifier score is not an ordinary correctness probability. A small tree is easy to inspect; the boosted selector can represent more detailed combinations.</p>']
        r=routing['models']['wide128'];rows=[]
        for k,v in r['candidates'].items():
            for mode in ['mean','risk']:
                p=v['policies'].get(mode)
                if not p:continue
                a=p['metrics']['all'];n=p['metrics']['under50']
                rows.append([method_label(k),MODES[mode],fmt(100*a['mean']),fmt(100*n['mean']),fmt(a['worse']['25']),fmt(n['worse']['25'])])
        body.append(table(['GRU128 selection rule','Objective','All mean cm','Under 50 m mean cm','All worsened >25 cm %','Under 50 m worsened >25 cm %'],rows))
        fig,axes=plt.subplots(1,2,figsize=(12,4.5));families=[['count']]+[['tree_gain'+str(i),'tree_cost'+str(i)] for i in [2,3,4,5]]+[['strong_gain','strong_cost']];labels=['Count','Tree2','Tree3','Tree4','Tree5','Boosted']
        for ax,group in zip(axes,['all','under50']):
            for mode in ['mean','risk']:
                names=[max(f,key=lambda k:r['candidates'][k]['policies'][mode]['tuning']['utility']) for f in families]
                ax.plot(np.arange(len(names)),[100*r['candidates'][k]['policies'][mode]['metrics'][group]['mean'] for k in names],marker='o',label=MODES[mode])
            ax.axhline(100*r['raw'][group]['mean'],ls='--',c='#999',label='Always GRU128')
            ax.set(xticks=np.arange(len(names)),xticklabels=labels,ylabel='Mean error (cm)',title='All distances' if group=='all' else 'True observer distance <50 m');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);body.append(savefig(out,'refinement-small-trees',fig));body.append('<p>At each plotted depth, the gain/weighted-winner family is chosen by calibration utility for that objective. Test results do not choose the plotted tree.</p>')
        text=r['candidates']['tree_gain3']['tree_text'];cut=r['candidates']['tree_gain3']['policies']['mean']['tuning']['threshold']
        body.append(f'<details><summary>Read one actual depth3 gain tree</summary><p>Leaf values are predicted gain in meters. Use the network only when the value exceeds{cut:g}; otherwise keep WM. posCount0 is always protected.</p><pre>{html.escape(text)}</pre></details>')
        body.append('<p><b>What this stage answers:</b> a simple rule can preserve reliable refreshes; a small tree can learn additional conditions. Compare its harm columns too. The highest gain on the calibration games chooses the candidate; test ranking does not choose it.</p>')
        controls=read(out,'refinement-tree-controls.json')
        if controls:
            rows=[]
            for network,mode in [('continue128','mean'),('wmresidual128_l1','near50_risk')]:
                originals=routing['models'][network]['candidates']
                for key,rec,inputs,size in [
                    ('tree_cost3',originals['tree_cost3'],14,str(originals['tree_cost3']['nodes'])+' nodes'),
                    ('tree74_cost3',controls['models'][network]['tree74_cost3'],74,str(controls['models'][network]['tree74_cost3']['nodes'])+' nodes'),
                    ('tree_cost5',originals['tree_cost5'],14,str(originals['tree_cost5']['nodes'])+' nodes'),
                    ('tree74_cost5',controls['models'][network]['tree74_cost5'],74,str(controls['models'][network]['tree74_cost5']['nodes'])+' nodes'),
                    ('boost14_cost',controls['models'][network]['boost14_cost'],14,'180 trees'),
                    ('strong_cost',originals['strong_cost'],74,'180 trees')]:
                    m=rec['policies'][mode]['metrics']['under50']
                    rows.append([LABELS[network],key,inputs,size,MODES[mode],fmt(100*m['mean']),fmt(m['worse']['25'])])
            body += ['<h3>Same-input controls on the retrained networks</h3>',
                '<p>The original small trees use14 simple features; the boosted model uses74. This control separates extra context from selector complexity: compare depth3/5 weighted trees and boosted selection with the same14 or74 inputs. Fit and threshold tuning still use only original validation games; scores below use the original common1M test subset. This additional exploratory control was appended after primary new-game confirmation and does not retune the frozen primary policies.</p>',
                table(['Position network','Selector','Inputs','Size','Objective','Under50m mean cm','Under50m worsened >25cm %'],rows),
                '<p>A small tree can be a useful implementation tradeoff even when it does not win the chosen risk objective. Compare both columns; a lower mean can coexist with more harmful replacements.</p>']
        body += ['<h2 id="harm-training">13. Train the model to avoid harmful replacements</h2>',
                 '<p>Controlled fine-tuning starts from the same GRU128 checkpoint and uses identical sampled endpoints. The ordinary continuation is the control. Harm penalty adds λ×max(model error−WM error,0) to ordinary position error. A fourth variant additionally gives 4× training weight to rows where WM error is below 25 cm. Truth is used only for training labels/weights; runtime inputs do not change. Fresh seeds repeat ordinary GRU128 training separately.</p>',
                 '<p><b>Additional residual candidate:</b> reuse the GRU128 encoder, zero the new readout and learn corrections around current WM instead of around constant-velocity extrapolation. It therefore starts by preserving WM exactly. It uses the same inputs, an additional 2.048M sampled endpoints and harm penalty1. This exploratory amendment is recorded before any new-game performance is evaluated.</p>']
        records=[(LABELS.get(k,k),v['raw']) for k,v in routing['models'].items()]
        body.append(table(HEADERS,rows_metrics(records)))
        body.append('<p>Checkpoint selection uses only selection games and the corresponding training objective on naturally weighted validation rows. Fine-tuning restarts AdamW at lower learning rate. It adds another 2.048M sampled endpoints; fresh seed repeats use the original2.048M budget. A repeated seed is not a fresh test-game set.</p>')
        body += ['<h3>Best routed candidate for each validation objective — common one-million-row test subset</h3>',table(HEADERS,rows_metrics([(MODES[k]+' / '+v['model']+' / '+v['candidate'],v['metrics']) for k,v in routing['winners'].items()]))]
    if blend:
        body += ['<h2 id="bounded-corrections">14. Blend or cap the correction instead of replacing it fully</h2>',
                 '<p>Blending moves partway from WM toward the network (25/50/75/100%). Capping then limits that movement to 25/50/100/200/400 cm or leaves it uncapped. Adaptive caps instead grow by 10/25/50/100 cm per cycle since the last named sighting, up to 1/2/4 m; these are tested with 50%/100% blending. We compare unconditional use and frozen learned selector scores. All choices are tuned on calibration games; rejected rows retain WM. A cap bounds added distance error from this correction, not final total error.</p>',
                 table(HEADERS,rows_metrics([(MODES[k]+f' / {v["model"]} / α={v["alpha"]:g} / cap='+('none' if v['cap_m']>1000 else f'{100*v["cap_m"]:g}cm')+(f' / growth={100*v["cap_per_cycle"]:g}cm/cycle' if v.get('cap_per_cycle') is not None else ''),v['metrics']) for k,v in blend['winners'].items()]))]
    if final:
        body += ['<h2 id="refinement-final">15. Final frozen policies on the full original test cohort</h2>',
                 '<p>Policies below were selected using calibration utility before full-test inference. The same 28,180,368 original paired rows are retained, with a separate true-distance<50 m comparison. “Risk” penalizes harm magnitude; “harm budget” is an empirical calibration target, not a guarantee on unseen games.</p>',
                 table(HEADERS,rows_metrics([(MODES.get(k,k),v['metrics']) for k,v in final['policies'].items()]))]
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,group in zip(axes,['all','under50']):
            for k in ['mean','near50_risk','harm1','harm3','cap25','cap50']:
                v=final['policies'][k]
                m=v['metrics'][group];ax.scatter(m['worse']['25'],100*m['mean'],s=65);ax.annotate({'mean':'Mean','near50_risk':'Practical','harm1':'Harm1%','harm3':'Harm3%','cap25':'Cap25cm','cap50':'Cap50cm'}[k],(m['worse']['25'],100*m['mean']),xytext=(4,5),textcoords='offset points',fontsize=8)
            ax.set(xlabel='Samples made >25 cm worse than WM (%)',ylabel='Mean error (cm)',title='All distances' if group=='all' else 'True distance <50 m');ax.grid(alpha=.2);ax.margins(x=.13,y=.08)
        body.append(savefig(out,'refinement-final-tradeoff',fig))
        body.append('<p><b>Where does the selected policy still fail?</b> These descriptive subgroups do not choose the policy. True-error groupings are evaluation labels, never runtime gates.</p>')
        rows=[]
        for mode,v in final['policies'].items():
            for distance_key,label in [('subgroups','All distances'),('under50_subgroups','Under 50 m')]:
                for group,m in v.get(distance_key,{}).items():rows.append([MODES.get(mode,mode),label,group,f'{m["n"]:,}',fmt(100*m['stock']),fmt(100*m['mean']),fmt(m['worse']['25']),fmt(100*m['max_added'])])
        body.append('<details><summary>Age, position count, speed, ball proximity and already-good WM — all/under 50 m</summary>'+table(['Policy','Distance group','Situation','Samples','WM cm','Policy cm','Worse >25 cm %','Max extra cm'],rows)+'</details>')
        ages=['named age1–2','named age3–5','named age6–10','named age>10']
        if all(k in next(iter(final['policies'].values())).get('under50_subgroups',{}) for k in ages):
            fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(len(ages))
            for mode in ['mean','risk','harm1','cap50']:
                v=final['policies'][mode]['under50_subgroups']
                axes[0].plot(x,[100*v[k]['mean'] for k in ages],marker='o',label=MODES[mode])
                axes[1].plot(x,[v[k]['worse']['25'] for k in ages],marker='o',label=MODES[mode])
            for ax in axes:ax.set(xticks=x,xticklabels=['1–2','3–5','6–10','>10'],xlabel='Cycles since confirmed named sighting');ax.grid(alpha=.2)
            axes[0].set_ylabel('Mean position error below 50 m (cm)');axes[1].set_ylabel('Made >25 cm worse below 50 m (%)');axes[0].legend(fontsize=8)
            body.append(savefig(out,'refinement-final-age-under50',fig))
        if final.get('stability'):
            body.append('<h3>Seed stability and paired game-level resampling</h3>')
            body.append('<p>Three ordinary GRU128 training seeds are compared on the same test subset. Uncertainty intervals for policy differences resample whole games, retaining natural sample weights within each bootstrap draw; they do not treat millions of correlated cycles as independent games.</p>')
            rows=[[k,fmt(100*v['all']['mean']),fmt(100*v['under50']['mean']),fmt(v['all']['worse']['25'])] for k,v in final['stability']['seeds'].items()]
            body.append(table(['GRU128 seed','All mean cm','Under 50 m mean cm','Worse >25 cm %'],rows))
            rows=[[k,fmt(100*v['gain_mean']),fmt(100*v['ci95'][0]),fmt(100*v['ci95'][1])] for k,v in final['stability']['bootstrap'].items()]
            body.append(table(['Policy improvement versus raw GRU128','Mean gain cm','95% game-bootstrap lower cm','Upper cm'],rows))
        body.append('<p>Extra seeds and game resampling check stability on the existing data. A separate new-game confirmation is reported below when complete. Fullstate never enters runtime features. No live deployment is included.</p>')
        with (out/'refinement-final.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(HEADERS);w.writerows(rows_metrics([(MODES.get(k,k),v['metrics']) for k,v in final['policies'].items()]))
    else:
        body.append('<p><b>Status:</b> completed stages above are saved. Remaining training, blending and full-cohort evaluation are still running; no final policy is claimed yet.</p>')
    if fresh:
        body += ['<h2 id="fresh-confirmation">16. Independent new-game confirmation after freezing policies</h2>',
                 f'<p>We generated{fresh["games"]} new full-length baseline games with new fixed seeds and the same adaptive-view/no-quantization/no-hearing configuration. Both teams run the unmodified baseline; predictions are evaluated offline. The{fresh["samples"]:,} eligible samples were never used to train models, choose checkpoints, select policies or tune thresholds. Policies were frozen before reading prediction performance.</p>']
        records=[('Stock WM',fresh['wm']),('Original GRU64',fresh['raw_models']['original64']),('Original GRU128',fresh['raw128'])]
        records += [(MODES.get(k,k),v['metrics']) for k,v in fresh['policies'].items()]
        body.append(table(HEADERS,rows_metrics(records)))
        fig,axes=plt.subplots(1,2,figsize=(13,5.5));examples=[('Original64',fresh['raw_models']['original64']),('Original128',fresh['raw128']),('Mean policy',fresh['policies']['mean']['metrics']),('Practical policy',fresh['policies']['near50_risk']['metrics']),('Harm target1%',fresh['policies']['harm1']['metrics'])]
        colors=['#087e8b','#cbd5da','#ffe1a3','#edaf5d','#d77643','#a73538']
        legends=['Better','Unchanged','Worse by≤5cm','Worse by>5–25cm','Worse by>25–50cm','Worse by>50cm']
        for ax,group in zip(axes,['all','under50']):
            columns=[]
            for _,v in examples:
                m=v[group];w=m['worse'];better=m['better_pct'];columns.append([better,max(0,100-better-w['0']),w['0']-w['5'],w['5']-w['25'],w['25']-w['50'],w['50']])
            columns=np.asarray(columns);bottom=np.zeros(len(examples))
            for j,(color,label) in enumerate(zip(colors,legends)):
                ax.bar(np.arange(len(examples)),columns[:,j],bottom=bottom,color=color,label=label);bottom+=columns[:,j]
            ax.set(xticks=np.arange(len(examples)),xticklabels=['Original64','Original128','Mean','Practical','Harm1%'],ylim=(0,100),ylabel='Share of paired samples (%)',title='New games: all distances' if group=='all' else 'New games: true distance<50 m')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,fontsize=9);fig.subplots_adjust(bottom=.2)
        body.append(savefig(out,'refinement-fresh-change-bands',fig,rect=(0,.08,1,1)))
        distance_keys=['true distance<10m','true distance10–25m','true distance25–50m','true distance>=50m']
        if all(k in fresh['policies']['near50_risk']['subgroups'] for k in distance_keys):
            fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(4);reference=fresh['policies']['near50_risk']['subgroups']
            for field,label in [('stock','Stock WM'),('raw128_mean','Original GRU128')]:
                axes[0].plot(x,[100*reference[k][field] for k in distance_keys],marker='o',label=label)
            for mode,label in [('mean','Mean focused'),('near50_risk','Practical'),('harm1','Harm target1%')]:
                sub=fresh['policies'][mode]['subgroups']
                axes[0].plot(x,[100*sub[k]['mean'] for k in distance_keys],marker='o',label=label)
                axes[1].plot(x,[sub[k]['worse']['25'] for k in distance_keys],marker='o',label=label)
            for ax in axes:ax.set(xticks=x,xticklabels=['<10','10–25','25–50','≥50'],xlabel='True observer–target distance (m)');ax.grid(alpha=.2)
            axes[0].set_ylabel('Mean position error on new games (cm)');axes[1].set_ylabel('Made >25cm worse than WM (%)');axes[0].legend(fontsize=8)
            body.append(savefig(out,'refinement-fresh-distance',fig))
        body.append('<p>Green shows every improvement; gray retains the same error, usually because WM is kept. Warm colors show added error relative to WM. The chart retains all samples, including rejected predictions.</p>')
        body.append('<p>These are independent trajectories from the same two baseline teams and configuration. They test transfer to new games, not new opponents, different visibility settings, a corrected-agent feedback loop or match-win benefit. We report every frozen policy; this batch is not used to retune or pick another threshold.</p>')
        rows=[[k,fmt(100*v['gain_mean']),fmt(100*v['ci95'][0]),fmt(100*v['ci95'][1])] for k,v in fresh['stability']['bootstrap'].items()]
        body.append(table(['Gain versus raw GRU128 on new games','Mean gain cm','95% paired-game bootstrap lower cm','Upper cm'],rows))
    if safety:
        body += ['<h3 id="practical-safety">A ceiling on the frozen practical policy: how much accuracy does safety cost?</h3>',
                 '<p>Keep exactly the same practical selector and acceptance decisions. Add a fixed 1/2/4 m ceiling to its proposed movement from WM, without any retuning. These sensitivity checks were declared before new-game scoring. They expose the tradeoff between average accuracy and a mathematical bound on added position error.</p>']
        records=[]
        for cohort,v in safety.items():
            for k,m in v['policies'].items():records.append([('New 10 games' if cohort=='fresh' else 'Original 20 games')+' / '+k,m])
        body.append(table(HEADERS,rows_metrics(records)))
        fig,axes=plt.subplots(1,2,figsize=(12,4.5));labels=['No cap','1m cap','2m cap','4m cap'];keys=['uncapped_practical','practical_cap100','practical_cap200','practical_cap400']
        for ax,group in zip(axes,['all','under50']):
            for cohort,v in safety.items():ax.plot(np.arange(4),[100*v['policies'][k][group]['mean'] for k in keys],marker='o',label='New 10 games' if cohort=='fresh' else 'Original 20 games')
            ax.set(xticks=np.arange(4),xticklabels=labels,ylabel='Mean error (cm)',title='All distances' if group=='all' else 'True distance <50 m');ax.grid(alpha=.2);ax.margins(x=.13,y=.08);ax.legend(fontsize=8)
        body.append(savefig(out,'refinement-practical-safety',fig))
    stages=[('Distance<50 m rescoring','Complete' if distance else 'Pending'),
            ('GRU128 + readable trees / simple rules','Complete' if routing else 'Pending'),
            ('Controlled retraining, residual candidate and seed repeats',f'{len(routing["models"])} models scored' if routing else 'Pending'),
            ('Fractional, fixed-cap and age-dependent-cap policies','Complete' if blend else 'Pending'),
            ('Full original test-cohort evaluation','Complete' if final else 'Running / pending'),
            ('New-game confirmation with frozen policies','Complete' if fresh else 'Running / pending')]
    lines=['# Exp5-S1 refinement: progress and findings','','All means use natural per-target-cycle weights. True distance<50 m is an evaluation label; runtime distance uses unmodified WM geometry.','','| Stage | Status |','|---|---|']+[f'| {name} | {status} |' for name,status in stages]
    if final:
        lines += ['', '## Full original-cohort results','','| Policy | All mean cm | Below50m mean cm | All >25 cm added error % | Below50m >25 cm added error % |','|---|---:|---:|---:|---:|']
        for mode,v in final['policies'].items():
            a=v['metrics']['all'];n=v['metrics']['under50'];lines.append(f'| {MODES.get(mode,mode)} | {100*a["mean"]:.2f} | {100*n["mean"]:.2f} | {a["worse"]["25"]:.2f} | {n["worse"]["25"]:.2f} |')
    if fresh:lines += ['', f'Independent confirmation complete: {fresh["games"]} new games, {fresh["samples"]:,} samples. No new-game result was used to fit, choose thresholds or select another policy.']
    lines += ['', 'The practical objective designated before new-game scoring is near50_risk. Alternatives explicitly trade mean error, harm frequency and a hard cap on correction distance. Fresh games use the same baseline teams, not corrected-agent feedback.', '', '[Open the visual report](report.html#refinement).']
    (out/'refinement-status.md').write_text('\n'.join(lines)+'\n')
    body.append('<p>Exact staged results: <a href="refinement-status.md">progress and findings</a>. Large predictions, models and TensorBoard logs remain under logs/exp5-s1/refinement.</p>')
    from refinement_tree_report import tree_followup
    body.append(tree_followup(out))
    return '\n'.join(body)
