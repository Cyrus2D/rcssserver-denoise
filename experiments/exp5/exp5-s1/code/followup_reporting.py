"""Small, reproducible report extensions for the staged Exp5-S1 studies."""
from pathlib import Path
import json,csv,re,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
def fmt(v,digits=2):return '—' if v is None else f'{v:,.{digits}f}'

def summary_section(out,d):
    capacity=load(out,'followup-capacity.json');selectors=load(out,'followup-selectors.json')
    if not capacity or not selectors:return ''
    latest=load(out,'refinement-fresh.json') or load(out,'refinement-final.json')
    if latest:
        label='10 new confirmation games' if latest['cohort']=='fresh' else 'full original20 test games (exploratory reuse)'
        rows=[]
        choices=[('Stock WM',latest['wm']),('Original GRU128',latest['raw128'])]+[(label,latest['policies'][k]['metrics']) for k,label in [('mean','Mean focused'),('near50_risk','Practical: penalize harm'),('harm1','Stricter harm target1%'),('cap50','Hard50cm correction cap')]]
        for name,v in choices:
            a=v['all'];n=v['under50'];rows.append([name,fmt(100*a['mean']),fmt(100*n['mean']),fmt(n['worse']['0']),fmt(n['worse']['25']),fmt(100*n['max_added'])])
        return '<div class="note"><b>Latest refinement: '+label+'</b>'+table(['Policy','All mean cm','Under50m mean cm','Under50m worse >0cm %','Under50m worse >25cm %','Under50m max extra cm'],rows)+'<p>mean: prioritize average accuracy. near50_risk: practical objective designated before new-game scoring, with a larger penalty for harm. harm1: target ≤1% of calibration samples made >25cm worse; this is not a test guarantee. cap50: move at most50cm. <a href="#refinement">Methods and staged results</a> · <a href="#practical-safety">Additional1/2/4m safety ceilings</a>. These are offline position results.</p></div>'
    original=d['models']['point']['groups']['raw']['all']
    winner=capacity['full_test_selection_winner'];w=capacity['models'][winner]['full_test']
    buckets=d['error_change_buckets']['point']['raw']
    worse=sum(v['pct'] for k,v in buckets.items() if k.startswith('worse_'))
    over25=sum(v['pct'] for k,v in buckets.items() if k.startswith('worse_') and ('25_50' in k or 'over_50' in k))
    rows=[['Stock librcsc',fmt(100*original['baseline_mean']),'0.00','0.00'],
          ['Original single-position GRU64',fmt(100*original['mean']),fmt(worse),fmt(over25)],
          ['Wider GRU128 (separate experiment)',fmt(100*w['mean_m']),fmt(w['worse_over_0cm_pct']),fmt(w['worse_over_25cm_pct'])]]
    for policy,label in [('mean','Original GRU64 + mean-focused selector'),('risk','Original GRU64 + risk-focused selector')]:
        m=selectors['full_test'][policy]['metrics'];rows.append([label,fmt(100*m['mean']),fmt(m['hurt_pct']['0']),fmt(m['hurt_pct']['0.25'])])
    return '<div class="note"><b>Follow-up results at a glance — same full 28,180,368-row cohort.</b>'+table(['Method','Mean error cm','Worse than WM >0 cm %','Worse than WM >25 cm %'],rows)+'<p>The wider network and the selectors were tested separately. Their gains have not been combined. New comparisons reuse previously inspected games and remain exploratory. Details: <a href="#confidence-gating">confidence gates</a>, <a href="#capacity">network/data study</a>, <a href="#selection">WM/model selection</a>.</p></div>'
def load(out,name):
    p=Path(out)/name
    return json.loads(p.read_text()) if p.exists() else None
def embed(path,prefix):
    raw=Path(path).read_text();raw=raw[raw.index('<svg'):]
    for ident in set(re.findall(r'\bid="([^"]+)"',raw)):
        raw=raw.replace('id="'+ident+'"','id="'+prefix+'-'+ident+'"').replace('#'+ident+'"','#'+prefix+'-'+ident+'"').replace('#'+ident+')','#'+prefix+'-'+ident+')')
    return raw
def savefig(out,name,fig,rect=None):
    fig.tight_layout(pad=2,rect=rect);fig.savefig(Path(out)/(name+'.svg'),bbox_inches='tight');fig.savefig(Path(out)/(name+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)
    return embed(Path(out)/(name+'.svg'),name)

def confidence_section(out):
    d=load(out,'followup-confidence.json')
    if not d:return '<h3 id="confidence-gating">5.1 Does confidence help choose model or WM?</h3><p>Requested fixed-radius probability and mixture-dominance experiments are in progress.</p>'
    rows=d['policies'];labels={'gaussian':'Gaussian','mixture':'Mixture'}
    def policy(r):
        if r['gate']=='always':return 'Always use '+r['readout']
        if r['gate']=='max_component_weight':return 'Largest-area weight ≥'+fmt(100*r['threshold'],0)+'%; '+r['readout']
        return 'P(error ≤'+fmt(100*r['radius_m'],0)+' cm) ≥'+fmt(100*r['threshold'],0)+'%; '+r['readout']
    main=[r for r in rows if r['gate']=='always' or r['gate']=='max_component_weight' or r['radius_m']==.25]
    headers=['Model','Rule','Use model %','Mean error cm','Worse >0 cm %','>5 cm %','>10 cm %','>25 cm %','>50 cm %','>100 cm %','Max extra error cm','Max final error cm']
    def tr(r):return [labels[r['kind']],policy(r),fmt(r['use_model_pct']),fmt(100*r['mean'])]+[fmt(r['worsened_pct'][str(t)]) for t in [0.,.05,.1,.25,.5,1.]]+[fmt(100*r['max_added_error']),fmt(100*r['max_error'])]
    body=['<h3 id="confidence-gating">5.1 Does using only high-confidence predictions help?</h3>',
          '<div class="note"><b>80% must describe a specific event.</b> The earlier 80% coverage row is a chosen region level, shared by all samples. Here confidence means the calibrated probability that the actual position is within a fixed distance of the point we would use: <b>P(position error ≤25 cm)</b>, with a 50 cm sensitivity check. Thresholds are 50/80/90/95%. When rejected, use the original WM position.</div>',
          '<p>Gaussian probabilities integrate its full 2D density inside the circle. Mixture probability gates integrate all three components around the highest-weight component center. Isotonic probability calibration uses only the ten calibration games. Component-weight gates are tested separately: an 80% component weight measures dominance, not 80% positional accuracy. The original weighted-mean and highest-weight-center readouts are both shown for those gates.</p>',
          '<p><b>All percentages and means retain the full 28,180,368-row denominator.</b> Rejected rows keep WM and have zero added error. “Worse” measures additional position error relative to WM; “Max final error” is total error. Numerical tolerance is 0.001 cm.</p>',table(headers,[tr(r) for r in main])]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    families=[('gaussian','calibrated_probability_within_radius',.25,'Gaussian: P(error≤25cm)'),
              ('mixture','calibrated_probability_within_radius',.25,'Mixture: P(error≤25cm)'),
              ('mixture','max_component_weight',None,'Mixture: largest-area weight')]
    for kind,gate,radius,label in families:
        r=[v for v in rows if v['kind']==kind and v['gate']==gate and v['radius_m']==radius and (gate!='max_component_weight' or v['readout']=='highest-weight center')]
        r=sorted(r,key=lambda v:v['threshold'])
        axes[0].plot([100*v['threshold'] for v in r],[100*v['mean'] for v in r],marker='o',label=label)
        axes[1].plot([100*v['threshold'] for v in r],[v['worsened_pct']['0.25'] for v in r],marker='o',label=label)
    axes[0].axhline(100*d['baseline']['mean'],ls='--',c='#8c394b',label='Always WM')
    axes[0].set(ylabel='Full-cohort mean position error (cm)')
    axes[1].set(ylabel='Samples made >25 cm worse than WM (%)')
    for ax in axes:ax.grid(alpha=.2);ax.set(xlabel='Confidence threshold (%) — otherwise keep WM',xticks=[50,80,90,95])
    axes[0].legend(fontsize=8);body.append(savefig(out,'followup-confidence-tradeoff',fig))
    body+=['<h4>Does the selected group actually meet the requested accuracy?</h4>',
           table(['Model','Rule','Selected samples','Selected error within requested radius %','Selected mean error cm'],
                 [[labels[r['kind']],policy(r),f'{r["accepted"]:,}',fmt(r['selected_within_radius_pct']),fmt(100*r['selected_mean_error'] if r['selected_mean_error'] is not None else None)] for r in rows if r['gate']=='calibrated_probability_within_radius' and r['radius_m']==.25]),
           '<details><summary>Sensitivity: confidence about error within 50 cm</summary>'+table(headers,[tr(r) for r in rows if r['gate']=='calibrated_probability_within_radius' and r['radius_m']==.5])+'</details>',
           '<p><b>Interpretation:</b> high absolute accuracy probability does not imply that the model is better than WM on that sample. A gate can reduce harmful replacements but leave large stock errors unchanged, increasing the pooled mean compared with always using the network. Both objectives must be reported.</p>',
           '<p>This is exploratory reuse of the old test games. No runtime code was changed. <a href="followup-confidence.json">All exact policies and results</a>.</p>']
    with (Path(out)/'followup-confidence.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(headers)
        writer.writerows(tr(r) for r in rows)
    return '\n'.join(body)

def selector_section(out):
    d=load(out,'followup-selectors.json');multi=load(out,'followup-multi-expert.json')
    if not d:return '<h2 id="selection">10. Learn when to use a model or keep WM</h2><p>Selector experiments are in progress.</p>'
    labels={'speed_rule':'Simple speed rule','gain':'Expected gain regression','errors':'Separate error regressions','win_classifier':'Win classifier','cost_classifier':'Severity-weighted classifier','gain_quantile':'Lower gain quantile','cluster':'128 local clusters','knn':'32 nearest neighbors'}
    rows=[]
    for key,c in d['candidates'].items():
        for policy,p in c['policies'].items():
            m=p['metrics'];rows.append([labels.get(key,key),policy,fmt(m['accept_pct']),fmt(100*m['mean']),fmt(m['hurt_pct']['0']),fmt(m['hurt_pct']['0.25']),fmt(100*m['max_added_error'])])
    body=['<h2 id="selection">10. Learn when to use a model or keep WM</h2>',
          '<p>Eight selector families were tested, including clustering and nearest neighbors. Each learns or estimates <b>advantage over WM</b>, rather than merely predicting absolute network accuracy. The three original networks are frozen. Fit uses 250,000 rows from ten selection games; thresholds and candidate selection use 500,000 rows from ten separate calibration games. All candidates below are compared on the same one-million-row old-test subset.</p>',
          '<p>Inputs are observer features, position age/count, uniform-number category, proposed correction and observable motion/context. True position, true error, future observations and truth-based anonymous-target matching are never selector inputs. Every learned policy preserves WM when its position count is zero.</p>',
          table(['Selector','Objective','Use model %','Mean error cm','Worse >0 cm %','Worse >25 cm %','Max extra error cm'],rows)]
    b=d['baselines'];body.append(table(['Common-subset reference','Mean error cm','Worse >0 cm %'],[[label,fmt(100*b[key]['mean']),fmt(b[key]['hurt_pct']['0'])] for key,label in [('wm','Always WM'),('always_point','Always original single position'),('protect_fresh_only','Single position, preserve WM count0'),('oracle_not_deployable','Oracle: choose smaller true error (unavailable live)')]]))
    body+=['<p><b>Two different objectives:</b> “mean” maximizes predicted error reduction. “risk” additionally penalizes the magnitude of harmful replacements: utility = gain − 3×max(−gain,0), where gain = WM error − model error. This risk objective is a preference, not a bound or guarantee. The severity-weighted classifier wins both validation comparisons. Its score is <b>not a calibrated probability</b> that the model wins.</p>',
           '<h3>Validation-selected policies on all 28,180,368 paired test samples</h3>']
    full=[]
    for p,v in d['full_test'].items():full.append(('Single-position selector: '+p,v['metrics']))
    if multi:
        for p,v in multi['full_test'].items():full.append(('Three-network selector: '+p,v['metrics']))
    headers=['Policy','Use model %','Mean error cm','p95 cm','Worse >0 cm %','>5 cm %','>10 cm %','>25 cm %','>50 cm %','>100 cm %','Max extra error cm','Max final error cm']
    resultrows=[[name,fmt(m['accept_pct']),fmt(100*m['mean']),fmt(100*m['p95'])]+[fmt(m['hurt_pct'][str(t)]) for t in [0,.05,.1,.25,.5,1.]]+[fmt(100*m['max_added_error']),fmt(100*m['max_error'])] for name,m in full]
    body.append(table(headers,resultrows))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));short=['Point\nmean','Point\nrisk','3 networks\nmean','3 networks\nrisk'];x=np.arange(len(full))
    axes[0].bar(x,[100*m['mean'] for _,m in full],color=['#387cb1','#008b8b']*2);axes[0].set(xticks=x,xticklabels=short,ylabel='Full-cohort mean error (cm)')
    width=.35
    axes[1].bar(x-width/2,[m['hurt_pct']['0'] for _,m in full],width,label='Any added error')
    axes[1].bar(x+width/2,[m['hurt_pct']['0.25'] for _,m in full],width,label='Added error >25cm')
    axes[1].set(xticks=x,xticklabels=short,ylabel='Samples made worse than WM (%)');axes[1].legend(fontsize=9)
    body.append(savefig(out,'followup-selector-comparison',fig))
    if multi:body.append(table(['Three-network policy','Keep WM %','Use single %','Use Gaussian %','Use mixture %'],[[p]+[fmt(v['choice_pct'][k]) for k in ['wm','point','gaussian','mixture']] for p,v in multi['full_test'].items()]))
    body+=['<p><b>What we learned:</b> predicting relative advantage is more useful than treating mixture dominance as correctness. The mean-focused single-position selector slightly improves the original network mean and substantially reduces how often it worsens WM. The risk-focused selector reduces harmful replacements further, at some cost in mean accuracy. The three-network selector offers another tradeoff; it does not dominate the simpler selector on all metrics. Rare large regressions remain, and fallback can retain a large WM error.</p>',
           '<p><b>Scope of this original selector study:</b> these selectors use the original 64-unit networks. The later <a href="#refinement">refinement study in sections11–16</a> tests GRU128 routing, harm-aware retraining, bounded corrections and new-game confirmation. Keep these stages separate when comparing gains.</p>',
           '<h3>Research and targeted training</h3>',
           '<p>The literature supports selective prediction, learning to defer and estimating local competence. We tested gradient-boosted advantage estimators, clustering and nearest neighbors here. <a href="selector-research.md">Verified research notes and eight ranked approaches</a> include <a href="https://proceedings.mlr.press/v97/geifman19a.html">SelectiveNet</a>, <a href="https://proceedings.mlr.press/v119/mozannar20b.html">Learning to Defer</a>, and <a href="https://arxiv.org/abs/1802.04967">DESlib</a>. These references motivate methods; they do not certify this soccer experiment.</p>',
           '<p>For targeted retraining, label harmful cases on training data, study observable groups and oversample or reweight them using validation feedback. Keep natural per-sample weighting for evaluation. A comparison using additional seeds, hard-case weighting and a fresh game set has not yet been run; we do not claim those improvements. Correlated player cycles also mean that row-level confidence or conformal terminology cannot automatically establish game-level guarantees.</p>',
           '<p><b>Evaluation limit:</b> the old 20 test games were previously inspected. These new comparisons are exploratory reuse, even though fit and threshold tuning use separate validation games. No live changes or match-win claims follow from these position metrics.</p>',
           '<p><a href="followup-selectors.json">All selector candidates and exact results</a> · <a href="followup-multi-expert.json">Multi-expert results</a>. Serialized offline estimators remain under logs/exp5-s1/selector-study.</p>']
    with (Path(out)/'followup-selectors.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(resultrows)
    return '\n'.join(body)

def capacity_section(out):
    d=load(out,'followup-capacity.json')
    if not d:return '<h2 id="capacity">9. Can more training, a larger network or more data help?</h2><p>Controlled capacity/data experiments are in progress.</p>'
    models=d['models'];names=list(models)
    labels={'baseline64':'Original64×1,10 epochs','continued64':'64×1,20 epochs','wide128':'128×1,10 epochs','deep64x2':'64×2,10 epochs','data40':'64×1,40 training games'}
    rows=[]
    for name,m in models.items():
        t=m['test'];rows.append([labels[name],f'{m["parameters"]:,}',fmt(100*m['train_probe_mean_m']),fmt(100*m['selection_mean_m']),fmt(100*t['mean_m']),fmt(100*t['p95_m']),fmt(t['worse_over_0cm_pct']),fmt(t['worse_over_25cm_pct']),fmt(100*t['max_added_error_m'])])
    body=['<h2 id="capacity">9. Can more training, a larger network or more data help?</h2>',
          '<p>Four controlled changes were actually trained. Fresh width/depth/data variants used the same 2.048M sampled-endpoint budget as the original network. Continuation added another 2.048M endpoints. Checkpoints were chosen only on the selection games. The table uses one common 500,000-row subset of the old test games for every variant.</p>',
          table(['Variant','Parameters','Train-pool probe cm','Selection cm','Same test subset cm','p95 cm','Worse >0 cm %','Worse >25 cm %','Max extra error cm'],rows)]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5));x=np.arange(len(names));short=['64×1','64×1\nlonger','128×1','64×2','40 games']
    for off,field,label in [(-.25,'train_probe_mean_m','Training-pool probe'),(0,'selection_mean_m','Selection')]:axes[0].bar(x+off,[100*models[k][field] for k in names],.25,label=label)
    axes[0].bar(x+.25,[100*models[k]['test']['mean_m'] for k in names],.25,label='Common test subset')
    axes[0].set(xticks=x,xticklabels=short,ylabel='Mean position error (cm)',ylim=(0,68));axes[0].legend(fontsize=8,loc='upper center',ncol=3)
    axes[1].bar(x,[models[k]['test']['worse_over_25cm_pct'] for k in names],color='#ce7546');axes[1].set(xticks=x,xticklabels=short,ylabel='Samples made >25 cm worse than WM (%)')
    body.append(savefig(out,'followup-capacity-comparison',fig))
    winner=min(models,key=lambda k:models[k]['selection_mean_m']);m=models[winner]
    if m.get('full_test'):
        t=m['full_test'];body.append(f'<p><b>Full-cohort check of the validation-selected candidate:</b> {html.escape(labels[winner])}, {t["n"]:,} samples, mean error <b>{100*t["mean_m"]:.2f} cm</b>, p95 {100*t["p95_m"]:.2f} cm, additional error >25 cm on {t["worse_over_25cm_pct"]:.2f}% of samples. This model was chosen by selection error, not by test rank.</p>')
    else:body.append('<p>The validation-selected winner is also being checked on the full old-test cohort; the controlled table above remains a common-subset comparison.</p>')
    body+=['<ul><li><b>More updates helped:</b> the original selection curve was still improving at epoch10; continued training reduced error further.</li><li><b>More width helped in this run:</b> 128 units improved selection and test error at the same fresh-training endpoint budget.</li><li><b>A second layer also helped, but less than width for mean error:</b> it changes both capacity and computation, so layer count alone is not a guarantee.</li><li><b>More source games alone did not visibly help at this fixed update budget:</b> 40 and 160 games produced almost identical mean error. This does not show that additional data can never help; it isolates diversity with the same number of sampled endpoints.</li><li><b>No large train-pool/selection gap is visible in these probes.</b> The probe is not the exact set of optimized endpoints. For the 40-game model it also includes other original training games, so it must not be read as its in-sample training error.</li></ul>',
           '<p>These are one-seed exploratory comparisons, not a search proving the best architecture. More games, more optimizer updates and deliberately covering difficult situations are separate interventions. For rare failure cases, choose sampling/weights using training/validation data while retaining natural sample weighting in evaluation.</p>',
           '<p><a href="followup-capacity.json">Exact results, configurations and game split</a>. Checkpoints and TensorBoard events remain under logs/exp5-s1/scaling-study.</p>']
    return '\n'.join(body)
