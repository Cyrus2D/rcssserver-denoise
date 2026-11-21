"""Standalone visual report for live_audit.py; charts embedded in HTML."""
from pathlib import Path
import csv
import html
import io
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AGES = ['0', '1', '2', '3', '4', '5', '>5']
KINDS = ['self', 'ball', 'teammate', 'opponent']
VIEWS = ['narrow', 'normal', 'wide']
COLORS = ['#087e8b', '#cf6840', '#5179b0', '#9a598b']
UNITS = dict(position='cm', velocity='cm/cycle', body='degrees')
FACTORS = dict(position=100, velocity=100, body=1)
SENSOR_LABELS = dict(n_dist='Raw range (m)', n_dir='Raw bearing (degrees)', n_dist_chg='Raw radial velocity (m/cycle)',
    n_dir_chg='Raw angular velocity (degrees/cycle)', n_body_rel='Raw relative body (degrees)', n_head_rel='Raw relative face (degrees)',
    n_rawfix_pos_d='Visual self-position fix (m)', n_rawfix_face='Visual looking-direction fix (degrees)',
    n_sb_speed_mag='Sense-body speed magnitude (m/cycle)', n_sb_speed_dir='Sense-body velocity direction (degrees)', n_sb_neck='Sense-body neck (degrees)')


def render(out, data):
    out = Path(out)
    m, c = data['metrics'], data['counts']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'bold'})
    plots = {}

    def embed_svg(raw, prefix):
        raw=raw[raw.index('<svg'):]
        for ident in set(re.findall(r'\bid="([^"]+)"',raw)):
            replacement=prefix+'-'+ident
            raw=raw.replace('id="'+ident+'"','id="'+replacement+'"').replace('#'+ident+'"','#'+replacement+'"').replace('#'+ident+')','#'+replacement+')')
        return raw

    def metric(kind, what, axis='own', age='all', view='all', field='mean'):
        return m.get(f'state/{kind}/{what}/{axis}/{age}/{view}', {}).get(field, np.nan)

    def save(name, fig):
        fig.tight_layout(pad=2)
        fig.savefig(out / (name + '.svg'), bbox_inches='tight')
        fig.savefig(out / (name + '.png'), dpi=130, bbox_inches='tight')
        plt.close(fig)
        raw = (out / (name + '.svg')).read_text()
        plots[name] = embed_svg(raw,name)

    for what in ['position', 'velocity', 'body']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for ax, kinds, title in [(axes[0], ['self'], 'Self'), (axes[1], KINDS[1:] if what != 'body' else KINDS[2:], 'Other objects')]:
            for kind in kinds:
                y = [FACTORS[what] * metric(kind, what, 'own', age) for age in AGES]
                ax.plot(AGES, y, marker='o', lw=2.3, color=COLORS[KINDS.index(kind)], label=kind.title())
                for age,value in zip(AGES,y):
                    n=metric(kind,what,'own',age,field='n')
                    if np.isfinite(value) and np.isfinite(n) and n<1000:
                        ax.annotate(f'n={int(n)}',(age,value),xytext=(-8,-17),textcoords='offset points',ha='right',fontsize=9)
            ax.set(title=title, xlabel=('Face' if what == 'body' and kinds == ['self'] else what.title()) + ' count', ylabel=f'Mean {what} error ({UNITS[what]})')
            ax.grid(alpha=.18); ax.legend()
        save(what + '-age', fig)

    if (out/'ball-missing-velocity-case.csv').exists() and '202611151205_13776' in data['games']:
        import pandas as pd
        case=pd.read_csv(out/'ball-missing-velocity-case.csv')
        fig,axes=plt.subplots(1,2,figsize=(12,4.3))
        motion=case.dist_chg.notna() & case.dir_chg.notna()
        posonly=case.seen.eq(1) & ~motion
        missing=~case.seen.eq(1)
        for ax,field,title,unit in [(axes[0],'n_trk_pos_d','Ball position error','m'),(axes[1],'n_trk_vel_d','Ball velocity error','m/cycle')]:
            ax.plot(case.cycle,case[field],color='#567080',lw=1.5)
            for mask,label,color,marker in [(motion,'Position + motion seen','#087e8b','o'),(posonly,'Position only seen','#cf6840','s'),(missing,'No ball observation','#6b7280','x')]:
                ax.scatter(case.loc[mask,'cycle'],case.loc[mask,field],c=color,marker=marker,s=65,label=label,zorder=3)
            ax.axvline(1891,color='#cf6840',ls=':',alpha=.6)
            ax.set(title=title,xlabel='Cycle (observer right 11)',ylabel=unit,xticks=case.cycle)
            ax.tick_params(axis='x',rotation=35);ax.grid(alpha=.15)
        axes[1].legend(fontsize=9)
        save('ball-field-gap-case',fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    view_max = max((100 * metric(k, 'position', 'pos', age, w) for k in KINDS[1:] for age in AGES for w in VIEWS
                    if np.isfinite(metric(k, 'position', 'pos', age, w))), default=1.)
    for ax, kind in zip(axes, KINDS[1:]):
        z = np.array([[100 * metric(kind, 'position', 'pos', age, width) for age in AGES] for width in VIEWS])
        ax.imshow(z, aspect='auto', cmap='YlOrRd', vmin=0, vmax=view_max)
        for y in range(3):
            for x in range(len(AGES)):
                ax.text(x, y, '-' if not np.isfinite(z[y,x]) else f'{z[y,x]:.1f}', ha='center', va='center', fontsize=9,
                        color='white' if np.isfinite(z[y,x]) and z[y,x] > view_max * .65 else '#183142')
        ax.set(title=kind.title(), xlabel='Position count', xticks=range(7), xticklabels=AGES, yticks=range(3), yticklabels=['60 degrees', '120 degrees', '180 degrees'])
    fig.suptitle('Mean position error (cm): compare view widths within the same age', fontsize=14)
    save('view-age', fig)

    for what, kinds in [('velocity',KINDS[1:]),('body',['self','teammate','opponent'])]:
        fig,axes=plt.subplots(1,3,figsize=(13,4.2))
        zlist=[np.array([[FACTORS[what]*metric(k,what,'own',age,w) for age in AGES] for w in VIEWS]) for k in kinds]
        finite=np.concatenate([z[np.isfinite(z)] for z in zlist]); upper=max(float(finite.max()),1e-12) if len(finite) else 1.
        for ax,kind,z in zip(axes,kinds,zlist):
            ax.imshow(z,aspect='auto',cmap='YlOrRd',vmin=0,vmax=upper)
            for y in range(3):
                for x in range(7):
                    ax.text(x,y,'-' if not np.isfinite(z[y,x]) else f'{z[y,x]:.1f}',ha='center',va='center',fontsize=9,color='white' if np.isfinite(z[y,x]) and z[y,x]>.65*upper else '#183142')
            ax.set(title=kind.title(),xlabel=('Face count' if kind=='self' and what=='body' else what.title()+' count'),xticks=range(7),xticklabels=AGES,yticks=range(3),yticklabels=['60 degrees','120 degrees','180 degrees'])
        fig.suptitle('Mean '+what+' error ('+UNITS[what]+'): own field counter held fixed',fontsize=14)
        save('view-'+what,fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, kind in zip(axes, KINDS[1:]):
        bottom = np.zeros(3)
        for i, age in enumerate(AGES):
            n = np.array([c.get(f'age_rows/{kind}/{w}/{age}', 0) for w in VIEWS], float)
            totals = np.array([sum(c.get(f'age_rows/{kind}/{w}/{a}', 0) for a in AGES) for w in VIEWS])
            y = np.divide(n * 100, totals, out=np.zeros(3), where=totals > 0)
            ax.bar(['60', '120', '180'], y, bottom=bottom, label=age, color=plt.cm.viridis(i / 7))
            bottom += y
        ax.set(title=kind.title(), xlabel='Actual view width (degrees)', ylabel='Share of named tracked rows (%)', ylim=(0,100))
    axes[-1].legend(title='Position count', bbox_to_anchor=(1.02, 1))
    save('age-exposure', fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, what in zip(axes, ['position', 'velocity', 'body']):
        y = [FACTORS[what] * metric('self', what, 'seen', str(s)) for s in [1, 0]]
        ax.bar(['See update', 'No see update'], y, color=['#087e8b', '#cf6840'])
        ax.set(title='Self ' + what, ylabel=UNITS[what])
        for i, value in enumerate(y):
            if np.isfinite(value): ax.annotate(f'{value:.4f}', (i, value), xytext=(0,5), textcoords='offset points', ha='center')
        ax.margins(y=.2)
    save('self-see', fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, kind in zip(axes, KINDS[1:]):
        for age, color in zip(['0','1','2','3','>5'], plt.cm.viridis(np.linspace(.05,.85,5))):
            y = [100 * metric(kind, 'position', 'distance', age, d) for d in ['0-3','3-10','10-20','20-40','40+']]
            ax.plot(['0-3','3-10','10-20','20-40','40+'], y, marker='o', label=age, color=color)
        ax.set(title=kind.title(), xlabel='True observer-target distance (m)', ylabel='Mean position error (cm)'); ax.grid(alpha=.2)
    axes[-1].legend(title='Position count', bbox_to_anchor=(1.02,1))
    save('distance-age', fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, what in zip(axes, ['position','velocity','body']):
        for kind in KINDS:
            if what == 'body' and kind == 'ball': continue
            y = []
            for age in AGES:
                n = metric(kind, what, 'own', age, field='n')
                k = metric(kind, what, 'own', age, field='above_threshold')
                y.append(100 * k / n if np.isfinite(n) and n else np.nan)
            ax.plot(AGES, y, marker='o', color=COLORS[KINDS.index(kind)], label=kind.title())
        ax.set(title=f'{what.title()}: error > {data["thresholds"][what]} ' + ('m' if what == 'position' else 'm/cycle' if what == 'velocity' else 'degrees'),
               xlabel='Own metric count (self body: face count)', ylabel='Rows above threshold (%)'); ax.grid(alpha=.2)
    axes[-1].legend(bbox_to_anchor=(1.02,1))
    save('large-errors', fig)

    # Machine-readable aggregate export includes denominators without cluttering the main tables.
    with (out / 'tables.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['group','samples','mean_native_units','rmse_native_units','max_native_units','above_threshold'])
        for key, v in sorted(m.items()): writer.writerow([key, v['n'], v['mean'], v['rmse'], v['max'], v['above_threshold']])

    def table(headers, rows):
        return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(str(h))+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'

    def val(v, digits=3): return '-' if not np.isfinite(v) else f'{v:,.{digits}f}'

    def state_table(what, axis):
        kinds = KINDS if what != 'body' else ['self','teammate','opponent']
        return table(['Count']+[k.title() for k in kinds], [[html.escape(age)]+[val(FACTORS[what]*metric(k,what,axis,age)) for k in kinds] for age in AGES])

    sensor = [(k,v) for k,v in sorted(m.items()) if k.startswith('sensor/')]
    failures = [(k,v) for k,v in sensor if v['max'] > data['sensor_tolerance']]
    total_sensor = sum(v['n'] for _,v in sensor)
    nonzero_sensor = sum(v['above_threshold'] for _,v in sensor)
    worst_sensor = max((v['max'] for _,v in sensor), default=float('nan'))
    status = 'PASS' if sensor and not failures else 'CHECK REQUIRED'
    sections = []
    sections.append(f'<p class="eyebrow">EXP5 / LIVE BASELINE / {len(data["games"])} GAMES</p><h1>What happens when vision is incomplete?</h1><p class="lead">Exact visible measurements, limited view, ordinary world-model prediction. This report measures the error that remains when observations become old.</p>')
    sections.append('<div class="cards">'+f'<div><strong>{status}</strong><span>Raw kinematic measurement audit</span></div><div><strong>{c["decisions"]:,}</strong><span>PlayOn observer decisions</span></div><div><strong>{100*c["see_updates"]/c["decisions"]:.1f}%</strong><span>Decisions with a see update</span></div></div>')
    sections.append('<nav><a href="#exact">Sensor check</a><a href="#position">Position</a><a href="#velocity">Velocity</a><a href="#body">Body</a><a href="#view">View angle</a><a href="#tails">Large errors</a><a href="#method">Method</a></nav>')
    sections.append('<div class="note"><b>Read this distinction first.</b> Sensor noise is an error in a measurement received now. Prediction error is the difference between the current world model and the true state while information is old. Turning off quantization removes the first; it cannot reveal unseen dash, turn, kick, collision or random physics outcomes.</div>')
    sections.append('<h2>What the main numbers say</h2><ul>'+f'<li>Self position error averages <b>{val(100*metric("self","position","pos","1"))} cm at position count 1</b> and <b>{val(100*metric("self","position","pos","2"))} cm at count 2</b>. Use the separate self scale when reading the plots.</li>'+f'<li>Teammate/opponent position errors are <b>{val(100*metric("teammate","position","pos","1"))} / {val(100*metric("opponent","position","pos","1"))} cm at count 1</b>, rising to <b>{val(100*metric("teammate","position","pos","5"))} / {val(100*metric("opponent","position","pos","5"))} cm at count 5</b>.</li>'+f'<li>Ball position error is <b>{val(100*metric("ball","position","pos","1"))} cm at count 1</b> and <b>{val(100*metric("ball","position","pos","5"))} cm at count 5</b>. Unseen motion changes make its long-gap behavior different from player motion.</li>'+'<li>Count-0 player outliers and missing old velocity/body fields are explained below. They must be handled explicitly when building a prediction dataset.</li></ul>')
    sections.append(f'<h2 id="exact">1. Did live measurements really have no quantization noise?</h2><p><b>{nonzero_sensor:,} of {total_sensor:,} checked scalar readings exceeded {data["sensor_tolerance"]} in their native units.</b> Largest absolute raw/fix error: {worst_sensor:.8f}. Tests include available range, bearing, motion, body/face and self sense-body fields. Missing fields are excluded, never filled with zero.</p>')
    sections.append(table(['Object / measurement','Mean absolute error','Largest error','Readings'], [[k.split('/')[1].title()+' / '+SENSOR_LABELS.get(k.split('/')[-1], k.split('/')[-1]), f'{v["mean"]:.8f}',f'{v["max"]:.8f}',f'{v["n"]:,}'] for k,v in sensor]))
    sections.append('<p>Checks use same-cycle fullstate labels and unambiguous identified visual entries. Player measurements matched only by a guessed bearing are excluded from this sensor claim. Full packet and landmark logs were disabled in this run; individual landmark values cannot be re-audited here. Fresh self fixes still test the resulting localization. Zero means indistinguishable from zero at the stored precision, not infinite mathematical precision.</p>')
    rows=[]
    for kind in KINDS[1:]:
        for field,label in [('n_ls_pos_d','Position (cm)'),('n_ls_vel_d','Velocity (cm/cycle)')]:
            v=m.get(f'localization/{kind}/{field}',{})
            if v: rows.append([label,kind.title(),val(100*v['mean'],6),val(100*v['max'],6),f'{v["n"]:,}'])
    sections.append('<details><summary>Current visual localization before WM fusion</summary>'+table(['Metric','Object','Mean error','Largest error','Samples'],sorted(rows))+'<p>Global localization also depends on the observer pose. A see message can contain no usable position landmark, leaving self position older while updating face. An exact relative measurement can then inherit that self-position error. Localized ball velocity can also be inferred when a raw motion field is absent; it is not automatically a direct sensor reading.</p></details>')
    identity_rows=[]
    for kind in KINDS[1:]:
        for label in ['confirmed','no_named_observation']:
            v=m.get(f'fresh_identity/{kind}/position/{label}')
            if v: identity_rows.append([kind.title(),label.replace('_',' '),f'{v["n"]:,}',val(100*v['mean'],5),val(v['max'],4),f'{100*v["above_threshold"]/v["n"]:.3f}'])
    sections.append('<h3>Count 0 does not guarantee the correct player identity</h3>'+table(['Object','Current visual confirmation','Rows','Mean error (cm)','Largest error (m)','Error >1 m (%)'],identity_rows)+'<p>The main WM curves retain these mistakes because they affect the live agent. A named track can be refreshed by an anonymous near-field observation of another player. The logger then compares its stored uniform number against that player\'s truth. That is a real association failure, not residual range/bearing quantization. The confirmed rows require an unambiguous current measurement with the target uniform number (ball: an unambiguous current sighting).</p>')
    ball_fresh=m.get('fresh_identity/ball/velocity/confirmed')
    if ball_fresh:
        sections.append('<h3>A fresh ball position can still have an old velocity</h3>'+f'<p>On current ball sightings with position count 0, mean velocity error was {100*ball_fresh["mean"]:.4f} cm/cycle, with a maximum of {ball_fresh["max"]:.6f} m/cycle. A fresh position alone is insufficient evidence that velocity was observed. Keep field-specific masks and counters in the later predictor.</p>')
        if '202611151205_13776' in data['games']:
            sections.append('<p>One example: game 202611151205_13776, observer right 11, cycle 1891 (PlayOn). The ball was 2.799137 m away and its position was fresh, but both raw motion fields were absent. The retained velocity had count 2 and error 4.902423 m/cycle. This is an old velocity estimate rather than a noisy raw velocity reading.</p>')
        if 'ball-field-gap-case' in plots:
            sections.append(plots['ball-field-gap-case']+'<p>At cycle 1891 the orange square restores position to zero error while velocity remains wrong. Subsequent unseen position predictions drift again. The near-field ball was outside the view cone and supplied no motion fields. <a href="ball-missing-velocity-case.csv">Saved case rows</a>.</p>')
    sections.append('<h2>2. Self between visual fixes</h2>'+plots['self-see']+'<p>Each bar uses the available valid errors on decisions with or without a see update. These are descriptive cohorts. A see update is not necessarily a successful self-position fix: it can update face without providing a usable position landmark. Exact sense-body speed does not make global velocity exact if the estimated looking direction has drifted. Successful fresh visual position fixes are checked separately above.</p>')
    if '202611151202_8681' in data['games']:
        sections.append('<p>Example: observer right 5, game 202611151202_8681, cycle 3924 received a narrow-view see message, with exact face and velocity, but no new self-position fix. Position count remained 1 and position error was 9.3963 cm; the previous successful fix at cycle 3923 was exact. This is missing position information, not sensor quantization.</p>')
    for i,what in enumerate(['position','velocity','body'],3):
        sections.append(f'<h2 id="{what}">{i}. {what.title()} error by its own counter</h2>'+plots[what+'-age'])
        sections.append('<p>'+({'position':'Position error distance = ||estimated position - true current position||. Values below are centimeters.',
            'velocity':'Velocity error distance = ||estimated velocity - true current velocity||. Values below are centimeters per cycle. Velocity count can differ from position count.',
            'body':'Body error = |wrap180(estimated body - true current body)|, in degrees. Self uses face count because librcsc shares face validity with body. The ball has no body angle.'}[what])+'</p>')
        if what=='position':
            sections.append('<p><code>e_pos(t) = sqrt((x_WM(t) - x_true(t))^2 + (y_WM(t) - y_true(t))^2)</code><br>Both positions refer to the current cycle t. For each count group, the table averages e_pos over its valid samples, then converts meters to centimeters.</p>')
            small=[(age,metric('self','position','own',age,field='n')) for age in AGES]
            small=[(a,int(n)) for a,n in small if np.isfinite(n) and n<1000]
            if small: sections.append('<p><b>Small self cohorts:</b> '+', '.join(f'count {html.escape(a)} has {n:,} samples' for a,n in small)+'. Treat those points as diagnostic examples rather than a stable error estimate.</p>')
        sections.append(state_table(what,'own'))
        if what=='velocity':
            sections.append('<p><b>Why some player columns are empty:</b> the current PlayerObject considers velocity invalid at velocity count 5 or higher. The logger intentionally leaves invalid velocity/error fields empty, so those own-count errors cannot be measured from this dataset. The position-count table below answers a different question using whichever velocity values remain valid.</p>')
        if what=='body':
            sections.append('<p><b>Why player body counts 2 and above are empty:</b> the current PlayerObject considers body direction invalid at body count 2 or higher. Those errors were not logged and cannot be recovered here. Empty cells do not mean old body directions are accurate. Internal stale values would require a separate diagnostic logger in a future run.</p>')
        if what != 'position': sections.append('<details><summary>Same metric grouped by position count (your requested common comparison)</summary>'+state_table(what,'pos')+'</details>')
        sections.append('<p class="caption">These are librcsc freshness/quality counters. Position count normally increases once per unseen cycle and resets on an update. Velocity/body counts can also describe an inferred field: for example, the library can infer body direction from movement and assign count 0 without receiving a body measurement. They are not universally a stopwatch since the last direct sensor reading. The &gt;5 group mixes all larger counts; missing groups are shown as a dash.</p>')
    sections.append('<h2 id="view">6. View width: accuracy and exposure are different questions</h2>'+plots['view-age']+'<p>Heatmap values are centimeters. Compare rows within a column to hold position age fixed. Current width alone does not identify which width produced the last sighting; adaptive view choices also depend on the game situation. These are associations, not proof that one width causes lower error.</p>'+plots['age-exposure']+'<p>These bars show how often a named track is fresh or old under each current width. A 60-degree view updates more frequently but sees less of the field; 180 degrees covers more space with slower updates. Even with exact sensors, objects outside the cone can remain unseen for many cycles.</p>')
    rows=[]
    for kind in KINDS[1:]:
        for w in VIEWS:
            total=c.get(f'visibility/{kind}/{w}/total',0)
            rows.append([kind.title(),dict(narrow='60',normal='120',wide='180')[w],val(100*c.get(f'visibility/{kind}/{w}/seen',0)/total if total else np.nan,2),val(100*c.get(f'visibility/{kind}/{w}/outside_cone',0)/total if total else np.nan,2)])
    sections.append(table(['Object','View degrees','Seen this cycle (%)','Outside current cone (%)'],rows))
    sections.append('<h3>Velocity and body under each view width</h3>'+plots['view-velocity']+plots['view-body']+'<p>Velocity and body use their own counters here. Compare the same column within each object panel. Blank cells mean no valid samples, rather than zero error. The same caution about adaptive view selection applies.</p>')
    sections.append('<h2>7. Distance and age together</h2>'+plots['distance-age']+'<p>The distance filter uses the true distance from observer to target at that cycle. In exact mode, distance should not create a quantization floor for fresh measurements. Larger errors at older ages reflect motion, visibility and tracking, rather than distance quantization.</p>')
    sections.append('<h2 id="tails">8. How often are errors large?</h2>'+plots['large-errors']+'<p>Mean error alone can hide a small set of costly mistakes. These plots show the proportion among valid logged fields above explicit reference thresholds; invalid/missing tracks are excluded, not counted as successes. Thresholds are reporting aids, not calibrated decision safety margins.</p>')
    sections.append('<h2 id="method">9. Scope, validity and reproducibility</h2><p>'+html.escape(data['method'])+'</p>')
    if data.get('configurations'):
        sections.append('<details><summary>Verified run configuration</summary>'+table(['Setting','Values across games'],[[html.escape(k),html.escape(', '.join(v))] for k,v in data['configurations'].items()])+'<p>View policies: '+html.escape(', '.join(data.get('view_policies',[])))+'. '+f'{c["observers"]:,} finalized observers; final metadata cycle range {html.escape(str(data.get("observer_last_cycle_range",[])))}.</p></details>')
    if (out/'run-audit.json').exists():
        checked=json.loads((out/'run-audit.json').read_text())
        if checked.get('games')==data['games']:
            sections.append('<p>Additional live-content check: '+f'{checked["counts"]["hear_files"]:,} hearing log files contained {checked["counts"].get("hear_rows",0):,} rows and {checked["counts"].get("content_rows",0):,} payload rows. All metadata confirmed reference-only fullstate. <a href="run-audit.json">Verification details</a>.</p>')
    if (out/'match-results.txt').exists():
        sections.append('<details><summary>Completed match outcomes (baseline versus baseline)</summary><pre>'+html.escape((out/'match-results.txt').read_text())+'</pre><p>Both teams use the baseline without neural correction. These outcomes verify collection completion; they do not measure the benefit of a denoising or prediction model.</p></details>')
    rows=[]
    for kind in KINDS[1:]:
        for what in ['position','velocity','body']:
            if kind=='ball' and what=='body':continue
            valid=c.get(f'measured/{kind}/{what}',0); eligible=c.get(f'potential/{kind}',0)
            rows.append([what.title(),kind.title(),f'{valid:,}',f'{eligible:,}',val(100*valid/eligible if eligible else np.nan,2)])
    sections.append('<details><summary>Available data and exclusions (sample denominators)</summary>'+table(['Metric','Object','Valid named-track errors','All truth object rows','Valid share (%)'],sorted(rows))+'<p>Rows without a valid track or field are absent from error means. Unknown-player tracks matched to truth by geometric proximity are excluded from the main state results. Availability drops at large ages, so a flat long-gap curve can reflect lost tracks. Samples and maxima for every group are retained in the downloadable aggregates.</p></details>')
    sections.append(f'<p>Only finalized games with all 22 status-ok agent logs are included. Only actual decisions in PlayOn with stopped=0 and same-cycle truth are scored. There were {c.get("missed_decision_placeholders",0):,} logged missed-decision placeholders in PlayOn; these are excluded. Each eligible observer-target-cycle has equal weight; games do not receive equal weight. Angular errors are wrapped, then made absolute. A view update for the observer does not mean every target was seen. Near-field anonymous observations retain native server behavior. Actuator and physics randomness remain enabled.</p>')
    sections.append('<p>Position/velocity are read from precomputed n_* error labels calculated before CSV rounding. WM coordinates themselves have four decimals, error labels six. Precision tolerances account for storage; do not reinterpret a numerical floor as gameplay noise.</p>')
    if (out/'identity-case.svg').exists() and '202611151200_20018' in data['games']:
        raw=(out/'identity-case.svg').read_text()
        sections.append('<h3>A concrete association failure</h3>'+embed_svg(raw,'identity-case')+'<p>Game 202611151200_20018, observer left 9, cycle 1262 (PlayOn). Opponent R3 was not seen, yet its stored track had position count 0 and was 7.357442 m from its true position. The track coincided with the exact anonymous localization of nearby opponent R5 (within CSV coordinate rounding). R5 was outside the view cone but within the native near-field visibility distance. Source inspection explains the mechanism: PlayerObject::updateBySee retains the old uniform number when an anonymous observation has no uniform number, while replacing position and resetting its count. The shaded sector shows the actual 120-degree cone. Truth labels in this diagnostic are for explanation only.</p>')
    sections.append('<h2>10. What this baseline tells us to try next</h2><ol><li><b>Preserve confirmed fresh measurements.</b> There is no quantization error here to denoise again. Concentrate prediction on missing or stale fields.</li><li><b>Separate identity failures from motion errors.</b> Reusing a stale uniform number for an anonymous close player can create a large error even at count 0. A motion predictor should retain identity confidence and reset on uncertain associations.</li><li><b>Evaluate player motion at each gap age.</b> Use whole-game train/validation/test splits and compare against the existing WM and causal constant-velocity baselines on identical rows. Preserve the large-error rates, not just the overall mean.</li><li><b>Treat ball motion changes as a separate regime.</b> Unseen kicks, tackles and collisions invalidate simple inertia. Use only causally available possession/context features, and report near-self and post-action cases separately before live application.</li></ol>')
    sections.append('<p><a href="metrics.json">Full metrics and frozen game list</a> &middot; <a href="tables.csv">Aggregate table data (CSV)</a></p><details><summary>Frozen game IDs</summary><pre>'+html.escape('\n'.join(data['games']))+'</pre></details>')
    style='''body{margin:0;color:#203447;background:#f4f6f6;font:16px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:42px 28px 80px}h1{font-size:42px;line-height:1.15;max-width:900px}h2{margin-top:58px;border-top:1px solid #cad6da;padding-top:28px;line-height:1.3}.eyebrow{color:#087e8b;font-weight:700;letter-spacing:.12em}.lead{font-size:21px;max-width:920px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.cards div{background:white;padding:22px;border-top:4px solid #087e8b}.cards strong{display:block;font-size:29px}.cards span,.caption{color:#526775;font-size:14px}nav{display:flex;gap:18px;flex-wrap:wrap;margin:25px 0}a{color:#006f81}.note{background:#e2f0ef;padding:22px;border-left:4px solid #087e8b}.scroll{overflow-x:auto;background:white}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 14px;border-bottom:1px solid #e0e7e9;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e7edee}tr:hover{background:#f0f8f7}svg{width:100%;height:auto;background:white;margin:15px 0}details{margin:22px 0}summary{cursor:pointer;color:#006f81;font-weight:600}pre{max-height:300px;overflow:auto}@media(max-width:700px){main{padding:24px 14px}h1{font-size:32px}.cards{grid-template-columns:1fr}.cards strong{font-size:24px}}'''
    (out/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp5: live visual-gap baseline</title><style>'+style+'</style><main>'+'\n'.join(sections)+'</main></html>')
    summary = ['# Exp5 live baseline: visual gaps', '', f'{len(data["games"])} finalized games; {c["decisions"]:,} PlayOn observer decisions.', '',f'Raw sensor audit: **{status}**; {nonzero_sensor:,}/{total_sensor:,} checked readings above {data["sensor_tolerance"]}.', '', '[Open the visual report](report.html). All means weight each valid sample equally.', '', '## Mean position error distance (cm)', '', '| Position count | Self | Ball | Teammate | Opponent |','|---|---:|---:|---:|---:|']
    for age in AGES: summary.append('| '+age+' | '+' | '.join(val(100*metric(k,'position','pos',age)) for k in KINDS)+' |')
    summary += ['', 'Sensor exactness does not eliminate unseen action/physics uncertainty. Current view-width groups are descriptive; compare matching ages. Fullstate is scoring only. See HTML for velocity, body, tails, view/distance, validity and scope.']
    (out/'report.md').write_text('\n'.join(summary)+'\n')
