"""Readable engineering alternatives and the final research conclusions."""
from pathlib import Path
import json,html
from followup_reporting import table,fmt

def diagram(tree):
    positions={};edges=[];leaves=[]
    def visit(node,depth):
        if 'use_model'in node:
            x=100+len(leaves)*260;leaves.append(node)
        else:
            left=visit(node['left'],depth+1);right=visit(node['right'],depth+1);x=(left+right)/2
            edges.extend([(id(node),id(node['left']),'Yes'),(id(node),id(node['right']),'No')])
        positions[id(node)]=(x,50+depth*120,node);return x
    visit(tree,0);parts=['<svg xmlns="http://www.w3.org/2000/svg" style="font-family:system-ui,sans-serif" viewBox="0 0 1000 470" role="img" aria-label="Exact simplified mean-focused selector; display thresholds rounded"><title>Seven-node learned selection rule</title>']
    for parent,child,label in edges:
        x,y,_=positions[parent];xx,yy,_=positions[child]
        parts.append(f'<line x1="{x}" y1="{y+28}" x2="{xx}" y2="{yy-28}" stroke="#65818a" stroke-width="2"/><text x="{(x+xx)/2+6}" y="{(y+yy)/2}" font-size="13">{label}</text>')
    names={'correction_norm':'Model–WM distance (m)','correction_per_cycle':'Correction / named-sighting age (m/c)','wm_ball_distance':'WM player–ball distance (m)','last_speed':'Last available speed (m/c)'}
    for x,y,node in positions.values():
        leaf='use_model'in node;width=150 if leaf else 260;color=('#bce3dc' if node.get('use_model') else '#f3e0ba') if leaf else '#e6eef1'
        parts.append(f'<rect x="{x-width/2}" y="{y-28}" width="{width}" height="56" rx="8" fill="{color}" stroke="#65818a"/>')
        if leaf:parts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="15">'+('Use model' if node['use_model'] else 'Keep WM')+'</text>')
        else:
            label=html.escape(names.get(node['name'],node['name']))
            parts.append(f'<text x="{x}" y="{y-6}" text-anchor="middle" font-size="12">{label}</text><text x="{x}" y="{y+15}" text-anchor="middle" font-size="14">≤ {node["threshold"]:.6g}?</text>')
    return ''.join(parts)+'</svg>'

def tree_followup(out):
    out=Path(out)
    if not all((out/f).exists() for f in ['refinement-tree-fresh.json','refinement-pruned-trees.json','refinement-fresh.json']):return ''
    trees=json.loads((out/'refinement-tree-fresh.json').read_text());pruned=json.loads((out/'refinement-pruned-trees.json').read_text());fresh=json.loads((out/'refinement-fresh.json').read_text())
    mean=fresh['policies']['mean']['metrics'];risk=fresh['policies']['near50_risk']['metrics']
    simplemean=trees['models']['continue128/tree74_cost3']['metrics'];simplerisk=trees['models']['wmresidual128_l1/tree74_cost5']['metrics']
    rows=[]
    for label,size,v in [('Mean focused: boosted','180 trees',mean),('Mean focused: simplified tree','7 nodes',simplemean),('Practical: boosted','180 trees',risk),('Practical: simplified tree','11 nodes',simplerisk)]:
        a=v['all'];n=v['under50'];rows.append([label,size,fmt(100*a['mean']),fmt(100*n['mean']),fmt(n['worse']['0']),fmt(n['worse']['25']),fmt(n['worse']['50']),fmt(n['max_added'])])
    (out/'refinement-seven-node-tree.svg').write_text(diagram(pruned['continue128']['tree']))
    body=['<h2 id="simple-tree-alternative">17. A tiny decision tree is a credible engineering alternative</h2>',
          '<p>The same-input controls fit new trees using only original selection games and tune thresholds on original calibration games. We additionally score their fixed choices on the new-game batch. This is <b>additional exploratory reuse</b> after primary confirmation; it does not change the original frozen policies or serve as another untouched confirmatory test.</p>',
          table(['Method','Selector size','All mean cm','Under50m mean cm','Under50m worse >0cm %','>25 cm %','>50cm %','Max added error m'],rows),
          f'<p><b>Accuracy versus complexity:</b> the small mean-focused tree adds only {100*(simplemean["under50"]["mean"]-mean["under50"]["mean"]):.2f} cm to mean error below 50 m compared with 180 boosted trees. The small practical tree adds{100*(simplerisk["under50"]["mean"]-risk["under50"]["mean"]):.2f} cm; its >25 cm-harm rate is slightly lower, while its any-harm rate is higher. Neither choice wins every metric.</p>',
          '<p>At each fixed threshold, branches whose leaves all make the same final choice can be merged. The 15-node mean tree becomes 7 nodes; the 49-node practical tree becomes 11 nodes. This preserves the decision for every input by construction; both exports also matched their original tree on all 500,000 calibration rows. No retraining or threshold change occurs. The compact trees output a choice, not calibrated confidence; changing the decision threshold requires exporting a new compact tree.</p>',
          '<h3>The actual seven-node mean-focused rule</h3>',
          '<p>First preserve WM for posCount0 and other ineligible cases. Then use this learned rule. “Correction” is the distance between the raw network prediction and WM; age is cycles since the last confirmed named sighting. Unknown ball position uses the same fallback as the training features.</p>',diagram(pruned['continue128']['tree']),
          '<p>Displayed thresholds are rounded. <a href="refinement-pruned-trees.json">The exact exported trees</a> retain full-precision thresholds and the float32 input contract.</p>',
          '<details><summary>The eleven-node practical rule</summary><pre>'+html.escape(pruned['wmresidual128_l1']['text'])+'</pre></details>',
          '<h3>What we learned, and what is worth keeping</h3>',
          '<ul><li><b>More optimizer updates helped.</b> Ordinary continuation gave the strongest mean-focused neural candidate.</li><li><b>Training against harmful replacements helped a different objective.</b> The WM-centered residual network and a severity-aware selector gave the designated practical policy.</li><li><b>Heavy weighting of already-good WM rows did not win the validation selection.</b> Its lower regression frequency came with a larger mean-error cost.</li><li><b>Extra context and selector complexity had small returns for mean accuracy.</b> A seven-node choice rule preserved almost all of the boosted selector’s mean improvement. Risk-focused comparisons still show tradeoffs.</li><li><b>Fixed or age-dependent partial corrections did not beat the uncapped choices on the selected validation utilities.</b> Hard ceilings remain explicit alternatives when bounding the size of a harmful correction matters more than mean accuracy.</li></ul>']
    v=fresh['policies']['near50_risk'];good=v['under50_subgroups']['WM error<10cm'];long=v['under50_subgroups']['named age>10'];nearball=v['under50_subgroups']['estimated target-ball<2m']
    body += [f'<p><b>What remains imperfect:</b> below 50 m, when WM already had <10cm error, its mean {100*good["stock"]:.2f} cm became {100*good["mean"]:.2f} cm under the primary boosted practical policy. For gaps above 10 cycles, WM mean {100*long["stock"]:.2f} cm became {100*long["mean"]:.2f} cm, but the largest observed added error remained {long["max_added"]:.2f} m. The largest fresh-game regressions occurred after 29-cycle gaps. Hidden actions during a long gap remain ambiguous.</p>',
             f'<p>For targets estimated within 2 m of the ball and truly within 50 m of the observer, the primary boosted practical policy reduced mean error from {100*nearball["stock"]:.2f} to {100*nearball["mean"]:.2f} cm;{nearball["worse"]["25"]:.2f}% were made >25 cm worse. This is a position diagnostic, not a measured pass or possession benefit.</p>',
             '<p><b>Engineering candidates retained:</b> continued GRU128 + seven-node tree for low mean error with a simple selector; WM-centered residual GRU128 + eleven-node tree for a practical accuracy/risk tradeoff. The 180-tree versions retain the primary frozen-policy evidence and the best tested mean. Hard correction caps are separate choices with a substantial average-error cost.</p>',
             '<p><a href="refinement-handoff.md">Inputs, outputs, resets, coordinate handling and future integration handoff</a>. Preserve the unmodified WM feature stream and the trained rolling 32-cycle history. A future shadow/live test must measure decision timing, state consistency and corrected-agent feedback separately.</p>']
    return '\n'.join(body)
