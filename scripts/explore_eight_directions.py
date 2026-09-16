"""Write a dependency-free interactive table for the complete direction screen."""

from __future__ import annotations

import json


def write_explorer(lookup, output, names, label):
    conditions = [
        ("structured", "isolated_pitch"),
        ("structured", "isolated_timing"),
        ("structured", "isolated_joint"),
        ("independent", "simultaneous"),
    ]
    data = []
    for condition, (profile, kind) in enumerate(conditions):
        for n in (1, 2, 4, 6, 8):
            for name in names:
                rows = [
                    lookup[n, profile, kind, m, name]
                    for m in ("pitch_directed", "onset_directed", "joint_directed")
                ]
                data.append(
                    [
                        condition,
                        n,
                        int(name.split("_")[1]),
                        name.endswith("_lw"),
                        label(name),
                        [[r["mean"], r["std"], r["median"]] for r in rows],
                    ]
                )
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>CeL direction screen</title><meta name="viewport" content="width=device-width">
<style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#193344}
h1{font-size:26px}p{line-height:1.5}label{display:inline-block;margin:8px 18px 8px 0}
select{font:inherit;padding:5px}table{border-collapse:collapse;width:100%;margin-top:16px}
th,td{padding:9px;text-align:right;border-bottom:1px solid #dde3e7}
th:first-child,td:first-child{text-align:left}
thead{position:sticky;top:0;background:white}.scroll{max-height:70vh;overflow:auto}
svg{width:24px;height:24px;vertical-align:middle;margin-right:3px}
small{color:#52636d}#required label{margin-right:15px}
</style><h1>All diagonal and orthogonal CeL subsets</h1>
<p>255 nonempty subsets, each with and without Log-Weighing. Percentages measure local
negative-gradient alignment with matched target displacement, not optimisation success.
Hover over a percentage for target-level sample SD and median.</p>
<label>Condition <select id="condition"><option value="0">Pitch-only (one event)</option>
<option value="1">Onset-only (one event)</option><option value="2">Joint error (one event)</option>
<option value="3" selected>Simultaneous errors (all events)</option></select></label>
<label>Events <select id="events"><option>1</option><option>2</option><option selected>4</option>
<option>6</option><option>8</option></select></label>
<label>Weighting <select id="weight"><option value="all">Both</option>
<option value="false" selected>Uniform</option>
<option value="true">Log-Weighing</option></select></label>
<label>Family <select id="family"><option value="all">All</option>
<option value="d">Diagonal only</option>
<option value="o">Orthogonal only</option><option value="m">Mixed</option></select></label>
<label>Sort by <select id="sort"><option value="2">Joint alignment</option>
<option value="0">Pitch alignment</option><option value="1">Onset alignment</option>
<option value="mask">Subset bitmask</option></select></label>
<div id="required">Require directions: </div><p id="count"></p>
<div class="scroll"><table><thead><tr><th>Directions</th><th>Pitch (%)</th>
<th>Onset (%)</th><th>Joint (%)</th></tr></thead><tbody id="rows"></tbody></table></div>
<p><small>Statistics first average candidates within a target, then targets equally.
A dash means no eligible displaced coordinates. Correct-coordinate masks follow Hungarian
matching; some eight-event onset-only slices acquire pitch displacement through reassignment.
Subsets average their raw directional losses equally; the two families are not rescaled.
Rankings across 510 configurations are exploratory.</small></p>
<script>const data=__DATA__;
const arrows=['↗','↘','↖','↙','↑','↓','→','←'];
const el=id=>document.getElementById(id);
const vectors=[[1,-1],[1,1],[-1,-1],[-1,1],[0,-1],[0,1],[1,0],[-1,0]];
const directionNames=['up-right','down-right','up-left','down-left','up','down','right','left'];
function icon(i){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
svg.setAttribute('viewBox','0 0 24 24');svg.setAttribute('aria-label',directionNames[i]);
const title=document.createElementNS(ns,'title');title.textContent=directionNames[i];
svg.append(title);
const [vx,vy]=vectors[i],norm=Math.hypot(vx,vy),dx=vx/norm,dy=vy/norm;
const x=12+8*dx,y=12+8*dy,path=document.createElementNS(ns,'path');
path.setAttribute('d',`M${12-8*dx},${12-8*dy}L${x},${y}
M${x-5*dx-4*dy},${y-5*dy+4*dx}L${x},${y}L${x-5*dx+4*dy},${y-5*dy-4*dx}`);
path.setAttribute('stroke','currentColor');path.setAttribute('stroke-width','1.8');
path.setAttribute('fill','none');svg.append(path);return svg}
arrows.forEach((a,i)=>{const l=document.createElement('label');
const c=document.createElement('input');c.type='checkbox';c.value=1<<i;
l.append(c,icon(i));el('required').append(l)});
function render(){
const required=[...document.querySelectorAll('input:checked')].reduce((a,b)=>a|+b.value,0);
const family=el('family').value,weight=el('weight').value,sort=el('sort').value;
const rows=data.filter(r=>r[0]===+el('condition').value&&r[1]===+el('events').value&&
(weight==='all'||String(r[3])===weight)&&(r[2]&required)===required&&
(family==='all'||family==='d'&&r[2]<16||family==='o'&&(r[2]&15)===0||
family==='m'&&(r[2]&15)!==0&&(r[2]&240)!==0));
rows.sort((a,b)=>sort==='mask'?a[2]-b[2]:(b[5][+sort][0]??-1)-(a[5][+sort][0]??-1));
el('count').textContent=rows.length+' configurations';el('rows').replaceChildren();
for(const r of rows){const tr=document.createElement('tr'),label=document.createElement('td');
for(let i=0;i<8;i++)if(r[2]&(1<<i))label.append(icon(i));
if(r[3])label.append(document.createTextNode(' + LW'));tr.append(label);
for(const [mean,sd,median] of r[5]){const td=document.createElement('td');
td.textContent=mean===null?'—':(100*mean).toFixed(2);
td.title='Sample SD: '+(sd===null?'—':(100*sd).toFixed(2))+
' percentage points; median: '+(median===null?'—':(100*median).toFixed(2)+'%');
if(mean!==null)td.style.backgroundColor='hsl(180 35% '+(98-mean*27)+'%)';tr.append(td)}
el('rows').append(tr)}}
document.querySelectorAll('select,input').forEach(e=>e.addEventListener('change',render));render();
</script></html>"""
    (output / "explorer.html").write_text(html.replace("__DATA__", json.dumps(data)))
