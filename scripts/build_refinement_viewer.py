#!/usr/bin/env python3
"""Build a static, credential-free viewer from an existing refinement run."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from refine_hf import Source, apply_result, color_for, digest, discover_inputs, loads


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ConCor · Video refinement pilot</title><style>
:root{color-scheme:light;--bg:#f5f5f0;--ink:#202d28;--muted:#637069;--line:#dce1d8;--green:#176044}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}button,input,select{font:inherit}button,select{cursor:pointer}a{color:var(--green)}header{padding:26px 32px 20px;border-bottom:1px solid var(--line);background:#fff}header p{margin:6px 0;color:var(--muted)}h1{font-size:27px;letter-spacing:-.6px;margin:0}h2{font-size:16px;margin:0 0 10px}h3{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 12px}.eyebrow{font-size:12px;letter-spacing:.12em;color:var(--green);font-weight:650}.stats{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}.stat{border:1px solid var(--line);padding:8px 14px;border-radius:7px;font-variant-numeric:tabular-nums}.stat b{margin-right:6px}.layout{display:grid;grid-template-columns:300px minmax(0,1fr);max-width:1800px;margin:auto}aside{padding:20px;border-right:1px solid var(--line);max-height:calc(100vh - 190px);overflow:auto;position:sticky;top:0}main{padding:24px;min-width:0}.filters{display:grid;gap:8px;margin-bottom:16px}input[type=search],select{width:100%;background:white;border:1px solid var(--line);border-radius:6px;padding:9px}.sample{display:block;text-align:left;width:100%;padding:13px;border:1px solid transparent;background:transparent;border-radius:7px;margin:3px 0;color:var(--ink)}.sample:hover{background:#e9eee6}.sample.active{border-color:#93b6a2;background:#e5efe6}.sample small{display:block;color:var(--muted);margin-top:5px}.snippet{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.pill{display:inline-block;border-radius:4px;font-size:11px;font-weight:650;padding:2px 7px;background:#e1ebe1;text-transform:uppercase;letter-spacing:.04em}.reject,.error{background:#fae6df;color:#923e25}.rewrite{background:#e5edf7;color:#365a89}.pending{background:#ede9de;color:#786338}.video{background:#101713;border-radius:9px;overflow:hidden}.video img{display:block;width:100%;height:auto;min-height:120px;object-fit:contain}.controls{display:flex;align-items:center;gap:14px;color:#e5ebe6;padding:13px 16px}.controls input{flex:1;min-width:40px;accent-color:#8ec69e}.controls button{background:#2c4135;border:0;color:white;border-radius:5px;padding:6px 14px}.controls span{font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}.meta{color:var(--muted);font-size:12px;overflow-wrap:anywhere;margin:6px 0 15px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}.card{background:white;border:1px solid var(--line);border-radius:8px;padding:20px}.caption{font-size:19px;line-height:1.85;margin:0 0 16px;white-space:pre-wrap}.linked{background:color-mix(in srgb,var(--color) 19%,transparent);border-bottom:2px solid var(--color);border-radius:2px;padding:1px 0}.chip{display:inline-block;padding:2px 7px;margin:3px;font-size:12px;border:1px solid var(--line);border-left:4px solid var(--color);border-radius:4px}.muted{color:var(--muted)}.reasons{margin-top:16px;font-size:13px}.foot{margin:25px 0 0;font-size:12px;color:var(--muted)}.downloads{display:flex;gap:15px;flex-wrap:wrap;margin-top:10px}.hidden{display:none}@media(max-width:850px){header{padding:20px}.layout{display:block}aside{position:static;max-height:290px;border-right:0;border-bottom:1px solid var(--line)}main{padding:16px}.columns{grid-template-columns:1fr}.filters{grid-template-columns:1fr 1fr}#search{grid-column:1/-1}h1{font-size:24px}.controls{gap:8px}}
</style></head><body>
<header><div class="eyebrow">CONCOR / VIDEO DATA</div><h1>Refinement pilot</h1><p>Original expressions, revised correspondences, and the visual evidence used for each decision.</p><div id="stats" class="stats"></div></header>
<div class="layout"><aside><div class="filters"><input id="search" type="search" placeholder="Search expressions or sample IDs" aria-label="Search expressions"><select id="dataset" aria-label="Dataset"><option value="">All datasets</option></select><select id="decision" aria-label="Decision"><option value="">All decisions</option><option>rewrite</option><option>keep</option><option>reject</option><option>error</option><option>pending</option></select></div><div id="list"></div></aside>
<main><div id="empty" class="card hidden">No matching expressions.</div><article id="detail"><h2 id="sample-title"></h2><p id="sample-meta" class="meta"></p><div class="video"><img id="frame" alt="Original frame on the left and numbered candidate tracklet overlays on the right"><div class="controls"><button id="play">Play</button><input id="scrub" type="range" min="0" value="0" aria-label="Selected video frame"><span id="frame-label"></span></div></div><p id="evidence" class="meta"></p><div class="columns"><section class="card"><h3>Original expression</h3><p id="before" class="caption"></p><div id="before-tracks"></div></section><section class="card"><h3 id="after-heading">Refined expression</h3><p id="after" class="caption"></p><div id="after-tracks"></div><div id="reasons" class="reasons muted"></div></section></div><p id="permissions" class="meta"></p></article>
<div class="foot">GPT-6 Luna · medium reasoning · one shared call per video group. This pilot is for inspection; passing structural checks does not establish semantic correctness. Images are sampled evidence, not every source frame.<div class="downloads" id="downloads"></div><p>Visual presentation adapted from <a href="https://github.com/Weiyang-Huang/VIPOSeg-Grounded-Caption">VIPOSeg</a>. Rules: <a href="https://github.com/suryathecreator/ConCor-Training-Data-Generation">ConCor BCC</a>. Source: <a href="https://huggingface.co/datasets/suryadv/concor-video-data-filtered">HF annotations</a>. <a href="https://github.com/suryathecreator/ConCor-Video-Data-Processing">Wrapper code</a>. Media retains Ref-YouTube-VOS / ReVOS upstream terms; ReVOS is CC BY-NC-SA 4.0.</p></div></main></div>
<script>
'use strict';
const $=id=>document.getElementById(id);let data,active,timer=null;
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
function pill(value){return node('span',value,'pill '+value)}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='Play'}
function color(id){return data.colors[id]||'#637069'}
function caption(target,text,links){target.replaceChildren();const chars=Array.from(text);const boundaries=[...new Set([0,chars.length,...links.flatMap(l=>[l.start,l.end])])].sort((a,b)=>a-b);for(let i=0;i<boundaries.length-1;i++){const a=boundaries[i],b=boundaries[i+1],ids=[...new Set(links.filter(l=>l.start<=a&&l.end>=b).flatMap(l=>l.ids))];const span=node('span',chars.slice(a,b).join(''),ids.length?'linked':'');if(ids.length){span.style.setProperty('--color',color(ids[0]));span.title='Tracklet '+ids.join(', ')}target.append(span)}}
function chips(target,ids,main){target.replaceChildren();if(!ids.length)target.append(node('span','No retained tracks','muted'));for(const id of ids){const chip=node('span','#'+id+(main.includes(id)?' · target':''),'chip');chip.style.setProperty('--color',color(id));target.append(chip)}}
function frame(){if(!active)return;const i=Number($('scrub').value);if(!active.images.length){$('frame').removeAttribute('src');$('frame-label').textContent='No visual packet';return}$('frame').src=active.images[i];$('frame-label').textContent=(i+1)+' / '+active.images.length+' · '+active.frame_ids[i]}
function select(sample){stop();active=sample;$('detail').classList.remove('hidden');$('empty').classList.add('hidden');for(const el of $('list').children)el.classList.toggle('active',el.dataset.sid===sample.sample_id);$('sample-title').replaceChildren(node('span',sample.dataset+' / '+sample.split+' '),pill(sample.decision));$('sample-meta').textContent=sample.sample_id;$('scrub').max=Math.max(0,sample.images.length-1);$('scrub').value=0;frame();$('evidence').textContent='Left: original · Right: candidate masks · '+sample.images.length+' frames shared across '+sample.group_size+' expression(s).';caption($('before'),sample.original_text,sample.original_links);chips($('before-tracks'),sample.allowed_ids,sample.main_ids);$('after-heading').textContent=sample.decision==='reject'?'Rejected sample':'Refined expression';caption($('after'),sample.refined_text||(sample.decision==='reject'?'Removed from the refined export.':'No validated result available.'),sample.refined_links);chips($('after-tracks'),sample.retained_ids,sample.main_ids);$('reasons').textContent=sample.reason_codes.map(s=>s.replaceAll('_',' ')).join(' · ');$('permissions').textContent='This expression may use '+(sample.allowed_ids.map(x=>'#'+x).join(', ')||'no tracklets')+'. Intended target: '+(sample.main_ids.map(x=>'#'+x).join(', ')||(sample.negative?'intentional negative':'unavailable'))+'. All retained tracklets keep their full source mask sequence.'}
function filter(){stop();const q=$('search').value.toLowerCase();const samples=data.samples.filter(s=>(!$('dataset').value||s.dataset===$('dataset').value)&&(!$('decision').value||s.decision===$('decision').value)&&(!q||(s.original_text+' '+s.refined_text+' '+s.sample_id).toLowerCase().includes(q)));$('list').replaceChildren();for(const s of samples){const b=node('button',undefined,'sample');b.dataset.sid=s.sample_id;b.append(pill(s.decision),node('small',s.dataset+' / '+s.split),node('span',s.original_text,'snippet'));b.onclick=()=>select(s);$('list').append(b)}if(samples.length)select(samples.find(s=>active&&s.sample_id===active.sample_id)||samples[0]);else{$('detail').classList.add('hidden');$('empty').classList.remove('hidden')}}
$('scrub').oninput=()=>{stop();frame()};$('play').onclick=()=>{if(timer){stop();return}if(!active||active.images.length<2)return;$('play').textContent='Pause';timer=setInterval(()=>{$('scrub').value=(Number($('scrub').value)+1)%active.images.length;frame()},850)};for(const id of ['search','dataset','decision'])$(id).oninput=filter;
fetch('data.json').then(r=>{if(!r.ok)throw Error('Unable to load pilot data');return r.json()}).then(value=>{data=value;for(const [label,count] of [['Expressions',data.samples.length],...Object.entries(data.summary.counts)]){const item=node('div',undefined,'stat');item.append(node('b',String(count)),node('span',label));$('stats').append(item)}$('stats').append(node('div','API cost ≤ $'+Number(data.summary.cost_upper_usd).toFixed(4),'stat'));for(const dataset of [...new Set(data.samples.map(s=>s.dataset))]){const option=node('option',dataset);option.value=dataset;$('dataset').append(option)}for(const path of ['data.json',...data.downloads]){const link=node('a',path);link.href=path;link.download='';$('downloads').append(link)}filter()}).catch(error=>{$('empty').textContent=error.message;$('empty').classList.remove('hidden');$('detail').classList.add('hidden')});
</script></body></html>'''


def build(input_path, run, destination):
    source = Source(discover_inputs(input_path))
    configuration = loads((run / "run.json").read_text())
    summary = loads((run / "summary.json").read_text())
    early = loads((run / "heuristics.json").read_text())
    manifests = [loads((run / "packets" / rid / "input.json").read_text())
                 for rid in loads((run / "requests.json").read_text())]
    by_sample = {e["sample_id"]: (m, e) for m in manifests for e in m["packet"]["expressions"]}
    destination.mkdir(parents=True, exist_ok=True)
    samples, colors = [], {}
    for sid in configuration["selected_ids"]:
        row = source.row(sid)
        path = run / "decisions" / (digest(sid) + ".json")
        result = loads(path.read_text()) if path.exists() else early.get(sid, {"decision": "pending", "reason_codes": []})
        manifest, entry = by_sample.get(sid, ({}, {}))
        mapping = manifest.get("mapping", {}).get(sid, {})
        local_to_visual = {tid: int(vid) for vid, tid in mapping.get("visual_to_local", {}).items()}
        images, frame_ids = [], []
        for relative, index in zip(manifest.get("images", []), manifest.get("packet", {}).get("selected_frame_indices", [])):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(run / relative, target)
            images.append(relative)
            frame_ids.append(str(manifest["packet"]["frame_ids"][index]))
        original_links = [{**span, "ids": group["ids"]} for group in entry.get("existing_groups", [])
                          for span in group["text_spans"] if group["ids"]]
        refined_links, refined_text, retained = [], "", []
        if result["decision"] in {"keep", "rewrite"}:
            rewritten = apply_result(row, result, mapping)
            refined_text = rewritten["text"]
            refined_links = [{"start": link["start"], "end": link["end"],
                              "ids": [local_to_visual[t] for t in link["tracklet_ids"]]}
                             for link in loads(rewritten["span_links_json"])]
            retained = sorted({tid for link in refined_links for tid in link["ids"]})
        for vid in entry.get("allowed_ids", []):
            b, g, r = color_for(vid)
            colors[str(vid)] = f"rgb({r},{g},{b})"
        samples.append({"sample_id": sid, "dataset": row["dataset"], "split": row["split"],
                        "video_id": row["video_id"], "negative": row["negative"],
                        "decision": result["decision"], "reason_codes": result.get("reason_codes", []),
                        "original_text": row["text"], "original_links": original_links,
                        "refined_text": refined_text, "refined_links": refined_links,
                        "allowed_ids": entry.get("allowed_ids", []), "main_ids": entry.get("main_ids", []),
                        "retained_ids": retained, "images": images, "frame_ids": frame_ids,
                        "group_size": len(manifest.get("mapping", {}))})
    downloads = []
    for path in sorted(run.glob("*.parquet")):
        shutil.copyfile(path, destination / path.name)
        downloads.append(path.name)
    (destination / "index.html").write_text(HTML, encoding="utf-8")
    (destination / "data.json").write_text(json.dumps({"summary": summary, "colors": colors, "samples": samples,
                                                     "downloads": downloads}, ensure_ascii=False), encoding="utf-8")
    print(f"Built viewer for {len(samples)} expressions at {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.input, args.run.resolve(), args.output.resolve())
