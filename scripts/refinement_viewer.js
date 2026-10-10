const $ = id => document.getElementById(id);
const node = (tag, text, cls) => {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
};
let data, active, timer = null, selection = null, renderRevision = 0;
let descriptorPromise = null;
const frameCache = new Map();
const label = d => d === 'ref_youtube_vos' ? 'Ref-YT-VOS' : d === 'revos' ? 'ReVOS' : d;
const human = xs => (xs || []).map(s => s.replaceAll('_', ' ')).join(' · ');
const color = id => data.colors[id] || '#637069';
const unique = ids => [...new Set(ids)].sort((a, b) => a - b);
function pill(value) { return node('span', value, 'pill ' + value); }
function stop() { clearInterval(timer); timer = null; $('play').textContent = 'Play'; }
function currentFrames() {
  return $('evidence-stage').value === 'original'
    ? [active.original_images, active.original_frame_ids] : [active.images, active.frame_ids];
}
function availableIds() {
  return $('evidence-stage').value === 'original' ? active.original_ids : active.allowed_ids;
}
function selectionStyles() {
  document.querySelectorAll('.caption').forEach(e => e.classList.toggle('selected-caption', selection?.caption === e.id));
  document.querySelectorAll('[data-mask-ids]').forEach(e => {
    const selected = selection && e.dataset.caption === selection.caption
      && JSON.parse(e.dataset.maskIds).every(id => selection.ids.includes(id));
    e.setAttribute('aria-pressed', selected ? 'true' : 'false');
  });
}
function chooseMasks(captionId, stage, ids, title) {
  const frameId = currentFrames()[1][Number($('scrub').value)];
  selection = {caption: captionId, ids: unique(ids), title};
  $('evidence-stage').value = stage;
  const index = currentFrames()[1].indexOf(frameId);
  $('scrub').max = Math.max(0, currentFrames()[0].length - 1);
  $('scrub').value = Math.max(0, index);
  selectionStyles(); evidenceNote(); frame();
}
function resetMasks() {
  if (!selection) return;
  selection = null; selectionStyles(); frame();
}
function caption(target, text, links, stage, title, enabled = true) {
  target.replaceChildren(); target.onclick = null; target.onkeydown = null;
  target.removeAttribute('tabindex'); target.removeAttribute('aria-label');
  const interactive = enabled && !!active.mask_data;
  const chars = Array.from(text);
  const boundaries = unique([0, chars.length, ...links.flatMap(l => [l.start, l.end])])
    .filter(x => x >= 0 && x <= chars.length);
  for (let i = 0; i < boundaries.length - 1; i++) {
    const a = boundaries[i], b = boundaries[i + 1];
    const ids = unique(links.filter(l => l.start <= a && l.end >= b).flatMap(l => l.ids));
    const span = node(interactive && ids.length ? 'button' : 'span', chars.slice(a, b).join(''), ids.length ? 'linked' : '');
    if (ids.length) {
      span.style.setProperty('--color', color(ids[0]));
      span.title = 'Mask IDs: ' + ids.join(', ');
      if (interactive) {
        span.type = 'button'; span.dataset.maskIds = JSON.stringify(ids); span.dataset.caption = target.id;
        span.setAttribute('aria-label', span.textContent + ' — show masks ' + ids.join(', '));
        span.onclick = event => {
          event.stopPropagation(); chooseMasks(target.id, stage, ids, title + ' · “' + span.textContent + '”');
        };
      }
    }
    target.append(span);
  }
  if (interactive) {
    target.tabIndex = 0; target.setAttribute('aria-label', title + ': click to show all linked masks');
    target.onclick = event => {
      event.stopPropagation(); chooseMasks(target.id, stage, links.flatMap(l => l.ids), title);
    };
    target.onkeydown = event => {
      if (event.target === target && ['Enter', ' '].includes(event.key)) {
        event.preventDefault(); target.click();
      }
    };
  }
}
function chips(target, ids, main, captionId, stage) {
  target.replaceChildren();
  for (const id of ids) {
    const c = node(active.mask_data ? 'button' : 'span', '#' + id + (main.includes(id) ? ' · target' : ''), 'chip');
    c.style.setProperty('--color', color(id));
    if (active.mask_data) {
      c.type = 'button'; c.dataset.maskIds = JSON.stringify([id]); c.dataset.caption = captionId;
      c.setAttribute('aria-label', 'Show mask ' + id);
      c.onclick = event => { event.stopPropagation(); chooseMasks(captionId, stage, [id], 'Mask #' + id); };
    }
    target.append(c);
  }
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image);
    image.onerror = () => reject(Error('Unable to load mask evidence. Please reload.')); image.src = url;
  });
}
async function loadFrame(record) {
  const key = record.image + '|' + record.layers.map(l => l.image).join('|');
  if (!frameCache.has(key)) {
    const promise = (async () => {
      const image = await loadImage(record.image);
      const canvas = document.createElement('canvas'); canvas.width = record.width; canvas.height = record.height;
      const context = canvas.getContext('2d', {willReadFrequently: true});
      context.drawImage(image, 0, 0);
      const base = context.getImageData(0, 0, record.width, record.height);
      const layers = [];
      for (const layer of record.layers) {
        const atlas = await loadImage(layer.image);
        canvas.width = record.width * 2;
        context.drawImage(atlas, 0, 0);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        const fill = new Uint32Array(record.width * record.height), outline = new Uint32Array(fill.length);
        for (let y = 0; y < record.height; y++) for (let x = 0; x < record.width; x++) {
          const i = y * record.width + x, p = (y * canvas.width + x) * 4, q = p + record.width * 4;
          fill[i] = pixels[p] | pixels[p + 1] << 8 | pixels[p + 2] << 16;
          outline[i] = pixels[q] | pixels[q + 1] << 8 | pixels[q + 2] << 16;
        }
        layers.push({ids: layer.ids, fill, outline});
      }
      return {image, base, layers};
    })();
    frameCache.set(key, promise);
    promise.catch(() => frameCache.delete(key));
    // Bound decoded image/geometry memory during long review sessions.
    while (frameCache.size > 3) frameCache.delete(frameCache.keys().next().value);
  }
  return frameCache.get(key);
}
function paintMasks(canvas, record, loaded, ids) {
  const selected = new Set(ids), pixels = new Uint8ClampedArray(loaded.base.data);
  const palette = id => color(id).match(/\d+/g).map(Number);
  const layers = loaded.layers.map(layer => ({...layer,
    selected: layer.ids.reduce((bits, id, bit) => selected.has(id) ? bits | (1 << bit) : bits, 0),
    colors: layer.ids.map(palette)}));
  for (const layer of layers) {
    if (!layer.selected) continue;
    for (let i = 0; i < layer.fill.length; i++) {
      let bits = layer.fill[i] & layer.selected;
      while (bits) {
        const bit = 31 - Math.clz32(bits & -bits), rgb = layer.colors[bit], p = i * 4;
        for (let c = 0; c < 3; c++) pixels[p + c] = Math.floor(.45 * pixels[p + c] + .55 * rgb[c]);
        bits &= bits - 1;
      }
    }
  }
  for (const layer of layers) {
    if (!layer.selected) continue;
    for (let i = 0; i < layer.outline.length; i++) {
      const bits = layer.outline[i] & layer.selected;
      if (bits) pixels.set(layer.colors[31 - Math.clz32(bits)], i * 4);
    }
  }
  canvas.width = record.width * 2; canvas.height = record.height;
  const context = canvas.getContext('2d');
  context.drawImage(loaded.image, 0, 0);
  context.putImageData(new ImageData(pixels, record.width, record.height), record.width, 0);
  context.font = 'bold 18px system-ui'; context.lineJoin = 'round'; context.lineWidth = 4;
  context.strokeStyle = '#000'; context.fillStyle = '#fff';
  const visible = [];
  for (const id of ids) {
    const position = record.labels[id];
    if (!position) continue;
    visible.push(id);
    context.strokeText(String(id), record.width + position[0], position[1]);
    context.fillText(String(id), record.width + position[0], position[1]);
  }
  canvas.dataset.maskIds = JSON.stringify(ids); canvas.dataset.visibleIds = JSON.stringify(visible);
  return visible;
}
async function frame() {
  if (!active) return;
  const revision = ++renderRevision, sample = active, stage = $('evidence-stage').value;
  const [images, frameIds] = currentFrames();
  $('scrub').max = Math.max(0, images.length - 1);
  const index = Math.min(Number($('scrub').value), Math.max(0, images.length - 1));
  $('scrub').value = index; $('play').disabled = images.length < 2; $('scrub').disabled = images.length < 2;
  $('frame').classList.add('hidden'); $('mask-frame').classList.toggle('hidden', !sample.mask_data || !images.length);
  $('mask-controls').classList.toggle('hidden', !sample.mask_data);
  if (!images.length) { $('frame-label').textContent = 'No visual input: ' + sample.stage; return; }
  $('frame-label').textContent = (index + 1) + ' / ' + images.length + ' · ' + frameIds[index];
  if (!sample.mask_data) { $('frame').src = images[index]; $('frame').classList.remove('hidden'); return; }
  const chosen = selection ? selection.ids : availableIds(), title = selection?.title;
  $('mask-status').textContent = 'Loading masks…';
  $('mask-frame').style.opacity = '.35'; $('mask-frame').setAttribute('aria-busy', 'true');
  try {
    const descriptor = await descriptorPromise, record = descriptor[stage][index];
    const loaded = await loadFrame(record);
    if (revision !== renderRevision || sample !== active) return;
    const visible = paintMasks($('mask-frame'), record, loaded, chosen);
    $('mask-frame').dataset.frameId = frameIds[index]; $('mask-frame').dataset.stage = stage;
    $('mask-frame').style.opacity = '1'; $('mask-frame').setAttribute('aria-busy', 'false');
    $('mask-status').textContent = (title ? title + ' · ' + chosen.length + ' linked masks' : 'All ' + chosen.length + ' masks for this view')
      + ' · ' + visible.length + ' visible in this frame';
  } catch (error) {
    if (revision !== renderRevision) return;
    $('mask-frame').classList.add('hidden'); $('mask-frame').setAttribute('aria-busy', 'false');
    $('mask-status').textContent = error.message; stop();
  }
}
function evidenceNote() {
  const original = $('evidence-stage').value === 'original';
  $('evidence').textContent = original
    ? 'Before NLP: original candidate masks for audit. No model call uses this extra view.'
    : 'After NLP: the sampled frames and permitted masks from the visual pass. IDs match the caption links below.';
  if (active.mask_data) $('evidence').textContent += ' Interactive overlays use the source annotations at display resolution; the original model-input JPEGs remain in the run checkpoints.';
  else if (data.image_encoding?.exact_jpeg === false) $('evidence').textContent += ' Display uses WebP quality ' + data.image_encoding.quality + ' copies. Original JPEG inputs remain in the run checkpoints.';
}
function evidence() {
  stop(); selection = null; selectionStyles(); $('scrub').value = 0; evidenceNote(); frame();
}
function select(s) {
  stop(); active = s; selection = null; frameCache.clear();
  descriptorPromise = s.mask_data ? fetch(s.mask_data).then(r => { if (!r.ok) throw Error('Unable to load mask data'); return r.json(); }) : null;
  location.hash = encodeURIComponent(s.sample_id);
  $('detail').classList.remove('hidden'); $('empty').classList.add('hidden');
  document.querySelectorAll('.sample').forEach(e => e.classList.toggle('active', e.dataset.sid === s.sample_id));
  $('featured-banner').classList.toggle('hidden', !s.featured);
  $('sample-title').replaceChildren(node('span', label(s.dataset) + ' / ' + s.video_id + ' '), pill(s.decision));
  $('sample-meta').textContent = s.split + ' · ' + s.density_bin + ' candidate density · ' + s.sample_id;
  $('evidence-stage').value = s.images.length ? 'visual' : 'original';
  caption($('before'), s.original_text, s.original_links, 'original', 'Original caption');
  chips($('before-tracks'), s.original_ids, s.main_ids, 'before', 'original');
  $('before-count').textContent = s.original_tracks + ' source tracks';
  $('offset-note').textContent = s.source_offsets_invalid ? 'Some original offsets disagree with their stored phrases; highlights locate those phrases for display. Source data is preserved.' : '';
  caption($('nlp'), s.nlp_text || (s.nlp_decision === 'error' ? 'Language output failed validation.' : 'Not run: ' + s.stage.replaceAll('_', ' ')), s.nlp_links, 'visual', 'NLP caption', !!s.nlp_text);
  chips($('nlp-tracks'), unique(s.nlp_links.flatMap(l => l.ids)), s.main_ids, 'nlp', 'visual');
  $('nlp-count').textContent = s.nlp_tracks === null ? 'No validated NLP result' : s.nlp_tracks + ' tracks remain · ' + (s.original_tracks - s.nlp_tracks) + ' removed';
  $('nlp-reasons').textContent = human(s.nlp_reasons);
  $('removed-links').replaceChildren();
  const removed = new Map();
  for (const g of s.nlp_removed) { const k = g.identity + ' → ' + g.phrases.join(', '); removed.set(k, (removed.get(k) || 0) + g.track_count); }
  for (const [k, n] of removed) $('removed-links').append(node('p', k + ' · ' + n + ' track(s)'));
  $('removed-detail').classList.toggle('hidden', !removed.size);
  caption($('after'), s.refined_text || (s.decision === 'reject' ? 'Rejected from the refined export.' : 'No validated final result.'), s.refined_links, 'visual', 'Final caption', !!s.refined_text);
  chips($('after-tracks'), s.retained_ids, s.main_ids, 'after', 'visual');
  $('after-count').textContent = s.retained_ids.length + ' retained visual tracks';
  $('reasons').textContent = human(s.reason_codes);
  $('invalid-proposal').classList.toggle('hidden', !s.invalid_proposal); $('proposal-text').textContent = s.invalid_proposal || '';
  $('permissions').textContent = 'Visual-pass permitted IDs: ' + (s.allowed_ids.map(i => '#' + i).join(', ') || 'none') + '. '
    + (s.negative ? 'Intentional negative.' : 'Intended target IDs: ' + s.main_ids.map(i => '#' + i).join(', ') + '.');
  evidence();
}
function button(s) {
  const b = node('button', undefined, 'sample'); b.dataset.sid = s.sample_id;
  b.append(pill(s.decision), node('small', label(s.dataset) + ' · ' + s.density_bin + ' candidates'), node('span', s.video_id), node('span', s.original_text, 'snippet'));
  b.onclick = () => select(s); return b;
}
function filter() {
  const q = $('search').value.toLowerCase(), nlp = $('nlp-filter').value;
  const matches = data.samples.filter(s => (!$('dataset').value || s.dataset === $('dataset').value)
    && (!$('density').value || s.density_bin === $('density').value)
    && (!$('decision').value || s.decision === $('decision').value)
    && (!nlp || (nlp === 'removed' ? s.nlp_tracks !== null && s.nlp_tracks < s.original_tracks : s.nlp_decision === nlp))
    && (!q || (s.original_text + ' ' + s.refined_text + ' ' + s.sample_id + ' ' + human(s.nlp_reasons) + ' ' + human(s.reason_codes)).toLowerCase().includes(q)));
  $('list').replaceChildren(...matches.filter(s => !s.featured).map(button));
  $('list-count').textContent = matches.length + ' matching examples · featured stays pinned';
  if (matches.length) select(matches.find(s => active && s.sample_id === active.sample_id) || matches[0]);
  else { stop(); ++renderRevision; $('detail').classList.add('hidden'); $('empty').classList.remove('hidden'); }
}
$('evidence-stage').onchange = evidence;
$('scrub').oninput = () => { stop(); frame(); };
$('play').onclick = () => {
  if (timer) { stop(); return; }
  if (currentFrames()[0].length < 2) return;
  $('play').textContent = 'Pause';
  timer = setInterval(() => { $('scrub').value = (Number($('scrub').value) + 1) % currentFrames()[0].length; frame(); }, 850);
};
document.addEventListener('click', resetMasks);
document.addEventListener('keydown', event => { if (event.key === 'Escape') resetMasks(); });
for (const id of ['search', 'dataset', 'density', 'decision', 'nlp-filter']) $(id).oninput = filter;
fetch('data.json').then(r => { if (!r.ok) throw Error('Unable to load review data'); return r.json(); }).then(value => {
  data = value;
  const videoCount = new Set(data.samples.map(s => [s.dataset, s.split, s.video_id].join('/'))).size;
  for (const [name, count] of [['Videos', videoCount], ['Expressions', data.samples.length], ...Object.entries(data.summary.counts)]) {
    const stat = node('div', undefined, 'stat'); stat.append(node('b', String(count)), node('span', name)); $('stats').append(stat);
  }
  $('stats').append(node('div', 'API cost ≤ $' + Number(data.summary.cost_upper_usd).toFixed(4), 'stat'));
  for (const [id, values] of [['dataset', [...new Set(data.samples.map(s => s.dataset))]], ['density', ['0–2', '3–5', '6–12', '13+'].filter(v => data.samples.some(s => s.density_bin === v))]]) {
    for (const v of values) { const o = node('option', id === 'dataset' ? label(v) : v + ' candidates'); o.value = v; $(id).append(o); }
  }
  for (const s of data.samples.filter(s => s.featured)) $('featured-list').append(button(s));
  $('featured').classList.toggle('hidden', !data.samples.some(s => s.featured));
  for (const p of ['data.json', ...data.downloads]) { const a = node('a', p); a.href = p; a.download = ''; $('downloads').append(a); }
  $('sampling').textContent = (data.selection.density_measure || 'Unstratified pilot.') + ' ' + (data.selection.media_eligibility || '')
    + (data.selection.videos_without_local_source ? ' ' + data.selection.videos_without_local_source + ' videos lacked a local source and were excluded before sampling.' : '')
    + (data.selection.videos.length ? ' Selection is deterministic: ' + Object.entries(data.summary.videos_by_dataset || {}).map(([k, v]) => label(k) + ' ' + v + ' videos').join(', ')
      + '. Equal coverage of four candidate-density ranges; the featured video counts within its quota. The densest source expression represents each video. This is a stress sample, not a prevalence estimate.' : '');
  let sid = ''; try { sid = decodeURIComponent(location.hash.slice(1)); } catch {}
  active = data.samples.find(s => s.sample_id === sid); filter();
}).catch(error => { $('empty').textContent = error.message; $('empty').classList.remove('hidden'); $('detail').classList.add('hidden'); });
