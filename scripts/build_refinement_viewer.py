#!/usr/bin/env python3
"""Build a credential-free three-stage viewer from a bounded refinement run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2

from refine_hf import Source, apply_result, color_for, digest, discover_inputs, loads, release_memory


def compact_images(destination, data, quality):
    """Encode display copies only; original model-input JPEGs stay in the run."""
    if not 0 <= quality <= 100:
        raise ValueError('WebP quality must be 0..100')
    cv2.setNumThreads(1)
    paths = sorted({path for sample in data['samples']
                    for field in ('images', 'original_images') for path in sample[field]})

    def encode(relative):
        frame = cv2.imread(str(destination / relative))
        if frame is None:
            raise ValueError('Unreadable viewer image: ' + relative)
        ok, encoded = cv2.imencode('.webp', frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
        if not ok:
            raise ValueError('WebP encoding failed: ' + relative)
        payload = encoded.tobytes()
        target = destination / 'assets' / (hashlib.sha256(payload).hexdigest() + '.webp')
        target.write_bytes(payload)
        return relative, str(target.relative_to(destination))

    with ThreadPoolExecutor(max_workers=4) as pool:
        mapping = dict(pool.map(encode, paths))
    for sample in data['samples']:
        for field in ('images', 'original_images'):
            sample[field] = [mapping[path] for path in sample[field]]
    data['image_encoding'] = {'format': 'webp', 'quality': quality, 'exact_jpeg': False}


def write_viewer(destination, data):
    (destination / 'index.html').write_text(Path(__file__).with_name('refinement_viewer.html').read_text(), encoding='utf-8')
    (destination / 'data.json').write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    used = {path for sample in data['samples']
            for field in ('images', 'original_images') for path in sample[field]}
    # Remove only this builder's unused content-addressed display assets.
    for path in (destination / 'assets').glob('*'):
        if re.fullmatch(r'[0-9a-f]{64}\.(jpg|webp)', path.name) and str(path.relative_to(destination)) not in used:
            path.unlink()


def manifests_by_sample(directory):
    path = directory / 'requests.json'
    if not path.exists():
        return {}
    manifests = [loads((directory / 'packets' / rid / 'input.json').read_text()) for rid in loads(path.read_text())]
    return {e['sample_id']: (m, e) for m in manifests for e in m['packet']['expressions']}


def links_for(row, local_to_visual):
    links, corrected = [], False
    for group in loads(row['groups_json'], []):
        ids = sorted({local_to_visual[t] for t in group.get('tracklet_ids', []) if t in local_to_visual})
        for span in group.get('text_spans', []):
            a, b, phrase = span['start'], span['end'], span['text']
            if row['text'][a:b] != phrase:
                # Preserve bad source offsets in the input snapshot; locate their
                # recorded literal phrase solely for readable audit highlighting.
                matches = [m.start() for m in re.finditer(re.escape(phrase), row['text'])] if phrase else []
                if not matches:
                    continue
                a = min(matches, key=lambda index: abs(index - a))
                b, corrected = a + len(phrase), True
            if ids:
                links.append({'start': a, 'end': b, 'ids': ids})
    return links, corrected


def build(input_path, run, destination, webp_quality=None):
    snapshot = run / 'input/selected.parquet'
    source = Source([snapshot] if snapshot.exists() else discover_inputs(input_path))
    config = loads((run / 'run.json').read_text())
    summary = loads((run / 'summary.json').read_text())
    early = loads((run / 'heuristics.json').read_text())
    after = manifests_by_sample(run)
    before = manifests_by_sample(run / 'preflight') or after
    selection = loads((run / 'selection.json').read_text()) if (run / 'selection.json').exists() else {'videos': []}
    selected = {sid: video for video in selection['videos'] for sid in video['sample_ids']}
    destination.mkdir(parents=True, exist_ok=True)
    samples, colors = [], {}

    def copy_images(manifest, base, prefix=''):
        images, frame_ids = [], []
        for relative, index in zip(manifest.get('images', []), manifest.get('packet', {}).get('selected_frame_indices', [])):
            payload = (base / relative).read_bytes()
            # Identical before/after evidence shares one immutable static asset.
            target = destination / 'assets' / (hashlib.sha256(payload).hexdigest() + '.jpg')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(payload)
            images.append(str(target.relative_to(destination)))
            frame_ids.append(str(manifest['packet']['frame_ids'][index]))
        return images, frame_ids

    for sid in config['selected_ids']:
        original = source.row(sid)
        decision_path = run / 'decisions' / (digest(sid) + '.json')
        result = loads(decision_path.read_text()) if decision_path.exists() else early.get(sid, {'decision': 'pending', 'reason_codes': []})
        old_manifest, old_entry = before.get(sid, ({}, {}))
        manifest, entry = after.get(sid, ({}, {}))
        old_mapping = old_manifest.get('mapping', {}).get(sid, {})
        mapping = manifest.get('mapping', {}).get(sid, {})
        old_ids = {tid: int(vid) for vid, tid in old_mapping.get('visual_to_local', {}).items()}
        old_ids.update({tid: int(vid) for tid, vid in old_mapping.get('local_to_visual', {}).items()})
        new_ids = {tid: int(vid) for vid, tid in mapping.get('visual_to_local', {}).items()}
        images, frame_ids = copy_images(manifest, run)
        original_images, original_frame_ids = copy_images(old_manifest, run / 'preflight' if (run / 'preflight/requests.json').exists() else run, 'original')
        original_links, offset_issue = links_for(original, old_ids)
        nlp_path = run / 'nlp/decisions' / (digest(sid) + '.json')
        nlp = loads(nlp_path.read_text()) if nlp_path.exists() else None
        cleaned = apply_result(original, nlp, nlp['mapping']) if nlp and nlp['decision'] in {'keep', 'rewrite'} else None
        nlp_links = links_for(cleaned, old_ids)[0] if cleaned else []
        refined_text, refined_links, retained = '', [], []
        if result['decision'] in {'keep', 'rewrite'}:
            rewritten = apply_result(cleaned or original, result, mapping)
            refined_text = rewritten['text']
            refined_links = links_for(rewritten, new_ids)[0]
            retained = sorted({tid for link in refined_links for tid in link['ids']})
        for vid in set(old_ids.values()) | set(new_ids.values()):
            b, g, r = color_for(vid)
            colors[str(vid)] = f'rgb({r},{g},{b})'
        raw_tracks = loads(original['tracklets_json'], [])
        clean_tracks = loads(cleaned['tracklets_json'], []) if cleaned else []
        raw_ids = {t['tracklet_id'] for t in raw_tracks}
        clean_ids = {t['tracklet_id'] for t in clean_tracks}
        removed = [{'identity': g.get('identity', ''), 'phrases': [s['text'] for s in g.get('text_spans', [])],
                    'track_count': len(set(g.get('tracklet_ids', [])) & (raw_ids - clean_ids))}
                   for g in loads(original['groups_json'], []) if cleaned and set(g.get('tracklet_ids', [])) & (raw_ids - clean_ids)]
        record = selected.get(sid, {})
        samples.append({'sample_id': sid, 'dataset': original['dataset'], 'split': original['split'],
                        'video_id': original['video_id'], 'negative': original['negative'],
                        'featured': record.get('featured', False), 'density_bin': record.get('density_bin', 'Unstratified'),
                        'candidate_tracks': record.get('candidate_tracks', len(raw_tracks)),
                        'decision': result['decision'], 'stage': result.get('stage', 'model'),
                        'reason_codes': result.get('reason_codes', []),
                        'invalid_proposal': result.get('model_result', {}).get('tagged_caption', ''),
                        'original_text': original['text'], 'original_links': original_links,
                        'source_offsets_invalid': offset_issue, 'original_tracks': len(raw_tracks),
                        'nlp_text': cleaned['text'] if cleaned else '', 'nlp_links': nlp_links,
                        'nlp_decision': nlp['decision'] if nlp else ('not_run' if (run / 'nlp').exists() else 'not_enabled'),
                        'nlp_reasons': nlp.get('reason_codes', []) if nlp else [],
                        'nlp_tracks': len(clean_tracks) if cleaned else None, 'nlp_removed': removed,
                        'refined_text': refined_text, 'refined_links': refined_links,
                        'allowed_ids': entry.get('allowed_ids', []), 'main_ids': entry.get('main_ids', old_entry.get('main_ids', [])),
                        'original_ids': old_entry.get('allowed_ids', []), 'retained_ids': retained,
                        'images': images, 'frame_ids': frame_ids, 'original_images': original_images,
                        'original_frame_ids': original_frame_ids, 'group_size': len(manifest.get('mapping', {}))})
    samples.sort(key=lambda s: not s['featured'])
    downloads = []
    for path in sorted(run.glob('*.parquet')) + [run / n for n in ['selection.json', 'summary.json', 'audit.json', 'preflight.json', 'verification.json'] if (run / n).exists()]:
        shutil.copyfile(path, destination / path.name)
        downloads.append(path.name)
    data = {'summary': summary, 'selection': selection, 'colors': colors, 'samples': samples,
            'downloads': downloads, 'image_encoding': {'format': 'jpeg', 'quality': 88, 'exact_jpeg': True}}
    if webp_quality is not None:
        release_memory()
        compact_images(destination, data, webp_quality)
    write_viewer(destination, data)
    print(f'Built viewer for {len(samples)} expressions at {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--webp-quality', type=int, help='Optional compact display copies, 0..100; leaves model inputs unchanged')
    args = parser.parse_args()
    if args.webp_quality is not None and not 0 <= args.webp_quality <= 100:
        parser.error('--webp-quality must be 0..100')
    build(args.input, args.run.resolve(), args.output.resolve(), args.webp_quality)
