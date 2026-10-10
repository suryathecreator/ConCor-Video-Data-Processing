"""Lossless independent mask layers for the static review viewer (no API calls)."""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

from concor_video.rle import decode_rle


def asset(destination, payload, suffix):
    relative = 'assets/' + hashlib.sha256(payload).hexdigest() + suffix
    path = destination / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    return relative


def pack_masks(masks):
    """Pack 24 independent masks per lossless RGB atlas, including overlaps.

    Left half stores fills, right half stores outlines. Browser RGB channel
    bits 0..7, 8..15, 16..23 map to the accompanying ordered IDs. No alpha or
    lossy image encoding is used for the geometry.
    """
    items = sorted(masks.items())
    for start in range(0, len(items), 24):
        chunk = items[start:start + 24]
        height, width = chunk[0][1].shape
        rgb = np.zeros((height, width * 2, 3), np.uint8)
        for bit, (_, mask) in enumerate(chunk):
            channel, value = bit // 8, 1 << (bit % 8)
            rgb[:, :width, channel][mask] |= value
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(outline, contours, -1, 1, 2)
            rgb[:, width:, channel][outline.astype(bool)] |= value
        ok, png = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise ValueError('Unable to encode mask atlas')
        yield [vid for vid, _ in chunk], png.tobytes()


def render_frame(payload, tracks, index, destination, quality):
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Unreadable source frame')
    source_shape = image.shape[:2]
    height = min(source_shape[0], 540)
    width = round(source_shape[1] * height / source_shape[0])
    image = cv2.resize(image, (width, height))
    ok, encoded = cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise ValueError('Unable to encode viewer frame')
    masks, labels = {}, {}
    for vid, track in sorted(tracks.items()):
        mask = decode_rle(track['masks'][index])
        if mask is None or not mask.any():
            continue
        if mask.shape != source_shape:
            raise ValueError('Mask and source frame dimensions differ')
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any():
            continue
        masks[vid] = mask
        ys, xs = np.where(mask)
        labels[vid] = [max(0, min(int(np.median(xs)), width - 35)), max(20, int(np.median(ys)))]
    return {'image': asset(destination, encoded.tobytes(), '.webp'), 'width': width, 'height': height,
            'layers': [{'ids': ids, 'image': asset(destination, png, '.png')} for ids, png in pack_masks(masks)],
            'labels': labels}


def interactive_sample(sample, row, old_mapping, mapping, media, destination, quality):
    """Render only the previously sampled frames; resume completed samples."""
    tracks = {t['tracklet_id']: t for t in json.loads(row['tracklets_json'])}
    old = {int(vid): tid for vid, tid in old_mapping.get('visual_to_local', {}).items()}
    new = {int(vid): tid for vid, tid in mapping.get('visual_to_local', {}).items()}
    # Stable IDs normally make the post-NLP geometry a subset of the original.
    # Also support older runs whose stages assigned different visual IDs.
    shared = all(vid in old and tracks[old[vid]]['masks'] == tracks[tid]['masks'] for vid, tid in new.items())
    keys = {'original': old, 'visual': old if shared else new}
    signature = json.dumps([3, quality, row['sample_id'], row['tracklets_json'], keys,
                            sample['original_frame_ids'], sample['frame_ids']], sort_keys=True)
    relative = 'masks/' + hashlib.sha256(signature.encode()).hexdigest() + '.json'
    descriptor = destination / relative
    result = json.loads(descriptor.read_text()) if descriptor.exists() else None
    if result is not None and not all((destination / p).is_file() for p in result['assets']):
        result = None
    if result is None:
        frame_indices = {str(fid): i for i, fid in enumerate(json.loads(row['frame_ids_json']))}
        result, cached = {'original': [], 'visual': []}, {}
        for stage, field in [('original', 'original_frame_ids'), ('visual', 'frame_ids')]:
            for fid in sample[field]:
                key = (fid, tuple(sorted(keys[stage].items())))
                if key not in cached:
                    cached[key] = render_frame(media.frame(row, fid),
                                              {vid: tracks[tid] for vid, tid in keys[stage].items()},
                                              frame_indices[fid], destination, quality)
                result[stage].append(cached[key])
        result['assets'] = sorted({path for frames in (result['original'], result['visual']) for frame in frames
                                   for path in [frame['image'], *[layer['image'] for layer in frame['layers']]]})
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        temporary = descriptor.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, separators=(',', ':')))
        temporary.replace(descriptor)
    sample.update(mask_data=relative, original_images=[f['image'] for f in result['original']],
                  images=[f['image'] for f in result['visual']])
    return result['assets']
