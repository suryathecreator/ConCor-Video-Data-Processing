"""Independent mask geometry must survive packing, overlap, and stage selection."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from concor_video.rle import encode_rle

spec = importlib.util.spec_from_file_location('viewer_masks', Path(__file__).resolve().parents[1] / 'scripts/viewer_masks.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def test_atlas_preserves_overlapping_masks_holes_and_all_rgb_bits():
    masks = {}
    for i in range(1, 29):
        mask = np.zeros((25, 30), bool)
        mask[2:23, i % 5:25 + i % 5] = True
        mask[8:17, 10:20] = False
        masks[i] = mask
    atlases = list(viewer.pack_masks(masks))
    assert [len(ids) for ids, _ in atlases] == [24, 4]
    for ids, payload in atlases:
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        bits = rgb.astype(np.uint32)
        bits = bits[:, :, 0] | bits[:, :, 1] << 8 | bits[:, :, 2] << 16
        for bit, vid in enumerate(ids):
            assert np.array_equal((bits[:, :30] & (1 << bit)) != 0, masks[vid])
            assert (bits[:, 30:] & (1 << bit)).any()
            assert not (bits[12, 15] & (1 << bit))  # The interior hole stays empty.


def fixture():
    image = np.full((30, 40, 3), 100, np.uint8)
    payload = cv2.imencode('.png', image)[1].tobytes()
    first, second = np.zeros((30, 40), bool), np.zeros((30, 40), bool)
    first[3:20, 2:25], second[10:27, 10:38] = True, True
    tracks = [{'tracklet_id': 'a', 'masks': [encode_rle(first), None]},
              {'tracklet_id': 'b', 'masks': [encode_rle(second), encode_rle(second)]}]
    row = {'sample_id': 'fixture', 'frame_ids_json': '["0","1"]', 'tracklets_json': json.dumps(tracks)}
    sample = {'original_frame_ids': ['0', '1'], 'frame_ids': ['0', '1']}
    return payload, row, sample


def test_interactive_frames_share_geometry_and_resume_without_reading_media(tmp_path):
    payload, row, sample = fixture()
    old, new = {'visual_to_local': {'1': 'a', '2': 'b'}}, {'visual_to_local': {'2': 'b'}}
    media = SimpleNamespace(frame=lambda *args: payload)
    viewer.interactive_sample(sample, row, old, new, media, tmp_path, 76)
    result = json.loads((tmp_path / sample['mask_data']).read_text())
    assert result['original'] == result['visual']
    assert result['original'][0]['layers'][0]['ids'] == [1, 2]
    assert result['original'][1]['layers'][0]['ids'] == [2]
    assert sample['images'] == sample['original_images']
    assert all((tmp_path / p).is_file() for p in result['assets'])
    media.frame = lambda *args: pytest.fail('Completed viewer sample should resume without media reads')
    viewer.interactive_sample(sample, row, old, new, media, tmp_path, 76)


def test_old_runs_with_renumbered_stages_get_distinct_geometry(tmp_path):
    payload, row, sample = fixture()
    viewer.interactive_sample(sample, row, {'visual_to_local': {'1': 'a'}}, {'visual_to_local': {'1': 'b'}},
                              SimpleNamespace(frame=lambda *args: payload), tmp_path, 76)
    result = json.loads((tmp_path / sample['mask_data']).read_text())
    assert result['original'][0]['layers'][0]['image'] != result['visual'][0]['layers'][0]['image']
    assert result['original'][1]['layers'] == []
    assert result['visual'][1]['layers'][0]['ids'] == [1]


def test_renderer_rejects_misaligned_source_annotations(tmp_path):
    payload, _, _ = fixture()
    with pytest.raises(ValueError, match='dimensions differ'):
        viewer.render_frame(payload, {1: {'masks': [encode_rle(np.ones((2, 2), bool))]}}, 0, tmp_path, 76)
