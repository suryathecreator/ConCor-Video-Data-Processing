from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from concor_video.pipeline import (
    DatasetProvider,
    configure_predictor_memory_mode,
    process_unit,
    temporal_iou,
)
from concor_video.rle import decode_rle, encode_rle


class FakePredictor:
    def __init__(self) -> None:
        self.prompt = ""

    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "test"}
        if request["type"] == "add_prompt":
            self.prompt = request["text"]
        return {}

    def handle_stream_request(self, request):
        mask = np.zeros((20, 20), dtype=bool)
        if "dog" in self.prompt:
            mask[2:12, 2:12] = True
        else:
            mask[10:20, 10:20] = True
        yield {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": np.asarray([1]),
                "out_binary_masks": np.asarray([mask]),
                "out_probs": np.asarray([0.9]),
            },
        }


def _unit(root, *, ground_truth: bool) -> dict:
    frame_dir = root / "frames/video"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(frame_dir / "00000.jpg")
    annotation_dir = None
    if ground_truth:
        annotation_dir = root / "annotations"
        (annotation_dir / "video").mkdir(parents=True)
        labels = np.zeros((20, 20), dtype=np.uint8)
        labels[0:8, 0:8] = 1
        Image.fromarray(labels).save(annotation_dir / "video/00000.png")
    text = "the dog beside a ball"
    extraction = {
        "spacy_model": "fixture",
        "target": {
            "surface": "the dog",
            "start": 0,
            "end": 7,
            "head": "dog",
            "sam_prompt": "dog",
        },
        "target_coreference_spans": [],
        "contexts": [
            {
                "surface": "a ball",
                "start": 15,
                "end": 21,
                "head": "ball",
                "sam_prompt": "ball",
            }
        ],
        "ignored": [],
    }
    return {
        "sample_id": "sample",
        "dataset": "ref_youtube_vos",
        "split": "train" if ground_truth else "val",
        "cohort": "full_video",
        "annotation_protocol": "public_full_video_expression",
        "provenance_warning": None,
        "frame_source": str(root / "frames"),
        "annotation_source": str(annotation_dir) if annotation_dir else None,
        "mask_dict_path": None,
        "video_id": "video",
        "expression_id": "0",
        "text": text,
        "frame_ids": ["00000"],
        "target_object_ids": [1],
        "target_annotation_ids": [],
        "target_source_expected": (
            "official_dataset_ground_truth" if ground_truth else "sam3.1_multiplex"
        ),
        "negative": False,
        "extraction": extraction,
        "sam_prompt_groups": [
            {"sam_prompt": "ball", "candidates": extraction["contexts"]}
        ],
    }


def test_ground_truth_target_and_sam_context(tmp_path) -> None:
    record = process_unit(
        _unit(tmp_path / "inputs", ground_truth=True),
        provider=DatasetProvider(tmp_path / "cache"),
        predictor=FakePredictor(),
    )
    assert record["disposition"] == "complete_bcc"
    assert {row["source"] for row in record["tracklets"]} == {
        "ground_truth",
        "sam3.1_context",
    }
    ball_link = next(row for row in record["span_links"] if row["text"] == "a ball")
    assert ball_link["tracklet_ids"] == ["sam-context-001"]


class NoProposalPredictor(FakePredictor):
    def handle_stream_request(self, request):
        raise RuntimeError("No points are provided; please add points first")


def test_no_sam_proposal_is_a_disposition_not_an_error(tmp_path) -> None:
    record = process_unit(
        _unit(tmp_path / "inputs", ground_truth=False),
        provider=DatasetProvider(tmp_path / "cache"),
        predictor=NoProposalPredictor(),
    )
    assert record["disposition"] == "missing_main_referent"
    assert record["tracklets"] == []


def test_missing_public_target_is_labeled_sam_prediction(tmp_path) -> None:
    record = process_unit(
        _unit(tmp_path / "inputs", ground_truth=False),
        provider=DatasetProvider(tmp_path / "cache"),
        predictor=FakePredictor(),
    )
    assert record["tracklets"][0]["source"] == "sam3.1_main_referent"
    assert record["pipeline"]["target_source"] == "sam3.1_multiplex"


def test_missing_official_target_is_audited_not_raised(tmp_path) -> None:
    unit = _unit(tmp_path / "inputs", ground_truth=True)
    unit["target_object_ids"] = [99]
    record = process_unit(
        unit,
        provider=DatasetProvider(tmp_path / "cache"),
        predictor=FakePredictor(),
    )
    assert record["disposition"] == "missing_main_referent"
    assert record["tracklets"] == []
    assert record["sam_prompt_audit"][0]["rejections"][0]["reason"] == (
        "official_target_tracklet_empty_or_missing"
    )


def test_invalid_existing_target_span_is_repaired(tmp_path) -> None:
    unit = _unit(tmp_path / "inputs", ground_truth=True)
    unit["text"] = "seating(s) designed for rest."
    unit["extraction"]["target"].update(
        {"surface": "", "start": 0, "end": 0, "head": "seating", "sam_prompt": "seating"}
    )
    unit["sam_prompt_groups"] = []
    record = process_unit(
        unit,
        provider=DatasetProvider(tmp_path / "cache"),
        predictor=FakePredictor(),
    )
    assert record["groups"][0]["text_spans"] == [
        {"start": 0, "end": 7, "text": "seating"}
    ]
    assert (
        "repaired invalid main-referent span deterministically"
        in record["extraction"]["notes"]
    )


def test_rle_and_temporal_iou_round_trip() -> None:
    mask = np.zeros((8, 11), dtype=bool)
    mask[2:7, 3:9] = True
    decoded = decode_rle(encode_rle(mask))
    assert np.array_equal(decoded, mask)
    assert temporal_iou([mask, None], [decoded, None]) == 1.0


def test_memory_safe_mode_microbatches_and_restores_fast_defaults():
    predictor = SimpleNamespace(
        model=SimpleNamespace(
            batched_grounding_batch_size=16, postprocess_batch_size=16
        )
    )
    predictor.model.max_num_objects = 32
    assert configure_predictor_memory_mode(predictor, enabled=True) == {
        "grounding": 1,
        "postprocess": 1,
        "objects": 16,
    }
    assert configure_predictor_memory_mode(
        predictor, enabled=True, max_num_objects=8
    )["objects"] == 8
    assert configure_predictor_memory_mode(predictor, enabled=False) == {
        "grounding": 16,
        "postprocess": 16,
        "objects": 32,
    }
