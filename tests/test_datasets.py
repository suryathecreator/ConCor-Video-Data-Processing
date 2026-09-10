from __future__ import annotations

import json

import numpy as np
from PIL import Image

from concor_video.datasets import build_refytvos_units, build_revos_units


def test_refytvos_modes_and_ground_truth_detection(tmp_path) -> None:
    root = tmp_path / "ref"
    (root / "meta_expressions/train").mkdir(parents=True)
    (root / "train/JPEGImages/video-a").mkdir(parents=True)
    (root / "train/Annotations/video-a").mkdir(parents=True)
    (root / "meta_expressions/train/meta_expressions.json").write_text(
        json.dumps(
            {
                "videos": {
                    "video-a": {
                        "frames": ["00000", "00001"],
                        "expressions": {"0": {"exp": "an owl on an arm", "obj_id": 1}},
                    }
                }
            }
        )
    )
    for frame in ("00000", "00001"):
        Image.new("RGB", (12, 12)).save(root / f"train/JPEGImages/video-a/{frame}.jpg")
        labels = np.zeros((12, 12), dtype=np.uint8)
        labels[:, :6] = 1
        Image.fromarray(labels).save(root / f"train/Annotations/video-a/{frame}.png")

    full = build_refytvos_units(
        root, split="train", mode="full_video", limit=None, seed=7
    )[0]
    first = build_refytvos_units(
        root, split="train", mode="first_frame", limit=None, seed=7
    )[0]
    assert full["frame_ids"] == ["00000", "00001"]
    assert first["frame_ids"] == ["00000"]
    assert full["target_source_expected"] == "official_dataset_ground_truth"
    assert "retired first-frame-language" in first["provenance_warning"]


def test_revos_category_selection(tmp_path) -> None:
    root = tmp_path / "revos"
    (root / "JPEGImages/video-a").mkdir(parents=True)
    (root / "mask_dict.json").write_text("{}")
    expressions = {
        "videos": {
            "video-a": {
                "frames": ["00000"],
                "expressions": {
                    "0": {"exp": "Which ball is on the rack?", "type_id": 0, "obj_id": [1], "anno_id": [9]},
                    "1": {"exp": "the object that never appears", "type_id": 2, "obj_id": [], "anno_id": []},
                },
            }
        }
    }
    (root / "meta_expressions_train_.json").write_text(json.dumps(expressions))
    rows = build_revos_units(
        root,
        split="train",
        categories={"explicit", "nonexistent"},
        limit_per_category=None,
        seed=9,
    )
    assert {row["cohort"] for row in rows} == {"explicit", "nonexistent"}
    assert next(row for row in rows if row["cohort"] == "nonexistent")["negative"]
