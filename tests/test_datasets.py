from __future__ import annotations

import json
import zipfile
import numpy as np
from PIL import Image
from concor_video.datasets import (
    build_refytvos_units,
    build_revos_units,
    sanitize_frame_ids,
)


def _write_ref_split(root, split, videos):
    (root / f"meta_expressions/{split}").mkdir(parents=True, exist_ok=True)
    (root / f"{split}/JPEGImages").mkdir(parents=True, exist_ok=True)
    (root / f"meta_expressions/{split}/meta_expressions.json").write_text(
        json.dumps({"videos": videos})
    )


def test_hidden_packaging_entries_are_not_frames() -> None:
    frames, ignored = sanitize_frame_ids(
        ["00000", ".ipynb_checkpoints", "nested/.cache", "00001"]
    )
    assert frames == ["00000", "00001"]
    assert ignored == [".ipynb_checkpoints", "nested/.cache"]


def test_refytvos_public_full_video_and_ground_truth(tmp_path) -> None:
    root = tmp_path / "ref"
    videos = {"video-a":{"frames":["00000",".ipynb_checkpoints","00001"],"expressions":{"0":{"exp":"an owl on an arm","obj_id":1}}}}
    _write_ref_split(root, "train", videos)
    (root / "train/Annotations/video-a").mkdir(parents=True)
    (root / "train/JPEGImages/video-a").mkdir(parents=True)
    for frame in ("00000","00001"):
        Image.new("RGB",(12,12)).save(root / f"train/JPEGImages/video-a/{frame}.jpg")
        labels=np.zeros((12,12),dtype=np.uint8); labels[:,:6]=1
        Image.fromarray(labels).save(root / f"train/Annotations/video-a/{frame}.png")
    row=build_refytvos_units(root,split="train",limit=None,seed=7)[0]
    assert row["frame_ids"]==["00000","00001"]
    assert row["cohort"]=="full_video"
    assert row["annotation_protocol"]=="public_full_video_expression"
    assert row["target_source_expected"]=="official_dataset_ground_truth"


def test_refytvos_validation_excludes_competition_test_videos(tmp_path) -> None:
    root=tmp_path/"ref"
    valid={
        "val-a":{"frames":["0"],"expressions":{"0":{"exp":"a cat","obj_id":1}}},
        "test-a":{"frames":["0"],"expressions":{"0":{"exp":"a dog","obj_id":1}}},
    }
    test={"test-a":valid["test-a"]}
    _write_ref_split(root,"valid",valid); _write_ref_split(root,"test",test)
    rows=build_refytvos_units(root,split="val",limit=None,seed=1)
    assert [row["video_id"] for row in rows]==["val-a"]
    assert rows[0]["target_source_expected"]=="sam3.1_multiplex"


def test_refytvos_public_test_uses_valid_pool_media(tmp_path) -> None:
    root = tmp_path / "ref"
    (root / "meta_expressions/test").mkdir(parents=True)
    (root / "archives").mkdir()
    videos = {
        "test-a": {
            "frames": ["00000"],
            "expressions": {"0": {"exp": "a dog", "obj_id": 1}},
        }
    }
    (root / "meta_expressions/test/meta_expressions.json").write_text(
        json.dumps({"videos": videos})
    )
    with zipfile.ZipFile(root / "archives/valid.zip", "w") as archive:
        archive.writestr("valid/JPEGImages/test-a/00000.jpg", b"jpeg")
    with zipfile.ZipFile(root / "archives/test_ytvos.zip", "w") as archive:
        archive.writestr("test/JPEGImages/unrelated/00000.jpg", b"jpeg")

    row = build_refytvos_units(root, split="test", limit=None, seed=1)[0]
    assert row["frame_source"] == str((root / "archives/valid.zip").resolve())


def test_revos_category_selection(tmp_path) -> None:
    root=tmp_path/"revos"; (root/"JPEGImages/video-a").mkdir(parents=True)
    (root/"mask_dict.json").write_text("{}")
    expressions={"videos":{"video-a":{"frames":["00000"],"expressions":{
        "0":{"exp":"Which ball is on the rack?","type_id":0,"obj_id":[1],"anno_id":[9]},
        "1":{"exp":"the object that never appears","type_id":2,"obj_id":[],"anno_id":[]},
    }}}}
    (root/"meta_expressions_train_.json").write_text(json.dumps(expressions))
    rows=build_revos_units(root,split="train",categories={"explicit","nonexistent"},limit_per_category=None,seed=9)
    assert {row["cohort"] for row in rows}=={"explicit","nonexistent"}
    assert next(row for row in rows if row["cohort"]=="nonexistent")["negative"]
