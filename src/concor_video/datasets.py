"""Dataset adapters that produce a shared, relocation-friendly work-unit shape."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


REF_SPLITS = {"train", "val", "test"}
REF_MODES = {"full_video", "first_frame"}
REVOS_SPLITS = {"train", "val"}
REVOS_CATEGORIES = {"implicit", "explicit", "nonexistent"}
REVOS_TYPE_NAMES = {0: "explicit", 1: "implicit", 2: "nonexistent"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(candidates: Iterable[Path], *, kind: str) -> Path:
    values = list(candidates)
    for path in values:
        if path.exists():
            return path.resolve()
    rendered = "\n  - ".join(str(path) for path in values)
    raise FileNotFoundError(f"could not find {kind}; checked:\n  - {rendered}")


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "-", value).strip("-")


def _sample(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    seed: int,
    one_expression_per_video: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["video_id"], row["expression_id"]))
    if limit is None:
        if not one_expression_per_video:
            return ordered
        seen: set[str] = set()
        distinct: list[dict[str, Any]] = []
        for row in ordered:
            if row["video_id"] not in seen:
                seen.add(row["video_id"])
                distinct.append(row)
        return distinct
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    if one_expression_per_video:
        seen: set[str] = set()
        distinct: list[dict[str, Any]] = []
        for row in shuffled:
            if row["video_id"] in seen:
                continue
            seen.add(row["video_id"])
            distinct.append(row)
        shuffled = distinct
    if len(shuffled) < limit:
        raise ValueError(f"requested {limit} examples, but only {len(shuffled)} are available")
    return shuffled[:limit]


def _normalize_object_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _ref_expression_path(root: Path, split: str) -> Path:
    aliases = [split]
    if split == "val":
        aliases.append("valid")
    return _first_existing(
        [
            candidate
            for alias in aliases
            for candidate in (
                root / "meta_expressions" / alias / "meta_expressions.json",
                root / alias / "meta_expressions.json",
                root / f"meta_expressions_{alias}.json",
            )
        ],
        kind=f"Ref-YouTube-VOS {split} expression metadata",
    )


def _ref_split_dir(root: Path, split: str) -> Path:
    aliases = {
        "train": ["train"],
        "val": ["valid", "val"],
        # The official competition val/test expression subsets share the
        # released `valid/JPEGImages` pool in common distributions.
        "test": ["test", "valid", "val"],
    }[split]
    return _first_existing(
        [root / alias for alias in aliases] + [root],
        kind=f"Ref-YouTube-VOS {split} media directory",
    )


def build_refytvos_units(
    data_root: Path,
    *,
    split: str,
    mode: str,
    limit: int | None,
    seed: int,
    one_expression_per_video: bool = False,
) -> list[dict[str, Any]]:
    """Load public full-video expressions and choose full-sequence/one-frame processing."""

    if split not in REF_SPLITS:
        raise ValueError(f"unsupported Ref-YouTube-VOS split: {split}")
    if mode not in REF_MODES:
        raise ValueError(f"unsupported Ref-YouTube-VOS mode: {mode}")
    root = data_root.resolve()
    expression_path = _ref_expression_path(root, split)
    split_dir = _ref_split_dir(root, split)
    frame_dir = _first_existing(
        [split_dir / "JPEGImages", root / "JPEGImages"],
        kind="Ref-YouTube-VOS JPEGImages directory",
    )
    annotation_candidates = [split_dir / "Annotations", root / "Annotations"]
    annotation_dir = next((path.resolve() for path in annotation_candidates if path.is_dir()), None)
    meta_candidates = [split_dir / "meta.json", root / "meta.json"]
    meta_path = next((path.resolve() for path in meta_candidates if path.is_file()), None)
    categories = _read_json(meta_path).get("videos", {}) if meta_path else {}
    videos = _read_json(expression_path).get("videos", {})

    rows: list[dict[str, Any]] = []
    for video_id, video in videos.items():
        all_frames = [str(value) for value in video.get("frames", [])]
        if not all_frames:
            continue
        frames = all_frames if mode == "full_video" else all_frames[:1]
        for expression_id, expression in video.get("expressions", {}).items():
            text = str(expression.get("exp", "")).strip()
            if not text:
                continue
            object_ids = _normalize_object_ids(expression.get("obj_id"))
            has_ground_truth = bool(
                annotation_dir
                and object_ids
                and (annotation_dir / str(video_id) / f"{frames[0]}.png").is_file()
            )
            category = "unknown"
            if object_ids:
                category = str(
                    categories.get(str(video_id), {})
                    .get("objects", {})
                    .get(str(object_ids[0]), {})
                    .get("category", "unknown")
                )
            rows.append(
                {
                    "sample_id": (
                        f"refytvos__{split}__{mode}__{_safe(str(video_id))}"
                        f"__e{_safe(str(expression_id))}"
                    ),
                    "dataset": "ref_youtube_vos",
                    "split": split,
                    "cohort": mode,
                    "annotation_protocol": "public_full_video_expression",
                    "provenance_warning": (
                        "first_frame processes one frame from the public full-video-language "
                        "release; it is not the retired first-frame-language annotation subset"
                        if mode == "first_frame"
                        else None
                    ),
                    "dataset_root": str(root),
                    "frame_source": str(frame_dir),
                    "annotation_source": str(annotation_dir) if annotation_dir else None,
                    "mask_dict_path": None,
                    "video_id": str(video_id),
                    "expression_id": str(expression_id),
                    "text": text,
                    "frame_ids": frames,
                    "target_category": category,
                    "target_object_ids": object_ids,
                    "target_annotation_ids": [],
                    "target_source_expected": (
                        "official_dataset_ground_truth" if has_ground_truth else "sam3.1_multiplex"
                    ),
                    "negative": False,
                }
            )
    return _sample(
        rows,
        limit=limit,
        seed=seed,
        one_expression_per_video=one_expression_per_video,
    )


def _revos_metadata_path(root: Path, split: str) -> Path:
    aliases = ["valid", "val"] if split == "val" else ["train"]
    return _first_existing(
        [
            candidate
            for alias in aliases
            for candidate in (
                root / f"meta_expressions_{alias}_.json",
                root / f"meta_expressions_{alias}.json",
                root / alias / "meta_expressions.json",
            )
        ],
        kind=f"ReVOS {split} expression metadata",
    )


def build_revos_units(
    data_root: Path,
    *,
    split: str,
    categories: set[str],
    limit_per_category: int | None,
    seed: int,
    one_expression_per_video: bool = False,
) -> list[dict[str, Any]]:
    if split not in REVOS_SPLITS:
        raise ValueError(f"unsupported ReVOS split: {split}")
    unknown = categories - REVOS_CATEGORIES
    if unknown or not categories:
        raise ValueError(f"invalid ReVOS categories: {sorted(unknown or categories)}")
    root = data_root.resolve()
    metadata_path = _revos_metadata_path(root, split)
    mask_dict_path = _first_existing(
        [root / "mask_dict.json", root / split / "mask_dict.json"],
        kind="ReVOS mask_dict.json",
    )
    frame_source = _first_existing(
        [root / "JPEGImages", root / "JPEGImages.zip", root / split / "JPEGImages"],
        kind="ReVOS JPEGImages directory or ZIP",
    )
    videos = _read_json(metadata_path).get("videos", {})
    by_category: dict[str, list[dict[str, Any]]] = {name: [] for name in categories}
    for video_id, video in videos.items():
        frames = [str(value) for value in video.get("frames", [])]
        if not frames:
            continue
        for expression_id, expression in video.get("expressions", {}).items():
            category = REVOS_TYPE_NAMES.get(int(expression.get("type_id", -1)))
            if category not in categories:
                continue
            text = str(expression.get("exp", "")).strip()
            if not text:
                continue
            negative = category == "nonexistent"
            by_category[category].append(
                {
                    "sample_id": (
                        f"revos__{split}__{category}__{_safe(str(video_id))}"
                        f"__e{_safe(str(expression_id))}"
                    ),
                    "dataset": "revos",
                    "split": split,
                    "cohort": category,
                    "annotation_protocol": "official_revos_reasoning_expression",
                    "provenance_warning": None,
                    "dataset_root": str(root),
                    "frame_source": str(frame_source),
                    "annotation_source": None,
                    "mask_dict_path": str(mask_dict_path),
                    "video_id": str(video_id),
                    "expression_id": str(expression_id),
                    "text": text,
                    "frame_ids": frames,
                    "target_category": None,
                    "target_object_ids": _normalize_object_ids(expression.get("obj_id")),
                    "target_annotation_ids": _normalize_object_ids(expression.get("anno_id")),
                    "target_source_expected": (
                        "none" if negative else "official_dataset_ground_truth"
                    ),
                    "negative": negative,
                }
            )

    selected: list[dict[str, Any]] = []
    for offset, category in enumerate(sorted(categories)):
        selected.extend(
            _sample(
                by_category[category],
                limit=limit_per_category,
                seed=seed + offset,
                one_expression_per_video=one_expression_per_video,
            )
        )
    return sorted(selected, key=lambda row: row["sample_id"])
