"""Dataset adapters that produce a shared, relocation-friendly work-unit shape."""

from __future__ import annotations

import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable


REF_SPLITS = {"train", "val", "test"}
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


def _ref_frame_source(root: Path, split: str, split_dir: Path) -> Path:
    archive_names = {
        "train": ("train.zip",),
        "val": ("valid.zip", "val.zip"),
        # Public test expressions are a 305-video subset of the original
        # 507-video valid pool, and their frames therefore live in valid.zip.
        # test_ytvos.zip is a separate 747-video challenge-media archive.
        "test": ("valid.zip", "val.zip", "test_ytvos.zip", "test.zip"),
    }[split]
    return _first_existing(
        [split_dir / "JPEGImages", root / "JPEGImages"]
        + [root / "archives" / name for name in archive_names]
        + [root / name for name in archive_names],
        kind=f"Ref-YouTube-VOS {split} JPEGImages directory or archive",
    )


def _json_member(path: Path, candidates: Iterable[str]) -> dict[str, Any] | None:
    if not path.is_file() or path.suffix.lower() != ".zip":
        return None
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        member = next((name for name in candidates if name in names), None)
        return json.loads(archive.read(member)) if member else None


def build_refytvos_units(
    data_root: Path,
    *,
    split: str,
    limit: int | None,
    seed: int,
    one_expression_per_video: bool = False,
) -> list[dict[str, Any]]:
    """Load only public full-video-authored expressions and complete sequences."""

    if split not in REF_SPLITS:
        raise ValueError(f"unsupported Ref-YouTube-VOS split: {split}")
    root = data_root.resolve()
    expression_path = _ref_expression_path(root, split)
    split_dir = _ref_split_dir(root, split)
    frame_source = _ref_frame_source(root, split, split_dir)
    annotation_candidates = [
        split_dir / "Annotations", root / "Annotations",
        root / "archives" / "train.zip", root / "train.zip",
    ]
    annotation_source = next(
        (path.resolve() for path in annotation_candidates
         if path.is_dir() or (split == "train" and path.is_file())),
        None,
    )
    meta_candidates = [split_dir / "meta.json", root / "meta.json"]
    meta_path = next((path.resolve() for path in meta_candidates if path.is_file()), None)
    metadata = _read_json(meta_path) if meta_path else _json_member(
        frame_source, ("train/meta.json", "meta.json")
    )
    categories = (metadata or {}).get("videos", {})
    videos = _read_json(expression_path).get("videos", {})

    # Public `valid` is the original 507-video pool; public `test` is a
    # 305-video subset. The competition validation split is the 202-video
    # set difference, preventing duplicate work when both splits are run.
    if split == "val":
        try:
            test_path = _ref_expression_path(root, "test")
        except FileNotFoundError:
            test_path = None
        if test_path is not None:
            test_video_ids = set(_read_json(test_path).get("videos", {}))
            videos = {key: value for key, value in videos.items() if key not in test_video_ids}

    rows: list[dict[str, Any]] = []
    for video_id, video in videos.items():
        frames = [str(value) for value in video.get("frames", [])]
        if not frames:
            continue
        for expression_id, expression in video.get("expressions", {}).items():
            text = str(expression.get("exp", "")).strip()
            if not text:
                continue
            object_ids = _normalize_object_ids(expression.get("obj_id"))
            has_ground_truth = bool(annotation_source and object_ids)
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
                        f"refytvos__{split}__full_video__{_safe(str(video_id))}"
                        f"__e{_safe(str(expression_id))}"
                    ),
                    "dataset": "ref_youtube_vos",
                    "split": split,
                    "cohort": "full_video",
                    "annotation_protocol": "public_full_video_expression",
                    "provenance_warning": None,
                    "dataset_root": str(root),
                    "frame_source": str(frame_source),
                    "annotation_source": str(annotation_source) if annotation_source else None,
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
        rows, limit=limit, seed=seed, one_expression_per_video=one_expression_per_video
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
        [
            root / "mask_dict.sqlite",
            root / "mask_dict.json",
            root / split / "mask_dict.sqlite",
            root / split / "mask_dict.json",
        ],
        kind="ReVOS mask_dict.sqlite or mask_dict.json",
    )
    # Prefer the indexed source tar over a directory. This avoids expanding
    # tens of thousands of small JPEGs onto shared filesystems; it also makes
    # an interrupted legacy extraction harmless.
    frame_source = _first_existing(
        [
            root / "ReVOS.tar",
            root / "JPEGImages.zip",
            root / "JPEGImages",
            root / split / "JPEGImages",
        ],
        kind="ReVOS indexed tar, JPEGImages directory, or ZIP",
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
