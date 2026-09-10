"""Deterministic single-dataset and official release campaign construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpointing import atomic_json
from .datasets import build_refytvos_units, build_revos_units
from .entity_extraction import extract_entities, grouped_prompts


WORKLIST_VERSION = "concor-video-worklist-v3"

# Counts in the currently public full-video-language Ref-YTVOS release and the
# official ReVOS train/validation metadata. These assertions catch partial,
# overlapping, or accidentally substituted releases before GPU work begins.
OFFICIAL_COUNTS = {
    "evaluation": {
        "ref_youtube_vos::val::full_video": 834,
        "ref_youtube_vos::test::full_video": 1262,
        "revos::val::explicit": 3130,
        "revos::val::implicit": 2475,
        "revos::val::nonexistent": 217,
    },
    "train": {
        "ref_youtube_vos::train::full_video": 12913,
        "revos::train::explicit": 16941,
        "revos::train::implicit": 12203,
        "revos::train::nonexistent": 108,
    },
}
OFFICIAL_VIDEO_COUNTS = {
    "evaluation": {
        "ref_youtube_vos::val": 202,
        "ref_youtube_vos::test": 305,
        "revos::val": 416,
    },
    "train": {
        "ref_youtube_vos::train": 3471,
        "revos::train": 626,
    },
}


def _annotate_units(units: list[dict[str, Any]]) -> None:
    for unit in units:
        extraction = extract_entities(
            unit["text"],
            target_category=unit.get("target_category"),
            negative=bool(unit["negative"]),
        )
        if (
            extraction.get("target")
            and unit["target_source_expected"] != "official_dataset_ground_truth"
        ):
            extraction["target"]["role"] = "target_sam"
            extraction["target"]["decision"] = "prompt_sam3.1_for_main_referent"
            extraction["target"]["reason"] = (
                "the split has no public target mask; provenance remains model-generated"
            )
        unit["extraction"] = extraction
        unit["sam_prompt_groups"] = grouped_prompts(extraction)


def _statistics(units: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    instruction_counts = Counter(
        f"{unit['dataset']}::{unit['split']}::{unit['cohort']}" for unit in units
    )
    videos: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        videos[f"{unit['dataset']}::{unit['split']}"].add(unit["video_id"])
    return (
        dict(sorted(instruction_counts.items())),
        {key: len(value) for key, value in sorted(videos.items())},
    )


def _assert_official_counts(
    scope: str, instruction_counts: dict[str, int], video_counts: dict[str, int]
) -> None:
    expected_instructions = OFFICIAL_COUNTS[scope]
    expected_videos = OFFICIAL_VIDEO_COUNTS[scope]
    if instruction_counts != expected_instructions:
        raise ValueError(
            f"{scope} instruction counts do not match the official release: "
            f"expected {expected_instructions}, got {instruction_counts}"
        )
    if video_counts != expected_videos:
        raise ValueError(
            f"{scope} video counts do not match the official release: "
            f"expected {expected_videos}, got {video_counts}"
        )


def build_worklist(
    *,
    dataset: str,
    data_root: Path,
    split: str,
    campaign_id: str,
    seed: int,
    revos_categories: set[str] | None = None,
    limit: int | None = None,
    limit_per_category: int | None = None,
    one_expression_per_video: bool = False,
) -> dict[str, Any]:
    if dataset == "refytvos":
        units = build_refytvos_units(
            data_root,
            split=split,
            limit=limit,
            seed=seed,
            one_expression_per_video=one_expression_per_video,
        )
    elif dataset == "revos":
        units = build_revos_units(
            data_root,
            split=split,
            categories=revos_categories or {"implicit", "explicit", "nonexistent"},
            limit_per_category=limit_per_category,
            seed=seed,
            one_expression_per_video=one_expression_per_video,
        )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    _annotate_units(units)
    counts, videos = _statistics(units)
    return {
        "schema_version": WORKLIST_VERSION,
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "data_root": str(data_root.resolve()),
        "split": split,
        "seed": seed,
        "selection": {
            "language_protocol": "public_full_video_expression"
            if dataset == "refytvos"
            else "official_revos_reasoning_expression",
            "revos_categories": sorted(revos_categories or []) if dataset == "revos" else [],
            "limit": limit,
            "limit_per_category": limit_per_category,
            "one_expression_per_video": one_expression_per_video,
        },
        "counts": counts,
        "video_counts": videos,
        "units": units,
    }


def build_release_worklist(
    *,
    scope: str,
    refytvos_root: Path,
    revos_root: Path,
    campaign_id: str,
    seed: int,
    verify_official_counts: bool = True,
) -> dict[str, Any]:
    """Build the standard full public evaluation or lower-priority train campaign."""

    if scope not in {"evaluation", "train"}:
        raise ValueError("scope must be evaluation or train")
    units: list[dict[str, Any]] = []
    if scope == "evaluation":
        for offset, split in enumerate(("val", "test")):
            units.extend(
                build_refytvos_units(
                    refytvos_root, split=split, limit=None, seed=seed + offset
                )
            )
        units.extend(
            build_revos_units(
                revos_root,
                split="val",
                categories={"implicit", "explicit", "nonexistent"},
                limit_per_category=None,
                seed=seed + 10,
            )
        )
    else:
        units.extend(
            build_refytvos_units(
                refytvos_root, split="train", limit=None, seed=seed
            )
        )
        units.extend(
            build_revos_units(
                revos_root,
                split="train",
                categories={"implicit", "explicit", "nonexistent"},
                limit_per_category=None,
                seed=seed + 10,
            )
        )
    units.sort(
        key=lambda row: (
            row["dataset"], row["split"], row["video_id"], row["expression_id"]
        )
    )
    _annotate_units(units)
    counts, videos = _statistics(units)
    if verify_official_counts:
        _assert_official_counts(scope, counts, videos)
    return {
        "schema_version": WORKLIST_VERSION,
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "mixed",
        "data_root": {
            "ref_youtube_vos": str(refytvos_root.resolve()),
            "revos": str(revos_root.resolve()),
        },
        "split": scope,
        "seed": seed,
        "selection": {
            "scope": scope,
            "ref_youtube_vos": "public_full_video_expressions_only",
            "revos_categories": ["explicit", "implicit", "nonexistent"],
            "official_count_assertions": verify_official_counts,
        },
        "counts": counts,
        "video_counts": videos,
        "units": units,
    }


def write_extraction_audit(path: Path, worklist: dict[str, Any]) -> None:
    lines = [
        f"# Extraction audit: {worklist['campaign_id']}",
        "",
        "Exact surface spans and normalized SAM prompts are stored separately.",
        "Ignored candidates are recorded instead of being silently discarded.",
        "",
    ]
    units = worklist["units"]
    if len(units) > 1000:
        lines.extend(
            [
                f"This campaign contains {len(units):,} instructions. The complete "
                "machine-readable extraction audit is embedded in `worklist.json`; "
                "this Markdown file intentionally remains a compact summary.",
                "",
                "## Instruction counts",
                "",
            ]
        )
        for key, value in worklist.get("counts", {}).items():
            lines.append(f"- {key}: {value:,}")
    else:
        for unit in units:
            extraction = unit["extraction"]
            lines.extend([f"## {unit['sample_id']}", "", f"- Text: `{unit['text']}`"])
            target = extraction.get("target")
            if target:
                lines.append(
                    f"- Main referent: `{target['surface']}` → "
                    f"`{target['sam_prompt']}` ({unit['target_source_expected']})"
                )
            else:
                lines.append("- Main referent: intentionally absent")
            for group in unit["sam_prompt_groups"]:
                surfaces = ", ".join(
                    f"`{row['surface']}`" for row in group["candidates"]
                )
                lines.append(
                    f"- Context SAM3.1 prompt `{group['sam_prompt']}` ← {surfaces}"
                )
            for row in extraction.get("ignored", []):
                lines.append(
                    f"- Ignored `{row['surface']}`: {row['decision']} — {row['reason']}"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_worklist(path: Path, value: dict[str, Any]) -> None:
    atomic_json(path, value)
    write_extraction_audit(path.with_name(f"{path.stem}-extraction-audit.md"), value)
