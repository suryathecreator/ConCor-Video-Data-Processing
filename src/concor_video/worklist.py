"""Deterministic campaign construction."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpointing import atomic_json
from .datasets import build_refytvos_units, build_revos_units
from .entity_extraction import extract_entities, grouped_prompts


WORKLIST_VERSION = "concor-video-worklist-v2"


def build_worklist(
    *,
    dataset: str,
    data_root: Path,
    split: str,
    campaign_id: str,
    seed: int,
    ref_mode: str = "full_video",
    revos_categories: set[str] | None = None,
    limit: int | None = None,
    limit_per_category: int | None = None,
    one_expression_per_video: bool = False,
) -> dict[str, Any]:
    if dataset == "refytvos":
        units = build_refytvos_units(
            data_root,
            split=split,
            mode=ref_mode,
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

    for unit in units:
        extraction = extract_entities(
            unit["text"],
            target_category=unit.get("target_category"),
            negative=bool(unit["negative"]),
        )
        # The language rules are independent of whether the official target
        # mask happens to be public for this split.
        if extraction.get("target") and unit["target_source_expected"] != "official_dataset_ground_truth":
            extraction["target"]["role"] = "target_sam"
            extraction["target"]["decision"] = "prompt_sam3.1_for_main_referent"
            extraction["target"]["reason"] = (
                "the split has no public target mask; provenance remains model-generated"
            )
        unit["extraction"] = extraction
        unit["sam_prompt_groups"] = grouped_prompts(extraction)

    counts = Counter(f"{unit['dataset']}::{unit['split']}::{unit['cohort']}" for unit in units)
    return {
        "schema_version": WORKLIST_VERSION,
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "data_root": str(data_root.resolve()),
        "split": split,
        "seed": seed,
        "selection": {
            "ref_mode": ref_mode if dataset == "refytvos" else None,
            "revos_categories": sorted(revos_categories or []) if dataset == "revos" else [],
            "limit": limit,
            "limit_per_category": limit_per_category,
            "one_expression_per_video": one_expression_per_video,
        },
        "counts": dict(sorted(counts.items())),
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
    for unit in worklist["units"]:
        extraction = unit["extraction"]
        lines.extend([f"## {unit['sample_id']}", "", f"- Text: `{unit['text']}`"])
        target = extraction.get("target")
        if target:
            lines.append(
                f"- Main referent: `{target['surface']}` → `{target['sam_prompt']}` "
                f"({unit['target_source_expected']})"
            )
        else:
            lines.append("- Main referent: intentionally absent")
        for group in unit["sam_prompt_groups"]:
            surfaces = ", ".join(f"`{row['surface']}`" for row in group["candidates"])
            lines.append(f"- Context SAM3.1 prompt `{group['sam_prompt']}` ← {surfaces}")
        for row in extraction.get("ignored", []):
            lines.append(f"- Ignored `{row['surface']}`: {row['decision']} — {row['reason']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_worklist(path: Path, value: dict[str, Any]) -> None:
    atomic_json(path, value)
    write_extraction_audit(path.with_name(f"{path.stem}-extraction-audit.md"), value)
