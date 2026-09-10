"""Convert per-sample checkpoints into compact, UI-friendly Parquet tables."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .checkpointing import atomic_json
from .tracklet_schema import validate_record


SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("cohort", pa.string()),
        ("video_id", pa.string()),
        ("expression_id", pa.string()),
        ("text", pa.string()),
        ("negative", pa.bool_()),
        ("target_source", pa.string()),
        ("disposition", pa.string()),
        ("frame_count", pa.int32()),
        ("tracklet_count", pa.int32()),
        ("unresolved_context_count", pa.int32()),
        ("frame_ids_json", pa.string()),
        ("frame_files_json", pa.string()),
        ("span_links_json", pa.string()),
        ("extraction_json", pa.string()),
        ("sam_prompt_audit_json", pa.string()),
        ("pipeline_json", pa.string()),
        ("runtime_seconds", pa.float64()),
    ]
)

TRACKLET_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("cohort", pa.string()),
        ("video_id", pa.string()),
        ("expression_id", pa.string()),
        ("text", pa.string()),
        ("disposition", pa.string()),
        ("tracklet_id", pa.string()),
        ("role", pa.string()),
        ("identity", pa.string()),
        ("source", pa.string()),
        ("source_annotation_id", pa.string()),
        ("sam_prompt", pa.string()),
        ("confidence", pa.float64()),
        ("max_confidence", pa.float64()),
        ("present_frames", pa.int32()),
        ("frame_count", pa.int32()),
        ("frame_ids_json", pa.string()),
        ("frame_files_json", pa.string()),
        ("text_spans_json", pa.string()),
        ("masks_rle_json", pa.string()),
    ]
)

LINK_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("text", pa.string()),
        ("span_start", pa.int32()),
        ("span_end", pa.int32()),
        ("span_text", pa.string()),
        ("tracklet_ids_json", pa.string()),
    ]
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _atomic_parquet(path: Path, rows: Iterable[dict[str, Any]], schema: pa.Schema) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    count = 0
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    batch: list[dict[str, Any]] = []
    try:
        for row in rows:
            batch.append(row)
            count += 1
            if len(batch) >= 256:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        writer.close()
        temporary.replace(path)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return count


def _sample_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "dataset": record["dataset"],
        "split": record["split"],
        "cohort": record["cohort"],
        "video_id": record["video_id"],
        "expression_id": record["expression_id"],
        "text": record["text"],
        "negative": bool(record["negative"]),
        "target_source": str(record.get("pipeline", {}).get("target_source", "unknown")),
        "disposition": record["disposition"],
        "frame_count": len(record["frame_ids"]),
        "tracklet_count": len(record["tracklets"]),
        "unresolved_context_count": len(record.get("unresolved_required_prompts", [])),
        "frame_ids_json": _json(record["frame_ids"]),
        "frame_files_json": _json(record.get("frame_files", [])),
        "span_links_json": _json(record["span_links"]),
        "extraction_json": _json(record.get("extraction", {})),
        "sam_prompt_audit_json": _json(record.get("sam_prompt_audit", [])),
        "pipeline_json": _json(record.get("pipeline", {})),
        "runtime_seconds": float(record.get("runtime", {}).get("seconds", 0.0)),
    }


def _group_by_tracklet(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for group in record["groups"]:
        for tracklet_id in group["tracklet_ids"]:
            prior = values.setdefault(
                tracklet_id,
                {"role": group["role"], "identities": set(), "text_spans": {}},
            )
            prior["identities"].add(group["identity"])
            for span in group["text_spans"]:
                prior["text_spans"][(span["start"], span["end"], span["text"])] = span
    return values


def _tracklet_rows(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in records:
        groups = _group_by_tracklet(record)
        for tracklet in record["tracklets"]:
            linked = groups[tracklet["tracklet_id"]]
            yield {
                "sample_id": record["sample_id"],
                "dataset": record["dataset"],
                "split": record["split"],
                "cohort": record["cohort"],
                "video_id": record["video_id"],
                "expression_id": record["expression_id"],
                "text": record["text"],
                "disposition": record["disposition"],
                "tracklet_id": tracklet["tracklet_id"],
                "role": linked["role"],
                "identity": "/".join(sorted(linked["identities"])),
                "source": tracklet["source"],
                "source_annotation_id": (
                    str(tracklet["source_annotation_id"])
                    if tracklet.get("source_annotation_id") is not None
                    else None
                ),
                "sam_prompt": tracklet.get("sam_prompt"),
                "confidence": float(tracklet.get("confidence", 0.0)),
                "max_confidence": float(
                    tracklet.get("max_confidence", tracklet.get("confidence", 0.0))
                ),
                "present_frames": int(tracklet["present_frames"]),
                "frame_count": len(record["frame_ids"]),
                "frame_ids_json": _json(record["frame_ids"]),
                "frame_files_json": _json(record.get("frame_files", [])),
                "text_spans_json": _json(
                    [linked["text_spans"][key] for key in sorted(linked["text_spans"])]
                ),
                "masks_rle_json": _json(tracklet["masks"]),
            }


def _link_rows(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in records:
        for link in record["span_links"]:
            yield {
                "sample_id": record["sample_id"],
                "text": record["text"],
                "span_start": int(link["start"]),
                "span_end": int(link["end"]),
                "span_text": link["text"],
                "tracklet_ids_json": _json(link["tracklet_ids"]),
            }


def export_campaign(
    *,
    worklist_path: Path,
    campaign_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    records_dir = campaign_root / "records"
    errors_dir = campaign_root / "errors"
    records: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    dispositions: Counter[str] = Counter()
    for unit in worklist["units"]:
        record_path = records_dir / f"{unit['sample_id']}.json"
        error_path = errors_dir / f"{unit['sample_id']}.json"
        record = None
        error = None
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            validate_record(record)
            records.append(record)
            status = "completed"
            dispositions[record["disposition"]] += 1
        elif error_path.is_file():
            error = json.loads(error_path.read_text(encoding="utf-8"))
            status = "failed"
        else:
            status = "pending"
        ledger.append(
            {
                "sample_id": unit["sample_id"],
                "dataset": unit["dataset"],
                "split": unit["split"],
                "cohort": unit["cohort"],
                "video_id": unit["video_id"],
                "expression_id": unit["expression_id"],
                "status": status,
                "disposition": record["disposition"] if record else "",
                "target_source_expected": unit["target_source_expected"],
                "context_prompt_count": len(unit.get("sam_prompt_groups", [])),
                "tracklet_count": len(record["tracklets"]) if record else 0,
                "error_type": error.get("error_type", "") if error else "",
                "error_message": error.get("message", "") if error else "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = _atomic_parquet(
        output_dir / "samples.parquet",
        (_sample_row(record) for record in records),
        SAMPLE_SCHEMA,
    )
    tracklet_count = _atomic_parquet(
        output_dir / "tracklets.parquet", _tracklet_rows(records), TRACKLET_SCHEMA
    )
    link_count = _atomic_parquet(output_dir / "links.parquet", _link_rows(records), LINK_SCHEMA)

    ledger_path = output_dir / "run_ledger.csv"
    temporary = ledger_path.with_suffix(f".csv.{os.getpid()}.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]) if ledger else [])
        if ledger:
            writer.writeheader()
            writer.writerows(ledger)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(ledger_path)

    status_counts = Counter(row["status"] for row in ledger)
    manifest = {
        "schema_version": "concor-video-export-v1",
        "campaign_id": worklist["campaign_id"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "worklist_units": len(worklist["units"]),
        "sample_rows": sample_count,
        "tracklet_rows": tracklet_count,
        "link_rows": link_count,
        "status_counts": dict(sorted(status_counts.items())),
        "disposition_counts": dict(sorted(dispositions.items())),
        "tables": {
            "samples": "samples.parquet",
            "tracklets": "tracklets.parquet",
            "links": "links.parquet",
            "ledger": "run_ledger.csv",
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest
