"""Convert per-sample checkpoints into compact, UI-friendly Parquet tables."""

from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import orjson
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
        ("annotation_protocol", pa.string()),
        ("provenance_warning", pa.string()),
        ("dataset_root", pa.string()),
        ("frame_source", pa.string()),
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

VERIFICATION_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("cohort", pa.string()),
        ("annotation_protocol", pa.string()),
        ("provenance_warning", pa.string()),
        ("dataset_root", pa.string()),
        ("frame_source", pa.string()),
        ("video_id", pa.string()),
        ("expression_id", pa.string()),
        ("text", pa.string()),
        ("negative", pa.bool_()),
        ("target_source", pa.string()),
        ("disposition", pa.string()),
        ("frame_ids_json", pa.string()),
        ("frame_files_json", pa.string()),
        ("tracklets_json", pa.string()),
        ("groups_json", pa.string()),
        ("span_links_json", pa.string()),
        ("extraction_json", pa.string()),
        ("sam_prompt_audit_json", pa.string()),
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
    return orjson.dumps(value).decode("utf-8")


class _AtomicParquetSink:
    """Incrementally build one Parquet file without retaining the campaign."""

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        *,
        max_batch_rows: int = 64,
        max_batch_chars: int = 64 * 1024 * 1024,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temporary = path.with_suffix(path.suffix + f".{os.getpid()}.part")
        self.schema = schema
        self.max_batch_rows = max_batch_rows
        self.max_batch_chars = max_batch_chars
        self.count = 0
        self._batch_chars = 0
        self._batch: list[dict[str, Any]] = []
        self._writer = pq.ParquetWriter(self.temporary, schema, compression="zstd")
        self._closed = False

    def append(self, row: dict[str, Any]) -> None:
        self._batch.append(row)
        self.count += 1
        self._batch_chars += sum(
            len(value) for value in row.values() if isinstance(value, str)
        )
        if (
            len(self._batch) >= self.max_batch_rows
            or self._batch_chars >= self.max_batch_chars
        ):
            self._flush()

    def _flush(self) -> None:
        if not self._batch:
            return
        self._writer.write_table(pa.Table.from_pylist(self._batch, schema=self.schema))
        self._batch.clear()
        self._batch_chars = 0

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._writer.close()
        self._closed = True

    def commit(self) -> None:
        self.close()
        self.temporary.replace(self.path)

    def abort(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True
        self.temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, rows: Iterable[dict[str, Any]], schema: pa.Schema) -> int:
    sink = _AtomicParquetSink(path, schema)
    try:
        for row in rows:
            sink.append(row)
        sink.commit()
    except BaseException:
        sink.abort()
        raise
    return sink.count


def _sample_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "dataset": record["dataset"],
        "split": record["split"],
        "cohort": record["cohort"],
        "annotation_protocol": record.get("annotation_protocol", ""),
        "provenance_warning": record.get("provenance_warning"),
        "dataset_root": record.get("dataset_root"),
        "frame_source": record.get("frame_source"),
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



def _verification_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "dataset": record["dataset"],
        "split": record["split"],
        "cohort": record["cohort"],
        "annotation_protocol": record.get("annotation_protocol", ""),
        "provenance_warning": record.get("provenance_warning"),
        "dataset_root": record.get("dataset_root"),
        "frame_source": record.get("frame_source"),
        "video_id": record["video_id"],
        "expression_id": record["expression_id"],
        "text": record["text"],
        "negative": bool(record["negative"]),
        "target_source": str(record.get("pipeline", {}).get("target_source", "unknown")),
        "disposition": record["disposition"],
        "frame_ids_json": _json(record["frame_ids"]),
        "frame_files_json": _json(record.get("frame_files", [])),
        "tracklets_json": _json(record["tracklets"]),
        "groups_json": _json(record["groups"]),
        "span_links_json": _json(record["span_links"]),
        "extraction_json": _json(record.get("extraction", {})),
        "sam_prompt_audit_json": _json(record.get("sam_prompt_audit", [])),
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
    worklist = orjson.loads(worklist_path.read_bytes())
    records_dir = campaign_root / "records"
    errors_dir = campaign_root / "errors"
    ledger: list[dict[str, Any]] = []
    dispositions: Counter[str] = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    sinks = {
        "samples": _AtomicParquetSink(output_dir / "samples.parquet", SAMPLE_SCHEMA),
        "verification": _AtomicParquetSink(
            output_dir / "verification.parquet", VERIFICATION_SCHEMA
        ),
        "tracklets": _AtomicParquetSink(
            output_dir / "tracklets.parquet", TRACKLET_SCHEMA
        ),
        "links": _AtomicParquetSink(output_dir / "links.parquet", LINK_SCHEMA),
    }
    try:
        for unit in worklist["units"]:
            record_path = records_dir / f"{unit['sample_id']}.json"
            error_path = errors_dir / f"{unit['sample_id']}.json"
            record = None
            error = None
            if record_path.is_file():
                record = orjson.loads(record_path.read_bytes())
                validate_record(record)
                sinks["samples"].append(_sample_row(record))
                sinks["verification"].append(_verification_row(record))
                for row in _tracklet_rows((record,)):
                    sinks["tracklets"].append(row)
                for row in _link_rows((record,)):
                    sinks["links"].append(row)
                status = "completed"
                dispositions[record["disposition"]] += 1
            elif error_path.is_file():
                error = orjson.loads(error_path.read_bytes())
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
        for sink in sinks.values():
            sink.close()
        for sink in sinks.values():
            sink.commit()
    except BaseException:
        for sink in sinks.values():
            sink.abort()
        raise

    sample_count = sinks["samples"].count
    verification_count = sinks["verification"].count
    tracklet_count = sinks["tracklets"].count
    link_count = sinks["links"].count

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
        "verification_rows": verification_count,
        "tracklet_rows": tracklet_count,
        "link_rows": link_count,
        "status_counts": dict(sorted(status_counts.items())),
        "disposition_counts": dict(sorted(dispositions.items())),
        "tables": {
            "samples": "samples.parquet",
            "verification": "verification.parquet",
            "tracklets": "tracklets.parquet",
            "links": "links.parquet",
            "ledger": "run_ledger.csv",
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest
