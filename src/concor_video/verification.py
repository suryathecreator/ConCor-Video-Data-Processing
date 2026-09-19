"""Offline, video-grouped verification server and deterministic Parquet exporter."""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import threading
import urllib.parse
import zipfile
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .tar_index import IndexedTarReader
from .tracklet_schema import rebuild_span_links


DECISIONS_VERSION = "concor-video-decisions-v1"
SELECTION_PROTOCOL = "selected_instructions_v1"
REVOS_PREVIEW_PROTOCOL = "revos_preview_v1"
REVOS_COHORTS = {"explicit", "implicit", "nonexistent"}
JSON_COLUMNS = {
    "frame_ids_json",
    "frame_files_json",
    "tracklets_json",
    "groups_json",
    "span_links_json",
    "extraction_json",
    "sam_prompt_audit_json",
}


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value) if isinstance(value, str) else value


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _video_key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}::{row['split']}::{row['video_id']}"


def _suggestion_order(rows: list[dict[str, Any]], video_key: str) -> list[str]:
    """Give stable variety while preferring instructions that have tracklets."""

    ranked = sorted(
        rows,
        key=lambda row: (
            not (row.get("tracklets_json") not in (None, "", "[]") and not row.get("negative")),
            hashlib.sha256(
                (video_key + "\0" + str(row["sample_id"])).encode("utf-8")
            ).digest(),
        ),
    )
    return [str(row["sample_id"]) for row in ranked]


def _is_revos_video(rows: list[dict[str, Any]]) -> bool:
    return bool(
        rows
        and str(rows[0].get("dataset", "")).lower() == "revos"
        and any(str(row.get("cohort", "")).lower() in REVOS_COHORTS for row in rows)
    )


def _revos_preview_order(rows: list[dict[str, Any]], video_key: str) -> list[str]:
    """All nonexistent expressions, then one stable explicit and implicit sample."""

    if not _is_revos_video(rows):
        return []
    ranked = _suggestion_order(rows, video_key)
    by_id = {str(row["sample_id"]): row for row in rows}
    nonexistent = [
        sample_id
        for sample_id in ranked
        if str(by_id[sample_id].get("cohort", "")).lower() == "nonexistent"
    ]
    preview = list(nonexistent)
    for cohort in ("explicit", "implicit"):
        sample_id = next(
            (
                candidate
                for candidate in ranked
                if str(by_id[candidate].get("cohort", "")).lower() == cohort
            ),
            None,
        )
        if sample_id is not None:
            preview.append(sample_id)
    return preview


def _candidate_media_sources(
    media_roots: list[Path], row: dict[str, Any]
) -> list[Path]:
    """Resolve archived source paths beneath one or more local media roots.

    Exported rows retain provenance paths from the processing host. For portable
    review, each suffix of that path is tried beneath every ``--media-root``. Thus
    ``.../ref-youtube-vos/archives/valid.zip`` naturally resolves below a local
    root without rewriting the Parquet.
    """

    stored = [
        Path(value)
        for value in (row.get("frame_source"), row.get("dataset_root"))
        if value
    ]
    candidates: list[Path] = []
    for root in media_roots:
        candidates.append(root)
        for source in stored:
            parts = source.parts[1:] if source.is_absolute() else source.parts
            for start in range(len(parts)):
                candidates.append(root.joinpath(*parts[start:]))
    candidates.extend(stored)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser())
        if key not in seen:
            seen.add(key)
            unique.append(candidate.expanduser())
    return unique


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    for key in JSON_COLUMNS:
        if key in value:
            value[key.removesuffix("_json")] = _loads(value.pop(key), [] if key != "extraction_json" else {})
    return value


def load_verification_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "sample_id",
        "dataset",
        "split",
        "video_id",
        "expression_id",
        "text",
        "frame_ids_json",
        "tracklets_json",
        "groups_json",
        "span_links_json",
    }
    for path in paths:
        table = pq.read_table(path)
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"{path} is not a verification Parquet; missing {sorted(missing)}")
        rows.extend(table.to_pylist())
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("input Parquets contain duplicate sample_id values")
    return sorted(
        rows,
        key=lambda row: (
            row["dataset"], row["split"], row["video_id"], row["expression_id"]
        ),
    )


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["text"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_decisions(value: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    if not value:
        value = {}
    version = value.get("schema_version", DECISIONS_VERSION)
    if version != DECISIONS_VERSION:
        raise ValueError(f"unsupported decisions schema: {version}")
    source = value.get("source_fingerprint")
    if source and source != fingerprint:
        raise ValueError("decisions.json belongs to a different input Parquet")
    normalized = {
        "schema_version": DECISIONS_VERSION,
        "source_fingerprint": fingerprint,
        "updated_at": value.get("updated_at"),
        "cursor": value.get("cursor", {"video_index": 0}),
        "quick_keys_enabled": bool(value.get("quick_keys_enabled", True)),
        "videos": value.get("videos", {}),
    }
    if "selection_protocol" in value:
        if value["selection_protocol"] != SELECTION_PROTOCOL:
            raise ValueError(f"unsupported instruction selection: {value['selection_protocol']}")
        normalized["selection_protocol"] = SELECTION_PROTOCOL
    if "suggestion_count" in value:
        count = value["suggestion_count"]
        if count not in (1, 3, 5, 10, "all"):
            raise ValueError("suggestion_count must be 1, 3, 5, 10, or 'all'")
        normalized["suggestion_count"] = count
    return normalized


def _validate_selections(
    rows: list[dict[str, Any]], decisions: dict[str, Any]
) -> dict[str, set[str]]:
    """Validate per-video selections before filtering any exported rows."""

    by_video: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_video[_video_key(row)].add(str(row["sample_id"]))
    selected_by_video: dict[str, set[str]] = {}
    protocol = decisions.get("selection_protocol")
    if protocol not in (None, SELECTION_PROTOCOL):
        raise ValueError(f"unsupported instruction selection: {protocol}")
    for key, video in decisions.get("videos", {}).items():
        archived = video.get("accepted_sample_ids", [])
        if not isinstance(archived, list) or any(not isinstance(item, str) for item in archived):
            raise ValueError(f"accepted_sample_ids for {key} must be a list of strings")
        if len(set(archived)) != len(archived) or not set(archived) <= by_video.get(key, set()):
            raise ValueError(f"{key} has duplicate or foreign accepted sample IDs")
        if video.get("review_mode") == REVOS_PREVIEW_PROTOCOL:
            preview = video.get("preview_sample_ids", [])
            if not isinstance(preview, list) or any(not isinstance(item, str) for item in preview):
                raise ValueError(f"preview_sample_ids for {key} must be a list of strings")
            if len(set(preview)) != len(preview) or not set(preview) <= by_video.get(key, set()):
                raise ValueError(f"{key} has duplicate or foreign ReVOS preview IDs")
            overrides = video.get("preview_override_sample_ids", [])
            if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
                raise ValueError(f"preview_override_sample_ids for {key} must be strings")
            if len(set(overrides)) != len(overrides) or not set(overrides) <= set(preview):
                raise ValueError(f"{key} has invalid ReVOS preview override IDs")
            for sample_id in preview:
                status = video.get("instructions", {}).get(sample_id, {}).get("status")
                if status not in {"accepted", "rejected"}:
                    raise ValueError(f"ReVOS preview instruction {sample_id} needs accepted/rejected status")
            continue
        if "selected_sample_ids" not in video:
            if protocol == SELECTION_PROTOCOL and video.get("status") == "accepted" and not archived:
                raise ValueError(f"accepted video {key} has no confirmed instruction")
            continue
        raw = video["selected_sample_ids"]
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise ValueError(f"selected_sample_ids for {key} must be a list of strings")
        selected = set(raw)
        if len(selected) != len(raw) or not selected <= by_video.get(key, set()):
            raise ValueError(f"{key} has duplicate or foreign selected sample IDs")
        if len(selected) > 1 and not video.get("allow_multiple", False):
            raise ValueError(f"{key} selected multiple instructions without multi-select")
        if video.get("status") == "accepted":
            if not selected:
                raise ValueError(f"accepted video {key} has no confirmed instruction")
            for sample_id in selected:
                edit = video.get("instructions", {}).get(sample_id, {})
                if edit.get("discarded") or edit.get("status") == "rejected":
                    raise ValueError(f"accepted video {key} selected discarded {sample_id}")
        selected_by_video[key] = selected
    return selected_by_video


def _eligible_ids(rows: list[dict[str, Any]], video: dict[str, Any]) -> list[str]:
    edits = video.get("instructions", {})
    return [
        str(row["sample_id"])
        for row in rows
        if not (
            edits.get(str(row["sample_id"]), {}).get("discarded")
            or edits.get(str(row["sample_id"]), {}).get("status") == "rejected"
        )
    ]


def _accepted_ids(
    rows: list[dict[str, Any]], video: dict[str, Any], selected: dict[str, set[str]], key: str
) -> list[str]:
    if video.get("status") != "accepted":
        return []
    eligible = set(_eligible_ids(rows, video))
    if key in selected:
        accepted = selected[key]
    elif "accepted_sample_ids" in video:
        accepted = set(video["accepted_sample_ids"])
    else:
        # Legacy video-level acceptance meant every non-discarded instruction.
        accepted = eligible
    candidates = accepted & eligible
    return [sample_id for sample_id in _suggestion_order(rows, key) if sample_id in candidates]


def _materialize_sampling(
    rows: list[dict[str, Any]], decisions: dict[str, Any]
) -> dict[str, Any]:
    """Materialize dataset-specific review state without losing legacy edits."""

    prepared = deepcopy(decisions)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_video_key(row)].append(row)
    videos = prepared.setdefault("videos", {})
    for key, video_rows in grouped.items():
        if not _is_revos_video(video_rows):
            continue
        had_decision = key in videos
        video = videos.setdefault(key, {"status": "undecided", "instructions": {}})
        was_adapter = video.get("review_mode") == REVOS_PREVIEW_PROTOCOL
        if was_adapter and video.get("video_decision_explicit") is not True:
            # The first adapter release derived video status from instruction count.
            # Those values were not user video decisions, so migrate them to undecided.
            video["status"] = "undecided"
            video["video_decision_explicit"] = False
        elif not was_adapter and had_decision and video.get("status") in {"accepted", "rejected"}:
            # A pre-adapter decision came from the original video-level interface.
            video["video_decision_explicit"] = True
        else:
            video.setdefault("video_decision_explicit", False)
        base_preview = _revos_preview_order(video_rows, key)
        base_preview_set = set(base_preview)
        available = {str(row["sample_id"]) for row in video_rows}
        raw_overrides = video.get("preview_override_sample_ids")
        if raw_overrides is None and was_adapter:
            raw_overrides = [
                sample_id
                for sample_id in video.get("preview_sample_ids", [])
                if sample_id not in base_preview_set
            ]
        raw_overrides = raw_overrides or []
        if not isinstance(raw_overrides, list) or any(
            not isinstance(sample_id, str) for sample_id in raw_overrides
        ):
            raise ValueError(f"preview_override_sample_ids for {key} must be strings")
        if len(set(raw_overrides)) != len(raw_overrides) or not set(raw_overrides) <= available:
            raise ValueError(f"{key} has duplicate or foreign ReVOS preview override IDs")
        overrides = [sample_id for sample_id in raw_overrides if sample_id not in base_preview_set]
        preview = [*base_preview, *overrides]
        video["review_mode"] = REVOS_PREVIEW_PROTOCOL
        video["preview_override_sample_ids"] = overrides
        video["preview_sample_ids"] = preview
        edits = video.setdefault("instructions", {})
        for sample_id in preview:
            edit = edits.setdefault(sample_id, {})
            if edit.get("status") not in {"accepted", "rejected"}:
                edit["status"] = (
                    "rejected"
                    if edit.get("discarded")
                    else "accepted"
                )
        accepted = [
            sample_id
            for sample_id in preview
            if edits[sample_id].get("status") == "accepted"
            and not edits[sample_id].get("discarded")
        ]
        video["accepted_sample_ids"] = accepted
        video.pop("sampled_sample_id", None)

    selected = _validate_selections(rows, prepared)
    for key, video in videos.items():
        if video.get("review_mode") == REVOS_PREVIEW_PROTOCOL:
            continue
        if video.get("status") != "accepted":
            video.pop("sampled_sample_id", None)
            continue
        candidates = _accepted_ids(grouped.get(key, []), video, selected, key)
        if not candidates:
            video.pop("sampled_sample_id", None)
            continue
        eligible = set(_eligible_ids(grouped[key], video))
        archived = set(video.get("accepted_sample_ids", [])) & eligible
        if "selected_sample_ids" not in video and "accepted_sample_ids" not in video:
            archived.update(candidates)
        archived.update(selected.get(key, set()))
        video["accepted_sample_ids"] = [
            sample_id for sample_id in _suggestion_order(grouped[key], key) if sample_id in archived
        ]
        video["sampled_sample_id"] = candidates[0]
    return prepared


def apply_decisions(
    rows: list[dict[str, Any]], decisions: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply decisions, including the ReVOS multi-instruction preview adapter."""

    prepared = _materialize_sampling(rows, decisions)
    video_decisions = prepared.get("videos", {})
    selected_by_video = _validate_selections(rows, prepared)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_video_key(row)].append(row)
    export_ids = {
        key: {accepted[0]}
        for key, video in video_decisions.items()
        if video.get("review_mode") != REVOS_PREVIEW_PROTOCOL
        if (accepted := _accepted_ids(grouped.get(key, []), video, selected_by_video, key))
    }
    for key, video in video_decisions.items():
        if video.get("review_mode") != REVOS_PREVIEW_PROTOCOL:
            continue
        edits = video.get("instructions", {})
        export_ids[key] = (
            {
                sample_id
                for sample_id in video.get("preview_sample_ids", [])
                if edits.get(sample_id, {}).get("status") == "accepted"
                and not edits.get(sample_id, {}).get("discarded")
            }
            if video.get("status") == "accepted"
            else set()
        )
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        key = _video_key(row)
        video = video_decisions.get(key, {})
        status = video.get("status", "undecided")
        if row["sample_id"] not in export_ids.get(key, set()):
            continue
        edit = video.get("instructions", {}).get(row["sample_id"], {})
        text = str(edit.get("text", row["text"]))
        tracklets = _loads(row["tracklets_json"], [])
        removed = {str(value) for value in edit.get("deleted_tracklet_ids", [])}
        tracklets = [
            tracklet for tracklet in tracklets if str(tracklet["tracklet_id"]) not in removed
        ]
        existing_ids = {str(tracklet["tracklet_id"]) for tracklet in tracklets}
        groups = edit.get("groups", _loads(row["groups_json"], []))
        normalized_groups = []
        for group in groups:
            ids = [str(value) for value in group.get("tracklet_ids", []) if str(value) in existing_ids]
            spans = []
            for span in group.get("text_spans", []):
                start, end = int(span["start"]), int(span["end"])
                if 0 <= start < end <= len(text) and text[start:end] == span.get("text"):
                    spans.append({"start": start, "end": end, "text": text[start:end]})
            if ids and spans:
                normalized_groups.append(
                    {
                        "group_id": str(group.get("group_id", f"manual-{len(normalized_groups)+1:03d}")),
                        "role": str(group.get("role", "manual_link")),
                        "identity": str(group.get("identity", "/".join(sorted(ids)))),
                        "text_spans": spans,
                        "tracklet_ids": sorted(set(ids)),
                    }
                )
        row["text"] = text
        row["tracklets_json"] = _dumps(tracklets)
        row["groups_json"] = _dumps(normalized_groups)
        row["span_links_json"] = _dumps(rebuild_span_links(normalized_groups))
        row["verification_status"] = str(edit.get("status", status))
        row["verification_decision_json"] = _dumps(edit)
        output.append(row)
    return output


def write_verified_parquet(
    rows: list[dict[str, Any]], decisions: dict[str, Any], output: Path
) -> bytes:
    edited = apply_decisions(rows, decisions)
    if edited:
        table = pa.Table.from_pylist(edited)
    elif rows:
        template = dict(rows[0])
        template["verification_status"] = "undecided"
        template["verification_decision_json"] = "{}"
        table = pa.Table.from_pylist([template]).slice(0, 0)
    else:
        raise ValueError("cannot export an empty verification dataset")
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd")
    payload = sink.getvalue()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.part")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    return payload


class VerificationState:
    def __init__(
        self,
        parquet_paths: list[Path],
        decisions_path: Path | None,
        media_roots: list[Path],
        output_path: Path,
    ) -> None:
        self.rows = load_verification_rows(parquet_paths)
        self.fingerprint = dataset_fingerprint(self.rows)
        self.output_path = output_path
        self.decisions_path = decisions_path or output_path.with_name("decisions.json")
        self.media_roots = [root.expanduser() for root in media_roots]
        self.row_by_sample = {str(row["sample_id"]): row for row in self.rows}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            grouped[_video_key(row)].append(row)
        self.videos = sorted(grouped.items())
        initial = {}
        if decisions_path and decisions_path.is_file():
            initial = json.loads(decisions_path.read_text(encoding="utf-8"))
        self.decisions = _materialize_sampling(
            self.rows, _normalize_decisions(initial, self.fingerprint)
        )
        if not self.rows:
            raise ValueError("verification input contains no instructions")
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self._tar_archives: dict[Path, IndexedTarReader] = {}
        self._archive_names: dict[Path, set[str]] = {}
        self._archive_lock = threading.Lock()
        self._decisions_lock = threading.Lock()

    def save_decisions(self, value: dict[str, Any]) -> dict[str, Any]:
        decisions = _materialize_sampling(
            self.rows, _normalize_decisions(value, self.fingerprint)
        )
        decisions["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(decisions, ensure_ascii=False, indent=2).encode("utf-8")
        with self._decisions_lock:
            self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.decisions_path.with_suffix(
                self.decisions_path.suffix + f".{os.getpid()}.part"
            )
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.decisions_path)
            self.decisions = decisions
        return decisions

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "concor-video-verifier-api-v1",
            "source_fingerprint": self.fingerprint,
            "video_count": len(self.videos),
            "instruction_count": len(self.rows),
            "storage": {
                "decisions_path": str(self.decisions_path),
                "output_path": str(self.output_path),
                "browser_storage_key": f"concor-video:{self.fingerprint}",
            },
            "videos": [
                {
                    "index": index,
                    "video_key": key,
                    "dataset": rows[0]["dataset"],
                    "split": rows[0]["split"],
                    "video_id": rows[0]["video_id"],
                    "instruction_count": len(rows),
                }
                for index, (key, rows) in enumerate(self.videos)
            ],
            "initial_decisions": self.decisions,
        }

    def video(self, index: int) -> dict[str, Any]:
        key, rows = self.videos[index]
        return {
            "index": index,
            "video_key": key,
            "dataset": rows[0]["dataset"],
            "split": rows[0]["split"],
            "video_id": rows[0]["video_id"],
            "suggestion_order": _suggestion_order(rows, key),
            "review_mode": REVOS_PREVIEW_PROTOCOL if _is_revos_video(rows) else None,
            "preview_order": _revos_preview_order(rows, key),
            "instructions": [_public_row(row) for row in rows],
        }

    @staticmethod
    def _member_candidates(row: dict[str, Any], frame_id: str) -> tuple[str, ...]:
        video_id = str(row["video_id"])
        return tuple(
            f"{prefix}{video_id}/{frame_id}{suffix}"
            for prefix in (
                "", "JPEGImages/", "train/JPEGImages/", "valid/JPEGImages/",
                "val/JPEGImages/", "test/JPEGImages/", "ReVOS/JPEGImages/",
            )
            for suffix in (".jpg", ".jpeg", ".png")
        )

    def frame(self, sample_id: str, frame_index: int) -> tuple[bytes, str]:
        row = self.row_by_sample[sample_id]
        frame_ids = _loads(row["frame_ids_json"], [])
        if not 0 <= frame_index < len(frame_ids):
            raise IndexError(frame_index)
        frame_id = str(frame_ids[frame_index])
        candidates = self._member_candidates(row, frame_id)
        sources = _candidate_media_sources(self.media_roots, row)
        for source in sources:
            if source.is_dir():
                for relative in candidates:
                    path = source / relative
                    if path.is_file():
                        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/jpeg"
            elif source.is_file() and source.suffix.lower() == ".zip":
                with self._archive_lock:
                    archive = self._archives.setdefault(source, zipfile.ZipFile(source))
                    names = set(archive.namelist())
                    member = next((name for name in candidates if name in names), None)
                    if member:
                        return archive.read(member), mimetypes.guess_type(member)[0] or "image/jpeg"
            elif source.is_file() and source.suffix.lower() == ".tar":
                with self._archive_lock:
                    archive = self._tar_archives.get(source)
                    if archive is None:
                        archive = IndexedTarReader(source, auto_reindex=True)
                        self._tar_archives[source] = archive
                    try:
                        member, payload = archive.read(candidates)
                    except FileNotFoundError:
                        continue
                    return payload, mimetypes.guess_type(member)[0] or "image/jpeg"
        raise FileNotFoundError(f"frame {frame_id} for {sample_id} was not found")


def _static_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "verification" / "static"
    if not root.is_dir():
        raise FileNotFoundError(f"verification static files are missing: {root}")
    return root


def _handler(state: VerificationState):
    static_root = _static_root()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ConCorVideoVerifier/1.0"

        def _json(self, value: Any, status: int = 200) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status=status)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path == "/api/summary":
                    self._json(state.summary())
                    return
                if parsed.path == "/api/video":
                    self._json(state.video(int(query.get("index", ["0"])[0])))
                    return
                if parsed.path == "/api/frame":
                    payload, content_type = state.frame(
                        query["sample_id"][0], int(query.get("frame", ["0"])[0])
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "private, max-age=3600")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                relative = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
                path = (static_root / relative).resolve()
                if static_root not in path.parents and path != static_root:
                    self._error(403, "invalid path")
                    return
                payload = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (KeyError, IndexError, ValueError) as error:
                self._error(400, str(error))
            except FileNotFoundError as error:
                self._error(404, str(error))
            except RuntimeError as error:
                self._error(500, str(error))

        def do_POST(self) -> None:
            endpoint = urllib.parse.urlparse(self.path).path
            if endpoint not in {"/api/decisions", "/api/export"}:
                self._error(404, "unknown endpoint")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 64 * 1024 * 1024:
                    raise ValueError("decisions payload is too large")
                decisions = state.save_decisions(json.loads(self.rfile.read(length)))
                if endpoint == "/api/decisions":
                    self._json(
                        {
                            "saved_at": decisions["updated_at"],
                            "path": str(state.decisions_path),
                            "decisions": decisions,
                        }
                    )
                    return
                payload = write_verified_parquet(state.rows, decisions, state.output_path)
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.apache.parquet")
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{state.output_path.name}"'
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, json.JSONDecodeError) as error:
                self._error(400, str(error))

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[verification] {self.address_string()} {format % args}")

    return Handler


def serve_verification(
    *,
    parquet_paths: list[Path],
    decisions_path: Path | None,
    media_roots: list[Path],
    host: str,
    port: int,
    output_path: Path,
) -> None:
    state = VerificationState(parquet_paths, decisions_path, media_roots, output_path)
    server = ThreadingHTTPServer((host, port), _handler(state))
    print(
        f"ConCor video verification: http://{host}:{port} "
        f"({len(state.videos)} videos, {len(state.rows)} instructions)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
