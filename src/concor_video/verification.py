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
    return {
        "schema_version": DECISIONS_VERSION,
        "source_fingerprint": fingerprint,
        "updated_at": value.get("updated_at"),
        "cursor": value.get("cursor", {"video_index": 0}),
        "quick_keys_enabled": bool(value.get("quick_keys_enabled", True)),
        "videos": value.get("videos", {}),
    }


def apply_decisions(
    rows: list[dict[str, Any]], decisions: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply edits deterministically; rejected videos are omitted."""

    video_decisions = decisions.get("videos", {})
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        video = video_decisions.get(_video_key(row), {})
        status = video.get("status", "undecided")
        if status == "rejected":
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
        row["verification_status"] = status
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
        self.media_roots = media_roots
        self.row_by_sample = {str(row["sample_id"]): row for row in self.rows}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            grouped[_video_key(row)].append(row)
        self.videos = sorted(grouped.items())
        initial = {}
        if decisions_path and decisions_path.is_file():
            initial = json.loads(decisions_path.read_text(encoding="utf-8"))
        self.decisions = _normalize_decisions(initial, self.fingerprint)
        if not self.rows:
            raise ValueError("verification input contains no instructions")
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self._tar_archives: dict[Path, IndexedTarReader] = {}
        self._archive_names: dict[Path, set[str]] = {}
        self._archive_lock = threading.Lock()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "concor-video-verifier-api-v1",
            "source_fingerprint": self.fingerprint,
            "video_count": len(self.videos),
            "instruction_count": len(self.rows),
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
        sources = self.media_roots + [
            Path(value) for value in (row.get("frame_source"), row.get("dataset_root")) if value
        ]
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
                        archive = IndexedTarReader(source)
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

        def do_POST(self) -> None:
            if urllib.parse.urlparse(self.path).path != "/api/export":
                self._error(404, "unknown endpoint")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 64 * 1024 * 1024:
                    raise ValueError("decisions payload is too large")
                decisions = _normalize_decisions(
                    json.loads(self.rfile.read(length)), state.fingerprint
                )
                decisions["updated_at"] = datetime.now(timezone.utc).isoformat()
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
