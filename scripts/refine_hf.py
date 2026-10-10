#!/usr/bin/env python3
"""Refine downloaded ConCor Parquets using shared VIPOSeg-style video packets.

Run from a checkout; no SAM, GPU, or spaCy model is required. Credentials stay in
process memory. Output annotation files preserve the input Arrow schema exactly.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
from copy import deepcopy
import gc
import getpass
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from concor_video.rle import decode_rle
from concor_video.tar_index import IndexedTarReader, index_tar
from concor_video.tracklet_schema import SCHEMA_VERSION, rebuild_span_links, validate_record
from concor_video.verification import VerificationState, _candidate_media_sources

MODEL = "gpt-6-luna"
VERSION = "concor-refine-v1"
API = "https://api.openai.com/v1"
META = ["sample_id", "dataset", "split", "video_id", "cohort", "text", "negative", "groups_json", "disposition"]
TAG = re.compile(r"\[(/?)([1-9][0-9]*)\]")
RESULT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["results"],
    "properties": {"results": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["sample_id", "decision", "tagged_caption", "reason_codes"],
        "properties": {
            "sample_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["keep", "rewrite", "reject"]},
            "tagged_caption": {"type": "string"},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
        },
    }}},
}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def loads(value, default=None):
    return json.loads(value) if isinstance(value, str) and value else (value if value is not None else default)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(dumps(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_prompt(path):
    content = Path(path).read_text(encoding="utf-8")
    if not content.startswith("---\n") or "\n---\n" not in content[4:]:
        raise ValueError("prompt must have documentation-only front matter between --- lines")
    metadata, prompt = content[4:].split("\n---\n", 1)
    if not prompt.strip():
        raise ValueError("empty model prompt")
    return metadata, prompt.strip()


def discover_inputs(path):
    path = Path(path).resolve()
    if path.is_file():
        return [path]
    # Never accidentally combine the manually verified subset with its originals.
    standard = [path / "train/verification.parquet", path / "eval/unfiltered/verification.parquet"]
    paths = [p for p in standard if p.is_file()]
    if paths:
        return paths
    paths = sorted(path.glob("*.parquet"))
    if not paths:
        raise ValueError("input must be a Parquet, a split directory, or the downloaded HF root")
    return paths


class Source:
    """Small metadata index; masks are loaded from only the required row groups."""
    def __init__(self, paths, include_media=False):
        self.paths = paths
        self.files = [pq.ParquetFile(p) for p in paths]
        self.schema = self.files[0].schema_arrow
        self.rows, self.locations = {}, {}
        for file_index, parquet in enumerate(self.files):
            if not parquet.schema_arrow.equals(self.schema, check_metadata=True):
                raise ValueError("input schemas differ; refine those Parquets in separate output directories")
            missing = set(META + ["frame_ids_json", "tracklets_json", "span_links_json"]) - set(self.schema.names)
            if missing:
                raise ValueError(f"missing columns: {sorted(missing)}")
            for group_index in range(parquet.num_row_groups):
                columns = META + ([c for c in ("frame_source", "dataset_root") if c in self.schema.names] if include_media else [])
                for offset, row in enumerate(parquet.read_row_group(group_index, columns=columns).to_pylist()):
                    sid = row["sample_id"]
                    if sid in self.rows:
                        raise ValueError(f"duplicate sample_id {sid}; do not combine raw and verified copies")
                    self.rows[sid] = row
                    self.locations[sid] = file_index, group_index, offset

    @lru_cache(maxsize=2)
    def row_group(self, file_index, group_index):
        return self.files[file_index].read_row_group(group_index).to_pylist()

    def row(self, sid):
        file_index, group_index, offset = self.locations[sid]
        return self.row_group(file_index, group_index)[offset]


def release_memory():
    """Return large temporary Arrow/RLE/render buffers on constrained shared hosts."""
    Source.row_group.cache_clear()
    gc.collect()
    pa.default_memory_pool().release_unused()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass


def video_key(row):
    return row["dataset"], row["split"], row["video_id"]


def select_ids(source, per_dataset, seed):
    if not per_dataset:
        return list(source.rows)
    by_dataset = defaultdict(list)
    for row in source.rows.values():
        by_dataset[row["dataset"]].append(row)
    selected = []
    for dataset, rows in sorted(by_dataset.items()):
        chosen = []
        videos = defaultdict(list)
        for row in rows:
            videos[video_key(row)].append(row)
        order = sorted(videos, key=lambda key: digest([seed, key]))
        # Include a nonexistent example when available, then related expressions
        # so the pilot exercises shared-input grouping as well as negative handling.
        negative = sorted((r for r in rows if r["negative"]), key=lambda r: digest([seed, r["sample_id"]]))
        if negative:
            chosen.append(negative[0]["sample_id"])
            first = video_key(negative[0])
            order = [first] + [key for key in order if key != first]
        for key in order:
            candidates = sorted(videos[key], key=lambda r: digest([seed, r["sample_id"]]))
            # Two expressions per video give both breadth and actual shared packets.
            added = 0
            for row in candidates:
                if len(chosen) >= per_dataset or added >= 2:
                    break
                if row["sample_id"] not in chosen:
                    chosen.append(row["sample_id"])
                    added += 1
            if len(chosen) >= per_dataset:
                break
        selected.extend(chosen)
    return selected


def main_ids(row):
    return {tid for group in loads(row["groups_json"], []) if group.get("role") == "main_referent"
            for tid in group.get("tracklet_ids", [])}


def track_signature(frame_ids, track):
    # Confidence, prompt names, and IDs are not evidence of identity; the complete
    # frame-aligned mask sequence must be exactly the same to share a visual ID.
    return digest([frame_ids, track["masks"]])


def visible_indices(track):
    return [index for index, mask in enumerate(track["masks"])
            if mask and sum(mask["counts"][1::2]) > 0]


def inspect_row(row, max_tracks):
    """Only inexpensive structural/noise checks; no language guessing or IoU merge."""
    frames = loads(row["frame_ids_json"], [])
    tracks = loads(row["tracklets_json"], [])
    if not frames or len(set(frames)) != len(frames):
        return None, "invalid_frame_sequence"
    if row["negative"]:
        return ([], {}), None
    mains = main_ids(row)
    if not mains:
        return None, "missing_main_referent"
    seen_ids, seen_masks, kept, aliases = set(), {}, [], {}
    # Prefer the original main track when an exact duplicate context track exists.
    for track in sorted(tracks, key=lambda t: t["tracklet_id"] not in mains):
        tid = track["tracklet_id"]
        if tid in seen_ids or len(track.get("masks", [])) != len(frames):
            return None, "invalid_tracklet_alignment"
        seen_ids.add(tid)
        for mask in track["masks"]:
            if mask is None:
                continue
            counts, size = mask.get("counts"), mask.get("size", [])
            if (len(size) != 2 or any(not isinstance(x, int) or x <= 0 for x in size)
                or not isinstance(counts, list) or any(not isinstance(x, int) or x < 0 for x in counts)
                or sum(counts) != size[0] * size[1]):
                return None, "invalid_mask_rle"
        if not visible_indices(track):
            if tid in mains:
                return None, "empty_main_referent"
            continue
        signature = track_signature(frames, track)
        if signature in seen_masks:
            aliases[tid] = seen_masks[signature]
        else:
            seen_masks[signature] = tid
            kept.append(track)
    if not mains.issubset(seen_ids):
        return None, "missing_main_referent"
    if len(kept) > max_tracks:
        return None, "excessive_tracklets"
    return (kept, aliases), None


def sample_frames(frame_count, tracks, count=12):
    selected = set(np.linspace(0, frame_count - 1, min(count, frame_count), dtype=int).tolist())
    for track in tracks:
        visible = visible_indices(track)
        if visible:
            selected.add(visible[0])
    return sorted(selected)


class Media:
    def __init__(self, roots, cache):
        self.roots, self.cache = roots, cache
        self.zips, self.tars = {}, {}

    def frame(self, row, frame_id):
        names = VerificationState._member_candidates(row, frame_id)
        for path in _candidate_media_sources(self.roots, row):
            if path.is_dir():
                for name in names:
                    candidate = path / name
                    if candidate.is_file():
                        return candidate.read_bytes()
            elif path.is_file() and path.suffix == ".zip":
                if path not in self.zips:
                    archive = zipfile.ZipFile(path)
                    self.zips[path] = archive, set(archive.namelist())
                archive, members = self.zips[path]
                found = next((name for name in names if name in members), None)
                if found:
                    return archive.read(found)
            elif path.is_file() and path.suffix == ".tar":
                if path not in self.tars:
                    try:
                        self.tars[path] = IndexedTarReader(path)
                    except RuntimeError:
                        index = self.cache / (digest(str(path))[:16] + ".sqlite")
                        index_tar(path, index)
                        self.tars[path] = IndexedTarReader(path, index)
                try:
                    return self.tars[path].read(names)[1]
                except FileNotFoundError:
                    continue
        raise FileNotFoundError(f"missing local frame {frame_id} for {row['sample_id']}")

    def close(self):
        for archive, _ in self.zips.values():
            archive.close()
        for archive in self.tars.values():
            archive.close()


def color_for(track_id):
    hsv = np.uint8([[[(track_id * 47) % 180, 210, 235]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()


def render_pair(payload, tracks, frame_index, frame_id):
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unreadable source frame")
    overlay = image.copy()
    outlines = []
    for visual_id, track in sorted(tracks.items()):
        mask = decode_rle(track["masks"][frame_index])
        if mask is None or not mask.any():
            continue
        if mask.shape != image.shape[:2]:
            raise ValueError("mask size differs from original frame; refusing to resize annotations")
        color = color_for(visual_id)
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.asarray(color)).astype(np.uint8)
        outlines.append((visual_id, mask, color))
    for visual_id, mask, color in outlines:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        ys, xs = np.where(mask)
        position = (max(0, min(int(np.median(xs)), image.shape[1] - 40)), max(22, int(np.median(ys))))
        cv2.putText(overlay, str(visual_id), position, cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, str(visual_id), position, cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
    height = min(image.shape[0], 540)
    width = round(image.shape[1] * height / image.shape[0])
    pair = np.concatenate([cv2.resize(image, (width, height)), cv2.resize(overlay, (width, height))], axis=1)
    label = f"original | overlay | source_frame={frame_id}"
    cv2.putText(pair, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(pair, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    success, jpeg = cv2.imencode(".jpg", pair, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise ValueError("JPEG encoding failed")
    return jpeg.tobytes(), [pair.shape[1], pair.shape[0]]


def parse_tags(tagged, allowed):
    stack, mentions, chunks, position, cursor = [], defaultdict(list), [], 0, 0
    for match in TAG.finditer(tagged):
        chunk = tagged[cursor:match.start()]
        chunks.append(chunk)
        position += len(chunk)
        closing, tid = bool(match[1]), int(match[2])
        if tid not in allowed:
            raise ValueError("unknown or disallowed tracklet ID")
        if not closing:
            if any(open_id == tid for open_id, _ in stack):
                raise ValueError("self-overlapping tags")
            stack.append((tid, position))
        else:
            if not stack or stack[-1][0] != tid:
                raise ValueError("crossed or unbalanced tags")
            _, start = stack.pop()
            if start == position:
                raise ValueError("empty tagged span")
            mentions[tid].append((start, position))
        cursor = match.end()
    chunks.append(tagged[cursor:])
    text = "".join(chunks)
    if stack or re.search(r"\[/?[^\]\n]+\]", text) or not text.strip() or text != text.strip():
        raise ValueError("unbalanced tags, unsupported annotation markup, or empty/outer-spaced caption")
    for spans in mentions.values():
        for start, end in spans:
            if text[start:end].casefold() == "facing":
                raise ValueError("verb-only link: facing cannot be an entity span")
            if text[start:end] != text[start:end].strip():
                raise ValueError("span contains surrounding whitespace")
            if start and text[start - 1].isalnum() and text[start].isalnum():
                raise ValueError("span starts inside a token")
            if end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
                raise ValueError("span ends inside a token")
    return text, mentions


def apply_result(row, result, mapping):
    if "nlp_groups" in mapping:
        return apply_nlp_result(row, result, mapping)
    if result["decision"] == "reject":
        if result["tagged_caption"]:
            raise ValueError("rejected result must have empty caption")
        return None
    if row["negative"]:
        if result["decision"] != "keep" or result["tagged_caption"] != row["text"]:
            raise ValueError("intentional negative must retain its exact original text")
        output = dict(row)
        output.update(tracklets_json="[]", groups_json="[]", span_links_json="[]", disposition="negative_unsegmentable")
        return output
    visual_to_local = {int(k): v for k, v in mapping["visual_to_local"].items()}
    text, mentions = parse_tags(result["tagged_caption"], set(visual_to_local))
    if not set(mapping["main_ids"]).issubset(mentions):
        raise ValueError("final expression dropped the intended main referent")
    tracks = {t["tracklet_id"]: t for t in loads(row["tracklets_json"], [])}
    old_groups = loads(row["groups_json"], [])
    retained, groups = [], []
    for visual_id, spans in sorted(mentions.items()):
        tid = visual_to_local[visual_id]
        retained.append(tracks[tid])
        old = next((g for g in old_groups if tid in g.get("tracklet_ids", [])), {})
        groups.append({
            "group_id": f"refined-{visual_id:03d}",
            "role": "main_referent" if visual_id in mapping["main_ids"] else "context_entity",
            "identity": old.get("identity", tracks[tid].get("sam_prompt") or ""),
            "tracklet_ids": [tid],
            "text_spans": [{"start": a, "end": b, "text": text[a:b]} for a, b in sorted(spans)],
        })
    links = rebuild_span_links(groups)
    record = {"schema_version": SCHEMA_VERSION, "text": text, "negative": False,
              "frame_ids": loads(row["frame_ids_json"]), "frame_files": loads(row.get("frame_files_json"), []),
              "tracklets": retained, "groups": groups, "span_links": links, "disposition": "complete_bcc"}
    validate_record(record)
    output = dict(row)
    output.update(text=text, tracklets_json=dumps(retained), groups_json=dumps(groups),
                  span_links_json=dumps(links), disposition="complete_bcc")
    # Extraction/SAM fields intentionally remain the provenance of the source run.
    # Refinement decisions belong in the sidecar, not in extra Parquet columns.
    return output


def apply_nlp_result(row, result, mapping):
    """Repair language links without choosing among an entity's mask candidates."""
    if result["decision"] not in {"keep", "rewrite"}:
        raise ValueError("text-only cleanup cannot semantically reject an example")
    if row["negative"]:
        if result["decision"] != "keep" or result["tagged_caption"] != row["text"]:
            raise ValueError("NLP cleanup must preserve intentional negatives without links")
        output = dict(row)
        output.update(groups_json="[]", span_links_json="[]", tracklets_json="[]")
        return output
    groups_by_id = {int(k): v for k, v in mapping["nlp_groups"].items()}
    text, mentions = parse_tags(result["tagged_caption"], set(groups_by_id))
    if text != row["text"]:
        raise ValueError("NLP cleanup must preserve the exact original text")
    if not set(mapping["main_ids"]).issubset(mentions):
        raise ValueError("NLP cleanup dropped the intended main referent")
    groups = []
    for gid, spans in sorted(mentions.items()):
        group = deepcopy(groups_by_id[gid])
        group["text_spans"] = [{"start": a, "end": b, "text": text[a:b]} for a, b in sorted(spans)]
        groups.append(group)
    retained_ids = {tid for g in groups for tid in g["tracklet_ids"]}
    output = dict(row)
    output.update(groups_json=dumps(groups), span_links_json=dumps(rebuild_span_links(groups)),
                  tracklets_json=dumps([t for t in loads(row["tracklets_json"], []) if t["tracklet_id"] in retained_ids]))
    # This is an intermediate linguistic audit, not a complete/visually verified BCC.
    return output


def prepare_nlp(source, ids, args, prompt, output):
    """Text-only packets merge identical linguistic groups, never mask instances."""
    packets, early, entries, mappings = [], {}, [], {}
    for sid in ids:
        row = source.row(sid)
        inspection, reason = inspect_row(row, args.max_tracklets)
        if reason:
            early[sid] = {"sample_id": sid, "decision": "reject", "tagged_caption": "",
                          "reason_codes": [reason], "stage": "heuristic"}
            continue
        kept, aliases = inspection
        allowed = {t["tracklet_id"] for t in kept}
        merged = {}
        if not row["negative"]:
            for group in loads(row["groups_json"], []):
                tids = {aliases.get(t, t) for t in group.get("tracklet_ids", [])} & allowed
                if not tids:
                    continue
                signature = digest([group.get("role"), group.get("identity"), group.get("text_spans", [])])
                if signature not in merged:
                    merged[signature] = {**deepcopy(group), "tracklet_ids": []}
                merged[signature]["tracklet_ids"] = sorted(set(merged[signature]["tracklet_ids"]) | tids)
        groups = {i: {**g, "group_id": f"nlp-{i:03d}"} for i, g in enumerate(merged.values(), 1)}
        mains = [i for i, g in groups.items() if g["role"] == "main_referent"]
        mappings[sid] = {"nlp_groups": groups, "main_ids": mains}
        entries.append({"sample_id": sid, "existing_text": row["text"], "negative": row["negative"],
                        "allowed_ids": list(groups), "main_ids": mains,
                        "existing_groups": [{"id": i, "role": g["role"], "identity": g.get("identity", ""),
                                             "text_spans": g.get("text_spans", []),
                                             "candidate_count": len(g["tracklet_ids"])} for i, g in groups.items()]})
    for start in range(0, len(entries), args.group_size):
        chunk = entries[start:start + args.group_size]
        packet = {"stage": "nlp", "expressions": chunk, "frame_ids": [], "selected_frame_indices": []}
        rid = digest([VERSION, "nlp-v1", prompt, packet])[:24]
        manifest = {"request_id": rid, "stage": "nlp", "packet": packet, "images": [], "dimensions": [],
                    "mapping": {e["sample_id"]: mappings[e["sample_id"]] for e in chunk},
                    "max_output_tokens": min(16384, 4096 + 768 * len(chunk))}
        atomic_json(output / "packets" / rid / "input.json", manifest)
        packets.append(manifest)
    atomic_json(output / "requests.json", [p["request_id"] for p in packets])
    atomic_json(output / "heuristics.json", early)
    return packets, early


def prepare(source, ids, args, prompt, output):
    by_video = defaultdict(list)
    for sid in ids:
        by_video[video_key(source.rows[sid])].append(sid)
    media = Media(args.media_root, output / "media-index")
    packets, early = [], {}
    try:
        for number, (key, sids) in enumerate(sorted(by_video.items()), 1):
            registry = dict(source.visual_registry(key)) if hasattr(source, "visual_registry") else {}
            tracks, prepared = {}, []
            frames_key = None
            for sid in sorted(sids):
                row = source.row(sid)
                inspection, reason = inspect_row(row, args.max_tracklets)
                if reason:
                    early[sid] = {"sample_id": sid, "decision": "reject", "tagged_caption": "", "reason_codes": [reason], "stage": "heuristic"}
                    continue
                row_frames = loads(row["frame_ids_json"])
                frame_hash = digest(row_frames)
                if frames_key is not None and frame_hash != frames_key:
                    raise ValueError(f"inconsistent full-video frame sequence for {key}")
                frames_key = frame_hash
                kept, aliases = inspection
                local_to_visual, visual_to_local = {}, {}
                mains = main_ids(row)
                for track in kept:
                    signature = track_signature(row_frames, track)
                    if signature not in registry:
                        registry[signature] = len(registry) + 1
                    vid = registry[signature]
                    tracks[vid] = track
                    local_to_visual[track["tracklet_id"]] = vid
                    visual_to_local[vid] = track["tracklet_id"]
                for alias, original in aliases.items():
                    local_to_visual[alias] = local_to_visual[original]
                groups = []
                for group in loads(row["groups_json"], []):
                    groups.append({"role": group.get("role"), "identity": group.get("identity"),
                                   "ids": sorted({local_to_visual[t] for t in group.get("tracklet_ids", []) if t in local_to_visual}),
                                   "text_spans": group.get("text_spans", [])})
                entry = {"sample_id": sid, "existing_text": row["text"], "negative": row["negative"],
                         "allowed_ids": sorted(visual_to_local),
                         "main_ids": sorted({local_to_visual[t] for t in mains if t in local_to_visual}),
                         "target_source": row.get("target_source"), "existing_groups": groups}
                prepared.append((entry, {"visual_to_local": visual_to_local, "local_to_visual": local_to_visual,
                                         "main_ids": entry["main_ids"]}))
            chunks, chunk, union = [], [], set()
            for item in prepared:
                new_ids = set(item[0]["allowed_ids"])
                if chunk and (len(chunk) >= args.group_size or len(union | new_ids) > args.group_tracklets):
                    chunks.append(chunk)
                    chunk, union = [], set()
                chunk.append(item)
                union.update(new_ids)
            if chunk:
                chunks.append(chunk)
            for chunk in chunks:
                first = source.row(chunk[0][0]["sample_id"])
                frames = loads(first["frame_ids_json"])
                used = sorted({tid for entry, _ in chunk for tid in entry["allowed_ids"]})
                subset = {tid: tracks[tid] for tid in used}
                selected = sample_frames(len(frames), list(subset.values()), args.sample_count)
                inventory = [{"id": tid, "source": tracks[tid].get("source"),
                              "proposal": tracks[tid].get("sam_prompt"),
                              "visible_frame_indices": visible_indices(tracks[tid])} for tid in used]
                packet = {"video": list(key), "frame_ids": frames, "selected_frame_indices": selected,
                          "tracklets": inventory, "expressions": [entry for entry, _ in chunk]}
                request_id = digest([VERSION, prompt, packet, [track_signature(frames, t) for t in subset.values()]])[:24]
                folder = output / "packets" / request_id
                folder.mkdir(parents=True, exist_ok=True)
                image_paths, dimensions = [], []
                try:
                    for index in selected:
                        path = folder / f"frame_{index:05d}.jpg"
                        if path.is_file():
                            image = cv2.imread(str(path))
                            size = [image.shape[1], image.shape[0]]
                        else:
                            cached = source.cached_pair(key, used, index) if hasattr(source, "cached_pair") else None
                            if cached:
                                cached_path, size = cached
                                payload = cached_path.read_bytes()
                            else:
                                payload, size = render_pair(media.frame(first, str(frames[index])), subset, index, str(frames[index]))
                            temporary = path.with_suffix(".jpg.part")
                            temporary.write_bytes(payload)
                            temporary.replace(path)
                        image_paths.append(str(path.relative_to(output)))
                        dimensions.append(size)
                except (OSError, ValueError, RuntimeError) as exc:
                    for entry, _ in chunk:
                        early[entry["sample_id"]] = {"sample_id": entry["sample_id"], "decision": "error", "reason_codes": [type(exc).__name__, str(exc)], "stage": "media"}
                    continue
                manifest = {"request_id": request_id, "packet": packet, "images": image_paths,
                            "dimensions": dimensions, "mapping": {entry["sample_id"]: mapping for entry, mapping in chunk},
                            "max_output_tokens": min(16384, 4096 + 768 * len(chunk))}
                atomic_json(folder / "input.json", manifest)
                packets.append(manifest)
            if number % 25 == 0 or number == len(by_video):
                print(dumps({"event": "prepared", "videos": number, "total_videos": len(by_video), "requests": len(packets)}), flush=True)
    finally:
        media.close()
    atomic_json(output / "heuristics.json", early)
    atomic_json(output / "requests.json", [p["request_id"] for p in packets])
    return packets, early


def request_body(manifest, prompt, output):
    content = []
    selected = manifest["packet"]["selected_frame_indices"]
    for path, index in zip(manifest["images"], selected):
        content.append({"type": "input_text", "text": f"Source frame {manifest['packet']['frame_ids'][index]}: left=original; right=candidate tracklet overlay."})
        encoded = base64.b64encode((output / path).read_bytes()).decode("ascii")
        content.append({"type": "input_image", "detail": "high", "image_url": "data:image/jpeg;base64," + encoded})
    content.append({"type": "input_text", "text": dumps(manifest["packet"])})
    if manifest.get("validation_retry_of"):
        prompt += "\nYour previous output failed local validation. Correct the listed validation_errors while following every original task constraint."
    return {"model": MODEL, "reasoning": {"effort": "medium"}, "store": False,
            "instructions": prompt, "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "concor_video_refinement", "strict": True, "schema": response_schema(manifest)}},
            "max_output_tokens": manifest["max_output_tokens"]}


def response_schema(manifest):
    # Constrain tag permissions PER expression, including shared-video calls.
    # Semantic choice and balanced nesting still need the prompt/local validator.
    schema = deepcopy(RESULT_SCHEMA)
    prototype = schema["properties"]["results"]["items"]
    variants = []
    for expression in manifest["packet"]["expressions"]:
        item = deepcopy(prototype)
        properties = item["properties"]
        if manifest.get("stage") == "nlp":
            properties["decision"]["enum"] = ["keep", "rewrite"]
        properties["sample_id"]["enum"] = [expression["sample_id"]]
        if expression["negative"]:
            properties["tagged_caption"]["enum"] = ["", expression["existing_text"]]
        else:
            alternatives = "|".join(str(tid) for tid in expression["allowed_ids"])
            properties["tagged_caption"]["pattern"] = rf"^(?:[^\[\]]|\[/?(?:{alternatives})\])*$"
        variants.append(item)
    schema["properties"]["results"]["items"] = {"anyOf": variants}
    return schema


def input_token_bound(manifest, prompt):
    # Deliberately conservative byte bound for text plus 8 tokens per 32px patch,
    # and fixed overhead. Includes all reasoning in the output ceiling, no cache
    # hits, and the higher cache-write input rate. Apply long-context premiums
    # whenever even this conservative input bound crosses the pricing threshold.
    text_bound = len((prompt + dumps(manifest["packet"]) + dumps(response_schema(manifest))).encode()) + 4096
    image_bound = sum(((w + 31) // 32) * ((h + 31) // 32) * 8 for w, h in manifest["dimensions"])
    return text_bound + image_bound


def estimated_max_cost(manifest, prompt, batch=False):
    factor = .5 if batch else 1.
    inputs = input_token_bound(manifest, prompt)
    long = inputs > 272_000
    return factor * (inputs * (.25 if long else .125) + manifest["max_output_tokens"] * (.75 if long else .5)) / 1e6


def usage_cost(usage, batch=False):
    details = usage.get("input_tokens_details") or {}
    inputs = usage.get("input_tokens", 0)
    cached = details.get("cached_tokens", 0)
    writes = details.get("cache_write_tokens", 0)
    factor = .5 if batch else 1.
    # When writes are not separately reported, count uncached input at the upper
    # write rate. The ledger is a conservative cost, never an optimistic bill.
    uncached_rate = .1 if "cache_write_tokens" in details else .125
    input_cost = max(0, inputs - cached - writes) * uncached_rate + cached * .01 + writes * .125
    return factor * (input_cost * (2 if inputs > 272_000 else 1) + usage.get("output_tokens", 0) * (.75 if inputs > 272_000 else .5)) / 1e6


class Ledger:
    def __init__(self, path, limit):
        self.db = sqlite3.connect(path)
        self.limit = limit
        self.db.execute("CREATE TABLE IF NOT EXISTS charges (id TEXT PRIMARY KEY, amount REAL, state TEXT)")
        self.db.commit()

    def total(self):
        return float(self.db.execute("SELECT COALESCE(SUM(amount),0) FROM charges").fetchone()[0])

    def reserve(self, request_id, amount):
        with self.db:
            if self.db.execute("SELECT 1 FROM charges WHERE id=?", (request_id,)).fetchone():
                return False
            if self.total() + amount > self.limit:
                raise RuntimeError("budget limit reached; pending work is preserved, never rejected for cost")
            self.db.execute("INSERT INTO charges VALUES (?,?,?)", (request_id, amount, "reserved"))
        return True

    def settle(self, request_id, amount):
        with self.db:
            self.db.execute("UPDATE charges SET amount=?,state='completed' WHERE id=?", (amount, request_id))


class APIClient:
    def __init__(self):
        key = os.environ.pop("OPENAI_API_KEY", None)
        if not key:
            key = getpass.getpass("OpenAI API key (hidden; held in memory only): ")
        self.session = requests.Session()
        self.session.headers["Authorization"] = "Bearer " + key.strip()
        del key

    def call(self, method, endpoint, **kwargs):
        response = self.session.request(method, API + endpoint, timeout=(30, 900), **kwargs)
        if not response.ok:
            try:
                kind = response.json().get("error", {}).get("code") or "request_failed"
            except ValueError:
                kind = "request_failed"
            # Never log API error messages or headers: some echo the credential.
            raise RuntimeError(f"OpenAI HTTP {response.status_code}: {kind}")
        return response

    def close(self):
        self.session.headers.pop("Authorization", None)
        self.session.close()


def result_text(response):
    if response.get("status") != "completed":
        raise ValueError("model response incomplete or failed")
    parts = [content.get("text", "") for item in response.get("output", []) if item.get("type") == "message"
             for content in item.get("content", []) if content.get("type") == "output_text"]
    return json.loads("".join(parts))


def save_response(manifest, response, source, output):
    folder = output / "packets" / manifest["request_id"]
    atomic_json(folder / "response.json", response)
    expected = set(manifest["mapping"])
    try:
        parsed = result_text(response)
        results = parsed["results"]
        if len(results) != len(expected) or {r["sample_id"] for r in results} != expected:
            raise ValueError("missing, duplicate or unexpected sample_id in grouped response")
    except (KeyError, ValueError, TypeError) as exc:
        results = [{"sample_id": sid, "decision": "error", "reason_codes": [str(exc)]} for sid in expected]
    invalid = []
    for result in results:
        sid = result["sample_id"]
        mapping = manifest["mapping"][sid]
        if result["decision"] != "error":
            try:
                apply_result(source.row(sid), result, mapping)
            except (ValueError, KeyError, TypeError) as exc:
                result = {"sample_id": sid, "decision": "error", "reason_codes": [str(exc)], "model_result": result}
        if result["decision"] == "error":
            invalid.append(sid)
        provenance = {k: manifest[k] for k in ("transport_retry_of", "validation_retry_of") if k in manifest}
        atomic_json(output / "decisions" / (digest(sid) + ".json"), {**result, **provenance, "mapping": mapping,
                    "request_id": manifest["request_id"], "stage": manifest.get("stage", "model")})
    return invalid


def run_regular(packets, prompt, source, output, ledger, client, workers=1, recover_interrupted=False):
    # Only HTTP runs in threads; validation, source reads and SQLite reservations
    # remain on the owning thread. Every call is reserved before submission.
    from concurrent.futures import ThreadPoolExecutor
    def finish(manifest, response):
        invalid = save_response(manifest, response, source, output)
        ledger.settle(manifest["request_id"], usage_cost(response.get("usage", {})))
        print(dumps({"event": "completed", "stage": manifest.get("stage", "visual"),
                     "request_id": manifest["request_id"], "invalid": len(invalid),
                     "cost_upper_usd": round(ledger.total(), 6)}), flush=True)
        return invalid
    def response_for(manifest):
        rid = manifest["request_id"]
        path = output / "packets" / rid / "response.json"
        if path.is_file():
            return loads(path.read_text())
        response = client.call("POST", "/responses", json=request_body(manifest, prompt, output)).json()
        # Checkpoint before joining sibling requests: a slow peer or interruption
        # must not strand a completed, potentially billed response in RAM.
        atomic_json(path, response)
        return response
    def retry_invalid(manifest, invalid):
        rid = manifest["request_id"]
        if invalid and not manifest.get("validation_retry_of"):
            # One protocol retry of only invalid results, using exactly the same
            # evidence and task. Never re-audit valid results or spend unreserved money.
            retry = deepcopy(manifest)
            retry["request_id"] = digest([rid, "validation-retry", sorted(invalid)])[:24]
            retry["validation_retry_of"] = rid
            retry["mapping"] = {sid: manifest["mapping"][sid] for sid in invalid}
            retry["packet"]["expressions"] = [e for e in manifest["packet"]["expressions"] if e["sample_id"] in invalid]
            retry["packet"]["validation_errors"] = {
                sid: loads((output / "decisions" / (digest(sid) + ".json")).read_text())["reason_codes"] for sid in invalid}
            atomic_json(output / "packets" / retry["request_id"] / "input.json", retry)
            request_ids = loads((output / "requests.json").read_text())
            if retry["request_id"] not in request_ids:
                atomic_json(output / "requests.json", request_ids + [retry["request_id"]])
            print(dumps({"event": "validation_retry", "expressions": len(invalid)}), flush=True)
            run_regular([retry], prompt, source, output, ledger, client, recover_interrupted=recover_interrupted)
    if not 1 <= workers <= 8:
        raise ValueError("workers must be 1..8")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded waves avoid pre-reserving the entire run and preserve all
        # successful responses even if a sibling call raises a network error.
        for start in range(0, len(packets), workers):
            pending, errors, retries = [], [], []
            for manifest in packets[start:start + workers]:
                rid = manifest["request_id"]
                if not (output / "packets" / rid / "response.json").is_file():
                    try:
                        if not ledger.reserve(rid, estimated_max_cost(manifest, prompt)):
                            state = ledger.db.execute("SELECT state FROM charges WHERE id=?", (rid,)).fetchone()[0]
                            if not recover_interrupted or manifest.get("transport_retry_of") or state != "reserved":
                                raise RuntimeError(f"request {rid} has an unresolved prior charge; inspect before resubmitting")
                            recovery = deepcopy(manifest)
                            recovery["request_id"] = digest([rid, "transport-recovery-v1"])[:24]
                            recovery["transport_retry_of"] = rid
                            # A new ID accounts for a possible second charge. Never
                            # release the original uncertain upper reservation.
                            atomic_json(output / "packets" / recovery["request_id"] / "input.json", recovery)
                            request_ids = loads((output / "requests.json").read_text())
                            if recovery["request_id"] not in request_ids:
                                atomic_json(output / "requests.json", request_ids + [recovery["request_id"]])
                            manifest = recovery
                            rid = manifest["request_id"]
                            if not (output / "packets" / rid / "response.json").is_file() and not ledger.reserve(rid, estimated_max_cost(manifest, prompt)):
                                raise RuntimeError(f"recovery {rid} also has an unresolved charge; no further automatic attempt")
                            print(dumps({"event": "interrupted_recovery", "original_request": recovery["transport_retry_of"],
                                         "request_id": rid, "original_reservation_retained": True}), flush=True)
                    except RuntimeError as exc:
                        errors.append(exc)
                        break
                pending.append((manifest, pool.submit(response_for, manifest)))
            for manifest, future in pending:
                try:
                    retries.append((manifest, finish(manifest, future.result())))
                except Exception as exc:
                    errors.append(exc)
            for manifest, invalid in retries:
                retry_invalid(manifest, invalid)
            release_memory()
            if errors:
                raise errors[0]


def submit_batches(packets, prompt, output, ledger, client, wait_for=None):
    existing = output / "batches.json"
    batches = loads(existing.read_text()) if existing.exists() else []
    submitted = {rid for batch in batches for rid in batch["requests"]}
    if batches and wait_for:
        wait_for(batches)
    elif batches:
        for batch in batches:
            status = client.call("GET", "/batches/" + batch["id"]).json()["status"]
            if status not in {"completed", "failed", "expired", "cancelled"}:
                return batches
    lines, active, size, input_tokens = [], [], 0, 0
    def submit():
        nonlocal lines, active, size, input_tokens
        if not lines:
            return
        path = output / f"batch-{len(batches):04d}.jsonl"
        with path.open("wb") as handle:
            handle.writelines(lines)
        with path.open("rb") as handle:
            file_id = client.call("POST", "/files", data={"purpose": "batch"}, files={"file": (path.name, handle, "application/jsonl")}).json()["id"]
        # Reserves are already durable; a crash after submission never blindly
        # repeats paid requests. The batch ID is checkpointed immediately.
        batch = client.call("POST", "/batches", json={"input_file_id": file_id, "endpoint": "/v1/responses", "completion_window": "24h"}).json()
        batches.append({"id": batch["id"], "requests": active, "input_file_id": file_id})
        atomic_json(existing, batches)
        if wait_for:
            wait_for([batches[-1]])
        lines, active, size, input_tokens = [], [], 0, 0
    for manifest in packets:
        rid = manifest["request_id"]
        if rid in submitted or (output / "packets" / rid / "response.json").exists():
            continue
        line = (dumps({"custom_id": rid, "method": "POST", "url": "/v1/responses", "body": request_body(manifest, prompt, output)}) + "\n").encode()
        if len(line) > 180_000_000:
            raise ValueError("one request exceeds the Batch file budget; reduce expression group size")
        bound = input_token_bound(manifest, prompt)
        if bound > 16_000_000:
            raise ValueError("one request exceeds the conservative Batch token queue bound")
        if lines and (size + len(line) > 180_000_000 or len(lines) >= 1000 or input_tokens + bound > 16_000_000):
            submit()
            if not wait_for:
                # Async mode submits one bounded wave. Run --mode batch again
                # after collection, or use --wait to finish every wave in order.
                return batches
        if not ledger.reserve(rid, estimated_max_cost(manifest, prompt, batch=True)):
            raise RuntimeError(f"unresolved submission for {rid}; collect existing batches before retrying")
        lines.append(line)
        active.append(rid)
        size += len(line)
        input_tokens += bound
    submit()
    return batches


def collect_batches(batches, packets, source, output, ledger, client, wait):
    by_id = {p["request_id"]: p for p in packets}
    while True:
        pending = 0
        for batch in batches:
            receipt = output / "batch-receipts" / (batch["id"] + ".json")
            if receipt.exists():
                continue
            status = client.call("GET", "/batches/" + batch["id"]).json()
            print(dumps({"event": "batch", "id": batch["id"], "status": status["status"], "counts": status.get("request_counts")}), flush=True)
            if status["status"] not in {"completed", "failed", "expired", "cancelled"}:
                pending += 1
                continue
            for field in ("output_file_id", "error_file_id"):
                if not status.get(field):
                    continue
                text = client.call("GET", "/files/" + status[field] + "/content").text
                for line in text.splitlines():
                    item = loads(line)
                    rid = item["custom_id"]
                    response = item.get("response") or {}
                    if response.get("status_code") == 200:
                        body = response["body"]
                        save_response(by_id[rid], body, source, output)
                        ledger.settle(rid, usage_cost(body.get("usage", {}), batch=True))
                    else:
                        for sid in by_id[rid]["mapping"]:
                            atomic_json(output / "decisions" / (digest(sid) + ".json"), {"sample_id": sid, "decision": "error", "request_id": rid, "reason_codes": ["batch_request_failed"]})
                        # An uncertain failed request retains its upper-bound
                        # reservation until billing is reconciled.
            for rid in batch["requests"]:
                for sid in by_id[rid]["mapping"]:
                    decision = output / "decisions" / (digest(sid) + ".json")
                    if not decision.exists():
                        atomic_json(decision, {"sample_id": sid, "decision": "error", "request_id": rid,
                                               "reason_codes": ["batch_" + status["status"], "no_returned_result"]})
            atomic_json(receipt, {"id": batch["id"], "status": status["status"], "counts": status.get("request_counts")})
        if not pending or not wait:
            break
        time.sleep(30)


def export(source, ids, output, early, ledger):
    writers, paths, counters, audit, buffers = {}, {}, Counter(), [], defaultdict(list)
    splits = sorted({source.rows[sid]["split"] for sid in ids})
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", split) for split in splits):
        raise ValueError("unsafe split name")
    def flush(split):
        if buffers[split]:
            writers[split].write_table(pa.Table.from_pylist(buffers[split], schema=source.schema))
            buffers[split].clear()
    try:
        for sid in ids:
            path = output / "decisions" / (digest(sid) + ".json")
            result = loads(path.read_text()) if path.exists() else early.get(sid, {"sample_id": sid, "decision": "pending", "reason_codes": []})
            counters[result["decision"]] += 1
            audit.append({k: v for k, v in result.items() if k not in {"mapping", "model_result"}})
            if result["decision"] not in {"keep", "rewrite"}:
                continue
            row = apply_result(source.row(sid), result, result["mapping"])
            split = row["split"]
            if split not in writers:
                final = output / f"{split}.parquet"
                paths[split] = final
                writers[split] = pq.ParquetWriter(str(final) + ".part", source.schema, compression="zstd")
            buffers[split].append(row)
            if len(buffers[split]) >= 64:
                flush(split)
        for split in writers:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()
    for split in splits:
        final = output / f"{split}.parquet"
        if split in paths:
            Path(str(final) + ".part").replace(final)
        else:
            pq.write_table(pa.Table.from_pylist([], schema=source.schema), str(final) + ".part", compression="zstd")
            Path(str(final) + ".part").replace(final)
        if not pq.read_schema(final).equals(source.schema, check_metadata=True):
            raise RuntimeError("exported Arrow schema changed")
    atomic_json(output / "audit.json", audit)
    summary = {"model": MODEL, "reasoning_effort": "medium", "selected": len(ids), "counts": dict(counters),
               "cost_upper_usd": round(ledger.total(), 6), "complete": not (counters["pending"] or counters["error"])}
    atomic_json(output / "summary.json", summary)
    print(dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="downloaded HF root, split directory, or one Parquet")
    parser.add_argument("--media-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["prepare", "regular", "batch", "collect", "export"], default="regular")
    parser.add_argument("--pilot-per-dataset", type=int, default=0, help="10 selects a 20-expression Ref-YT-VOS/ReVOS pilot; 0 means all")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--group-tracklets", type=int, default=24, help="split shared groups above this inventory size")
    parser.add_argument("--max-tracklets", type=int, default=64, help="reject a single pathological expression above this limit after exact deduplication")
    parser.add_argument("--sample-count", type=int, default=12, help="VIPOSeg evenly spaced frame target, NOT FPS; first appearances are added")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts/video_refinement.md")
    parser.add_argument("--budget-usd", type=float, default=100.)
    parser.add_argument("--wait", action="store_true", help="monitor submitted Batch requests until terminal")
    args = parser.parse_args()
    if (not 1 <= args.group_size <= 12 or args.sample_count < 2 or args.budget_usd <= 0
            or args.pilot_per_dataset < 0 or args.group_tracklets < 1 or args.max_tracklets < 1):
        parser.error("group-size must be 1..12, sample-count >=2; budget/tracklet limits positive; pilot count nonnegative")
    if args.mode in {"regular", "prepare", "batch"} and not args.media_root:
        parser.error("--media-root is required to prepare visual inputs")
    args.media_root = [p.resolve() for p in args.media_root]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Prevent two runs from racing the same budget, decisions and atomic exports.
    import fcntl
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another refinement process is using this output directory") from None
    paths = discover_inputs(args.input)
    source = Source(paths)
    metadata, prompt = read_prompt(args.prompt)
    ids = select_ids(source, args.pilot_per_dataset, args.seed)
    configuration = {"version": VERSION, "model": MODEL, "reasoning_effort": "medium", "prompt_sha256": digest(prompt),
                     "sources": [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths],
                     "selected_ids": ids, "group_size": args.group_size, "group_tracklets": args.group_tracklets,
                     "max_tracklets": args.max_tracklets, "sample_count": args.sample_count}
    config_path = output / "run.json"
    if config_path.exists() and loads(config_path.read_text()) != configuration:
        raise ValueError("output belongs to a different input/prompt/configuration; use a new output directory")
    atomic_json(config_path, configuration)
    (output / "prompt_metadata.txt").write_text(metadata, encoding="utf-8")
    (output / "prompt_sent.txt").write_text(prompt, encoding="utf-8")
    ledger = Ledger(output / "budget.sqlite", args.budget_usd)
    client = None
    try:
        if args.mode in {"regular", "batch", "collect"}:
            execution = "regular" if args.mode == "regular" else "batch"
            execution_path = output / "execution.json"
            if execution_path.exists() and loads(execution_path.read_text())["mode"] != execution:
                raise ValueError("regular and Batch billing cannot be mixed in one output directory")
            atomic_json(execution_path, {"mode": execution})
        if args.mode in {"collect", "export"}:
            packets = [loads((output / "packets" / rid / "input.json").read_text()) for rid in loads((output / "requests.json").read_text())]
            early = loads((output / "heuristics.json").read_text())
        else:
            packets, early = prepare(source, ids, args, prompt, output)
        estimate = sum(estimated_max_cost(p, prompt, batch=args.mode != "regular") for p in packets)
        print(dumps({"event": "cost_preflight", "requests": len(packets), "expressions": len(ids), "reserved_upper_usd": round(estimate, 4)}), flush=True)
        if args.mode in {"regular", "batch"} and estimate > args.budget_usd:
            raise RuntimeError("prepared workload exceeds conservative budget; no API requests sent")
        if args.mode in {"regular", "batch", "collect"}:
            client = APIClient()
        if args.mode == "regular":
            run_regular(packets, prompt, source, output, ledger, client)
        elif args.mode in {"batch", "collect"}:
            wait_for = (lambda batches: collect_batches(batches, packets, source, output, ledger, client, True)) if args.wait else None
            batches = submit_batches(packets, prompt, output, ledger, client, wait_for) if args.mode == "batch" else loads((output / "batches.json").read_text())
            collect_batches(batches, packets, source, output, ledger, client, args.wait)
        if args.mode != "prepare":
            export(source, ids, output, early, ledger)
    finally:
        if client is not None:
            client.close()
        ledger.db.close()
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, requests.RequestException) as exc:
        # No tracebacks of API calls and no headers/credentials in diagnostics.
        message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(exc))
        print(f"Refinement stopped: {message}", file=sys.stderr)
        raise SystemExit(1)
