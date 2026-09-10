"""Canonical tracklet-native Bidirectional Concept Correspondence records.

The representation deliberately exposes both directions of the relation:
``groups`` is tracklet -> text spans, while ``span_links`` is text span -> one
or more tracklets.  Character offsets are Unicode/Python half-open offsets.
"""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "concor-video-tracklet-bcc-v2"


def make_span(text: str, start: int, end: int) -> dict[str, Any]:
    if not (0 <= start < end <= len(text)):
        raise ValueError(f"invalid span [{start}, {end}) for text of length {len(text)}")
    return {"start": start, "end": end, "text": text[start:end]}


def rebuild_span_links(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive the text-to-tracklet view without losing shared plural spans."""

    by_span: dict[tuple[int, int, str], set[str]] = {}
    for group in groups:
        tracklet_ids = {str(value) for value in group.get("tracklet_ids", [])}
        for span in group.get("text_spans", []):
            key = (int(span["start"]), int(span["end"]), str(span["text"]))
            by_span.setdefault(key, set()).update(tracklet_ids)
    return [
        {
            "start": start,
            "end": end,
            "text": value,
            "tracklet_ids": sorted(tracklet_ids),
        }
        for (start, end, value), tracklet_ids in sorted(by_span.items())
    ]


def validate_record(record: dict[str, Any]) -> None:
    """Fail closed on broken BCC links or silently truncated tracklets."""

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema_version")
    text = str(record.get("text", ""))
    frame_ids = list(record.get("frame_ids", []))
    if not frame_ids:
        raise ValueError("record has no frames")
    frame_files = list(record.get("frame_files", []))
    if frame_files and len(frame_files) != len(frame_ids):
        raise ValueError("frame_files must align one-to-one with frame_ids")
    tracklets = list(record.get("tracklets", []))
    tracklet_by_id: dict[str, dict[str, Any]] = {}
    for tracklet in tracklets:
        tracklet_id = str(tracklet["tracklet_id"])
        if tracklet_id in tracklet_by_id:
            raise ValueError(f"duplicate tracklet_id: {tracklet_id}")
        masks = list(tracklet.get("masks", []))
        if len(masks) != len(frame_ids):
            raise ValueError(
                f"tracklet {tracklet_id} has {len(masks)} masks for {len(frame_ids)} frames"
            )
        for rle in masks:
            if rle is None:
                continue
            if len(rle.get("size", [])) != 2 or any(int(value) <= 0 for value in rle["size"]):
                raise ValueError(f"tracklet {tracklet_id} has an invalid RLE size")
            counts = rle.get("counts")
            if not isinstance(counts, list) or sum(int(value) for value in counts) != int(
                rle["size"][0]
            ) * int(rle["size"][1]):
                raise ValueError(f"tracklet {tracklet_id} has invalid uncompressed RLE counts")
        tracklet_by_id[tracklet_id] = tracklet

    linked: set[str] = set()
    for group in record.get("groups", []):
        ids = [str(value) for value in group.get("tracklet_ids", [])]
        if not ids:
            raise ValueError("BCC group has no tracklet_ids")
        for tracklet_id in ids:
            if tracklet_id not in tracklet_by_id:
                raise ValueError(f"group references missing tracklet {tracklet_id}")
            linked.add(tracklet_id)
        if not group.get("text_spans"):
            raise ValueError("BCC group has no text span")
        for span in group["text_spans"]:
            start, end = int(span["start"]), int(span["end"])
            if text[start:end] != span["text"]:
                raise ValueError(
                    f"span mismatch [{start}, {end}): {span['text']!r} != {text[start:end]!r}"
                )

    if linked != set(tracklet_by_id):
        missing = sorted(set(tracklet_by_id) - linked)
        raise ValueError(f"unlinked retained tracklets: {missing}")

    expected = rebuild_span_links(record.get("groups", []))
    if record.get("span_links") != expected:
        raise ValueError("span_links is not the canonical inverse of groups")
    if record.get("negative") and (tracklets or record.get("groups")):
        raise ValueError("negative/nonexistent record must not contain tracklets")
    disposition = record.get("disposition")
    if record.get("negative") and disposition != "negative_unsegmentable":
        raise ValueError("negative record has the wrong disposition")
    if not record.get("negative") and not tracklets and disposition != "missing_main_referent":
        raise ValueError("positive record without tracklets must be a missing main referent")
    if tracklets and not any(group.get("role") == "main_referent" for group in record["groups"]):
        raise ValueError("positive tracklets require a main_referent group")
