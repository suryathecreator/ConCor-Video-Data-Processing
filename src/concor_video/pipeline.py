"""Official-target anchoring plus SAM3.1 contextual tracklet generation."""

from __future__ import annotations

import inspect
import json
import math
import os
import shutil
import statistics
import time
import types
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .rle import decode_rle, encode_rle
from .tracklet_schema import SCHEMA_VERSION, make_span, rebuild_span_links, validate_record


def temporal_iou(
    left: list[np.ndarray | None], right: list[np.ndarray | None]
) -> float:
    """Volume IoU over aligned frames."""

    intersection = 0
    union = 0
    for left_mask, right_mask in zip(left, right, strict=True):
        if left_mask is None and right_mask is None:
            continue
        if left_mask is None:
            union += int(np.asarray(right_mask).sum())
            continue
        if right_mask is None:
            union += int(np.asarray(left_mask).sum())
            continue
        left_bool = np.asarray(left_mask, dtype=np.bool_)
        right_bool = np.asarray(right_mask, dtype=np.bool_)
        intersection += int(np.logical_and(left_bool, right_bool).sum())
        union += int(np.logical_or(left_bool, right_bool).sum())
    return float(intersection / union) if union else 0.0


def _present_frame_count(masks: list[np.ndarray | None]) -> int:
    return sum(1 for mask in masks if mask is not None and bool(mask.any()))


def install_session_compatibility(predictor):
    """Adapt the current shared base predictor to SAM3.1's init-state signature."""

    signature = inspect.signature(predictor.model.init_state)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return predictor
    accepted = set(signature.parameters)
    base_keys = {
        "resource_path",
        "offload_video_to_cpu",
        "offload_state_to_cpu",
        "async_loading_frames",
    }
    if base_keys <= accepted:
        return predictor

    def compatible_start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        inference_state = self.model.init_state(
            **{key: value for key, value in init_kwargs.items() if key in accepted}
        )
        resolved_session_id = session_id or str(uuid.uuid4())
        now = time.time()
        self._all_inference_states[resolved_session_id] = {
            "state": inference_state,
            "session_id": resolved_session_id,
            "start_time": now,
            "last_use_time": now,
        }
        return {"session_id": resolved_session_id}

    predictor.start_session = types.MethodType(compatible_start_session, predictor)
    return predictor


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    try:
        temporary.write_bytes(payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _find_frame(directory: Path, video_id: str, frame_id: str) -> Path:
    video_dir = directory / video_id
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = video_dir / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing frame {video_id}/{frame_id} under {directory}")


class DatasetProvider:
    """Materialize ordered frames once and keep large mask maps/ZIP indexes warm."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root.resolve()
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self._archive_names: dict[Path, set[str]] = {}
        self._mask_dicts: dict[Path, dict[str, list[dict[str, Any] | None]]] = {}

    def _archive(self, path: Path) -> zipfile.ZipFile:
        if path not in self._archives:
            archive = zipfile.ZipFile(path)
            self._archives[path] = archive
            self._archive_names[path] = set(archive.namelist())
        return self._archives[path]

    def _frame_payload(self, source: Path, video_id: str, frame_id: str) -> bytes:
        archive = self._archive(source)
        names = self._archive_names[source]
        candidates = (
            f"{video_id}/{frame_id}.jpg",
            f"JPEGImages/{video_id}/{frame_id}.jpg",
            f"{video_id}/{frame_id}.jpeg",
            f"JPEGImages/{video_id}/{frame_id}.jpeg",
        )
        name = next((value for value in candidates if value in names), None)
        if name is None:
            raise FileNotFoundError(f"missing frame {video_id}/{frame_id} in {source}")
        return archive.read(name)

    def _materialize_frames(self, unit: dict[str, Any]) -> list[Path]:
        source = Path(unit["frame_source"])
        destination = self.cache_root / "frames" / unit["sample_id"]
        paths: list[Path] = []
        for index, frame_id in enumerate(unit["frame_ids"]):
            output = destination / f"{index:05d}.jpg"
            if not output.is_file():
                if source.is_dir():
                    _atomic_copy(_find_frame(source, unit["video_id"], frame_id), output)
                else:
                    _atomic_bytes(output, self._frame_payload(source, unit["video_id"], frame_id))
            paths.append(output)
        return paths

    def _ref_targets(self, unit: dict[str, Any]) -> list[tuple[str, list[np.ndarray | None]]]:
        annotation_value = unit.get("annotation_source")
        if not annotation_value or unit["target_source_expected"] != "official_dataset_ground_truth":
            return []
        annotation_root = Path(annotation_value)
        rows: list[tuple[str, list[np.ndarray | None]]] = []
        for object_id in unit.get("target_object_ids", []):
            masks: list[np.ndarray | None] = []
            for frame_id in unit["frame_ids"]:
                path = annotation_root / unit["video_id"] / f"{frame_id}.png"
                if not path.is_file():
                    masks.append(None)
                    continue
                labels = np.asarray(Image.open(path))
                mask = labels == int(object_id)
                masks.append(mask if bool(mask.any()) else None)
            rows.append((str(object_id), masks))
        return rows

    def _revos_targets(self, unit: dict[str, Any]) -> list[tuple[str, list[np.ndarray | None]]]:
        if unit["negative"]:
            return []
        path = Path(unit["mask_dict_path"])
        if path not in self._mask_dicts:
            self._mask_dicts[path] = json.loads(path.read_text(encoding="utf-8"))
        mask_dict = self._mask_dicts[path]
        rows: list[tuple[str, list[np.ndarray | None]]] = []
        for annotation_id in unit.get("target_annotation_ids", []):
            sequence = mask_dict[str(annotation_id)]
            masks = [
                decode_rle(sequence[index]) if index < len(sequence) and sequence[index] else None
                for index in range(len(unit["frame_ids"]))
            ]
            rows.append((str(annotation_id), masks))
        return rows

    def materialize(
        self, unit: dict[str, Any]
    ) -> tuple[list[Path], list[tuple[str, list[np.ndarray | None]]]]:
        frames = self._materialize_frames(unit)
        if unit["dataset"] == "ref_youtube_vos":
            targets = self._ref_targets(unit)
        elif unit["dataset"] == "revos":
            targets = self._revos_targets(unit)
        else:
            raise ValueError(f"unsupported dataset: {unit['dataset']}")
        return frames, targets

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()


def _normalize_mask(value: Any) -> np.ndarray:
    mask = np.asarray(value, dtype=np.bool_)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"unexpected SAM mask shape {mask.shape}")
    return mask


def _sam_tracklets_for_prompt(
    predictor,
    *,
    session_id: str,
    prompt: str,
    frame_count: int,
    output_threshold: float,
) -> list[dict[str, Any]]:
    predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
            "output_prob_thresh": output_threshold,
        }
    )
    masks_by_object: dict[int, dict[int, np.ndarray]] = {}
    scores_by_object: dict[int, list[float]] = {}
    for response in predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": frame_count,
            "output_prob_thresh": output_threshold,
        }
    ):
        frame_index = int(response["frame_index"])
        if not 0 <= frame_index < frame_count:
            continue
        outputs = response["outputs"]
        object_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
        masks = np.asarray(outputs.get("out_binary_masks", []))
        scores = np.asarray(outputs.get("out_probs", np.ones(len(object_ids))), dtype=float).reshape(-1)
        for local_index, object_id_raw in enumerate(object_ids):
            object_id = int(object_id_raw)
            masks_by_object.setdefault(object_id, {})[frame_index] = _normalize_mask(
                masks[local_index]
            )
            score = float(scores[local_index]) if local_index < len(scores) else 1.0
            scores_by_object.setdefault(object_id, []).append(score)
    rows: list[dict[str, Any]] = []
    for object_id, frame_map in sorted(masks_by_object.items()):
        arrays = [frame_map.get(index) for index in range(frame_count)]
        scores = scores_by_object.get(object_id, [])
        rows.append(
            {
                "sam_object_id": object_id,
                "arrays": arrays,
                "present_frames": _present_frame_count(arrays),
                "total_pixels": sum(int(mask.sum()) for mask in arrays if mask is not None),
                "mean_score": statistics.fmean(scores) if scores else 0.0,
                "max_score": max(scores, default=0.0),
            }
        )
    return rows


def _passes_track_filter(row: dict[str, Any], frame_count: int) -> tuple[bool, str]:
    minimum_frames = 1 if frame_count <= 4 else max(2, math.ceil(frame_count * 0.05))
    if row["present_frames"] < minimum_frames:
        return False, f"present_frames<{minimum_frames}"
    if row["total_pixels"] < 64:
        return False, "total_pixels<64"
    if row["max_score"] < 0.55 or row["mean_score"] < 0.45:
        return False, "confidence_below_threshold"
    return True, "retained"


def _unique_spans(text: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        start, end = int(candidate["start"]), int(candidate["end"])
        values[(start, end)] = make_span(text, start, end)
    return [values[key] for key in sorted(values)]


def _append_tracklet(
    record: dict[str, Any],
    *,
    tracklet_id: str,
    arrays: list[np.ndarray | None],
    source: str,
    source_annotation_id: str | None,
    sam_prompt: str | None,
    confidence: float,
    max_confidence: float | None = None,
) -> None:
    row: dict[str, Any] = {
        "tracklet_id": tracklet_id,
        "source": source,
        "source_annotation_id": source_annotation_id,
        "sam_prompt": sam_prompt,
        "confidence": confidence,
        "present_frames": _present_frame_count(arrays),
        "masks": [encode_rle(mask) for mask in arrays],
    }
    if max_confidence is not None:
        row["max_confidence"] = max_confidence
    record["tracklets"].append(row)


def process_unit(
    unit: dict[str, Any],
    *,
    provider: DatasetProvider,
    predictor,
    output_threshold: float = 0.5,
) -> dict[str, Any]:
    frame_paths, gt_sequences = provider.materialize(unit)
    text = unit["text"]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": unit["sample_id"],
        "dataset": unit["dataset"],
        "split": unit["split"],
        "cohort": unit["cohort"],
        "annotation_protocol": unit["annotation_protocol"],
        "provenance_warning": unit.get("provenance_warning"),
        "video_id": unit["video_id"],
        "expression_id": unit["expression_id"],
        "text": text,
        "negative": bool(unit["negative"]),
        "frame_ids": list(unit["frame_ids"]),
        "frame_files": [f"{unit['video_id']}/{frame_id}.jpg" for frame_id in unit["frame_ids"]],
        "tracklets": [],
        "groups": [],
        "span_links": [],
        "extraction": unit["extraction"],
        "sam_prompt_audit": [],
        "pipeline": {
            "target_source": unit["target_source_expected"],
            "context_source": "sam3.1_multiplex",
            "sam_output_prob_threshold": output_threshold,
            "duplicate_target_temporal_iou": 0.65,
            "duplicate_context_temporal_iou": 0.80,
        },
    }
    if unit["negative"]:
        record["disposition"] = "negative_unsegmentable"
        validate_record(record)
        return record

    target = unit["extraction"]["target"]
    target_spans = [make_span(text, int(target["start"]), int(target["end"]))]
    target_spans.extend(unit["extraction"].get("target_coreference_spans", []))
    target_arrays: list[list[np.ndarray | None]] = []
    target_ids: list[str] = []
    for index, (source_id, arrays) in enumerate(gt_sequences, 1):
        if not any(mask is not None and bool(mask.any()) for mask in arrays):
            continue
        tracklet_id = f"gt-{index:03d}"
        target_ids.append(tracklet_id)
        target_arrays.append(arrays)
        _append_tracklet(
            record,
            tracklet_id=tracklet_id,
            arrays=arrays,
            source="ground_truth",
            source_annotation_id=source_id,
            sam_prompt=None,
            confidence=1.0,
        )

    if unit["target_source_expected"] == "official_dataset_ground_truth" and not target_ids:
        raise RuntimeError("official target tracklet is empty or missing")
    needs_target_sam = (
        unit["target_source_expected"] != "official_dataset_ground_truth" and not target_ids
    )
    prompt_groups = list(unit.get("sam_prompt_groups", []))
    needs_sam = needs_target_sam or bool(prompt_groups)
    if needs_sam and predictor is None:
        raise RuntimeError("SAM3.1 predictor is required for this sample")

    session_id: str | None = None
    accepted_context: list[dict[str, Any]] = []
    try:
        if needs_sam:
            session = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(frame_paths[0].parent),
                    "offload_video_to_cpu": False,
                    "offload_state_to_cpu": False,
                }
            )
            session_id = session["session_id"]

        if needs_target_sam:
            target_prompt = str(target.get("surface") or target.get("sam_prompt") or "").strip()
            raw_targets = _sam_tracklets_for_prompt(
                predictor,
                session_id=session_id,
                prompt=target_prompt,
                frame_count=len(frame_paths),
                output_threshold=output_threshold,
            )
            audit = {
                "role": "main_referent",
                "sam_prompt": target_prompt,
                "surface_spans": [target["surface"]],
                "raw_tracklets": len(raw_targets),
                "retained_tracklets": 0,
                "rejections": [],
            }
            for raw in raw_targets:
                passed, reason = _passes_track_filter(raw, len(frame_paths))
                if not passed:
                    audit["rejections"].append(
                        {"sam_object_id": raw["sam_object_id"], "reason": reason}
                    )
                    continue
                tracklet_id = f"sam-target-{len(target_ids) + 1:03d}"
                target_ids.append(tracklet_id)
                target_arrays.append(raw["arrays"])
                _append_tracklet(
                    record,
                    tracklet_id=tracklet_id,
                    arrays=raw["arrays"],
                    source="sam3.1_main_referent",
                    source_annotation_id=None,
                    sam_prompt=target_prompt,
                    confidence=raw["mean_score"],
                    max_confidence=raw["max_score"],
                )
                audit["retained_tracklets"] += 1
            record["sam_prompt_audit"].append(audit)

        if target_ids:
            record["groups"].append(
                {
                    "group_id": "main-referent",
                    "role": "main_referent",
                    "identity": target["head"],
                    "text_spans": target_spans,
                    "tracklet_ids": target_ids,
                }
            )

        # Without the main referent, context-only output is not a valid BCC pair.
        for prompt_group in prompt_groups if target_ids else []:
            prompt = prompt_group["sam_prompt"]
            raw_tracks = _sam_tracklets_for_prompt(
                predictor,
                session_id=session_id,
                prompt=prompt,
                frame_count=len(frame_paths),
                output_threshold=output_threshold,
            )
            audit = {
                "role": "context_entity",
                "sam_prompt": prompt,
                "surface_spans": [candidate["surface"] for candidate in prompt_group["candidates"]],
                "raw_tracklets": len(raw_tracks),
                "retained_tracklets": 0,
                "rejections": [],
            }
            for raw in raw_tracks:
                passed, reason = _passes_track_filter(raw, len(frame_paths))
                if not passed:
                    audit["rejections"].append(
                        {"sam_object_id": raw["sam_object_id"], "reason": reason}
                    )
                    continue
                target_overlap = max(
                    (temporal_iou(raw["arrays"], arrays) for arrays in target_arrays),
                    default=0.0,
                )
                if target_overlap >= 0.65:
                    audit["rejections"].append(
                        {
                            "sam_object_id": raw["sam_object_id"],
                            "reason": "duplicates_main_referent",
                            "temporal_iou": target_overlap,
                        }
                    )
                    continue
                duplicate = next(
                    (
                        prior
                        for prior in accepted_context
                        if temporal_iou(raw["arrays"], prior["arrays"]) >= 0.80
                    ),
                    None,
                )
                if duplicate is not None:
                    duplicate["prompts"].add(prompt)
                    duplicate["candidates"].extend(prompt_group["candidates"])
                    audit["rejections"].append(
                        {
                            "sam_object_id": raw["sam_object_id"],
                            "reason": "merged_duplicate_context_tracklet",
                        }
                    )
                    audit["retained_tracklets"] += 1
                    continue
                accepted_context.append(
                    {
                        **raw,
                        "prompts": {prompt},
                        "candidates": list(prompt_group["candidates"]),
                    }
                )
                audit["retained_tracklets"] += 1
            record["sam_prompt_audit"].append(audit)
    finally:
        if session_id is not None:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id, "run_gc_collect": False}
            )

    for index, context in enumerate(accepted_context, 1):
        tracklet_id = f"sam-context-{index:03d}"
        prompts = sorted(context["prompts"])
        spans = _unique_spans(text, context["candidates"])
        identity = "/".join(
            sorted({candidate["head"] for candidate in context["candidates"]})
        )
        _append_tracklet(
            record,
            tracklet_id=tracklet_id,
            arrays=context["arrays"],
            source="sam3.1_context",
            source_annotation_id=None,
            sam_prompt=prompts[0],
            confidence=context["mean_score"],
            max_confidence=context["max_score"],
        )
        record["tracklets"][-1]["sam_prompt_aliases"] = prompts
        record["groups"].append(
            {
                "group_id": f"context-{index:03d}",
                "role": "context_entity",
                "identity": identity,
                "text_spans": spans,
                "tracklet_ids": [tracklet_id],
            }
        )

    unresolved = [
        audit["sam_prompt"]
        for audit in record["sam_prompt_audit"]
        if audit["role"] == "context_entity" and audit["retained_tracklets"] == 0
    ]
    record["unresolved_required_prompts"] = unresolved
    if not target_ids:
        record["disposition"] = "missing_main_referent"
    elif unresolved:
        record["disposition"] = "incomplete_context"
    else:
        record["disposition"] = "complete_bcc"
    record["span_links"] = rebuild_span_links(record["groups"])
    validate_record(record)
    return record


def build_predictor(
    *,
    checkpoint_path: str,
    compile_model: bool,
    warm_up: bool,
    use_fa3: bool,
):
    """Import SAM lazily so CPU preparation and export do not initialize Torch."""

    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        checkpoint_path=checkpoint_path,
        version="sam3.1",
        compile=compile_model,
        warm_up=warm_up,
        max_num_objects=32,
        multiplex_count=16,
        use_fa3=use_fa3,
        use_rope_real=True,
        async_loading_frames=True,
    )
    return install_session_compatibility(predictor)
