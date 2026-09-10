"""Checkpointable, dynamically load-balanced, video-batched worker."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpointing import atomic_json, sample_claim
from .datasets import sanitize_frame_ids
from .pipeline import DatasetProvider, build_predictor, process_unit


STOP_REQUESTED = False


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[drain] received signal {signum}; finishing the active expression", flush=True)


def _requires_sam(unit: dict[str, Any]) -> bool:
    return bool(
        not unit["negative"]
        and (
            unit["target_source_expected"] != "official_dataset_ground_truth"
            or unit.get("sam_prompt_groups")
        )
    )


def _video_key(unit: dict[str, Any]) -> str:
    return f"{unit['dataset']}::{unit['split']}::{unit['video_id']}"


def _error_attempts(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempt", 1))
    except (OSError, ValueError, TypeError):
        return 1


def _pending(
    unit: dict[str, Any],
    *,
    records_dir: Path,
    errors_dir: Path,
    retry_errors: bool,
    max_error_attempts: int,
) -> bool:
    if (records_dir / f"{unit['sample_id']}.json").is_file():
        return False
    attempts = _error_attempts(errors_dir / f"{unit['sample_id']}.json")
    return attempts == 0 or (retry_errors and attempts < max_error_attempts)


def _claim_name(video_key: str) -> str:
    return hashlib.sha256(video_key.encode("utf-8")).hexdigest()[:24]


def run_worker(
    *,
    worklist_path: Path,
    campaign_root: Path,
    checkpoint: Path,
    frame_cache_root: Path | None,
    shard_index: int,
    shard_count: int,
    output_threshold: float,
    compile_model: bool,
    warm_up: bool,
    use_fa3: bool,
    retry_errors: bool,
    max_error_attempts: int,
    max_samples: int | None,
) -> int:
    worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    records_dir = campaign_root / "records"
    errors_dir = campaign_root / "errors"
    claims_dir = campaign_root / "claims"
    cache_root = frame_cache_root or (campaign_root / "cache")
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in worklist["units"]:
        frames, ignored_frames = sanitize_frame_ids(unit.get("frame_ids", []))
        if ignored_frames:
            unit["frame_ids"] = frames
            unit["extraction"].setdefault("notes", []).append(
                "ignored hidden non-frame metadata entries: "
                + ", ".join(repr(value) for value in ignored_frames)
            )
        by_video[_video_key(unit)].append(unit)
    video_batches = sorted(by_video.items())
    if video_batches:
        offset = (shard_index * max(1, len(video_batches) // shard_count)) % len(video_batches)
        video_batches = video_batches[offset:] + video_batches[:offset]

    initially_remaining = sum(
        _pending(
            unit,
            records_dir=records_dir,
            errors_dir=errors_dir,
            retry_errors=retry_errors,
            max_error_attempts=max_error_attempts,
        )
        for unit in worklist["units"]
    )
    print(
        f"[worker] dynamic={shard_index}/{shard_count} videos={len(video_batches)} "
        f"campaign_remaining={initially_remaining}",
        flush=True,
    )
    if not initially_remaining:
        return 0

    stable_job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    stable_task = os.environ.get("SLURM_ARRAY_TASK_ID", str(shard_index))
    owner = (
        f"slurm:{stable_job}:{stable_task}"
        if stable_job
        else f"local:{socket.gethostname()}:{os.getpid()}:{shard_index}"
    )
    provider = DatasetProvider(cache_root)
    predictor = None
    committed_by_worker = 0
    try:
        while not STOP_REQUESTED:
            saw_pending = False
            claimed_video = False
            for video_key, units in video_batches:
                if STOP_REQUESTED or (
                    max_samples is not None and committed_by_worker >= max_samples
                ):
                    break
                pending_units = [
                    unit
                    for unit in units
                    if _pending(
                        unit,
                        records_dir=records_dir,
                        errors_dir=errors_dir,
                        retry_errors=retry_errors,
                        max_error_attempts=max_error_attempts,
                    )
                ]
                if not pending_units:
                    continue
                saw_pending = True
                video_claim = claims_dir / "videos" / f"{_claim_name(video_key)}.claim"
                with sample_claim(video_claim, owner=owner) as owns_video:
                    if not owns_video:
                        continue
                    claimed_video = True
                    pending_units = [
                        unit
                        for unit in pending_units
                        if _pending(
                            unit,
                            records_dir=records_dir,
                            errors_dir=errors_dir,
                            retry_errors=retry_errors,
                            max_error_attempts=max_error_attempts,
                        )
                    ]
                    if not pending_units:
                        continue
                    needs_sam = any(_requires_sam(unit) for unit in pending_units)
                    if needs_sam and predictor is None:
                        if not checkpoint.is_file():
                            raise FileNotFoundError(f"missing SAM3.1 checkpoint: {checkpoint}")
                        print("[model] loading SAM3.1 Multiplex", flush=True)
                        predictor = build_predictor(
                            checkpoint_path=str(checkpoint),
                            compile_model=compile_model,
                            warm_up=warm_up,
                            use_fa3=use_fa3,
                        )
                    shared_frames = None
                    session_id = None
                    prompt_cache: dict[str, list[dict[str, Any]]] = {}
                    batch_failed = False
                    try:
                        if needs_sam:
                            reference = next(unit for unit in pending_units if not unit["negative"])
                            shared_frames = provider.materialize_frames(reference)
                            response = predictor.handle_request(
                                {
                                    "type": "start_session",
                                    "resource_path": str(shared_frames[0].parent),
                                    "offload_video_to_cpu": False,
                                    "offload_state_to_cpu": False,
                                }
                            )
                            session_id = response["session_id"]
                        print(
                            f"[video] {video_key} instructions={len(pending_units)} "
                            f"shared_prompt_cache=on",
                            flush=True,
                        )
                        for unit in pending_units:
                            if STOP_REQUESTED or (
                                max_samples is not None
                                and committed_by_worker >= max_samples
                            ):
                                break
                            output_path = records_dir / f"{unit['sample_id']}.json"
                            error_path = errors_dir / f"{unit['sample_id']}.json"
                            with sample_claim(
                                claims_dir / "samples" / f"{unit['sample_id']}.claim",
                                owner=owner,
                            ) as owns_sample:
                                if not owns_sample or output_path.is_file():
                                    continue
                                print(
                                    f"[sample] {unit['sample_id']} "
                                    f"context_prompts={len(unit.get('sam_prompt_groups', []))}",
                                    flush=True,
                                )
                                started = time.monotonic()
                                started_at = datetime.now(timezone.utc).isoformat()
                                try:
                                    record = process_unit(
                                        unit,
                                        provider=provider,
                                        predictor=predictor,
                                        output_threshold=output_threshold,
                                        shared_frame_paths=shared_frames,
                                        shared_session_id=session_id,
                                        prompt_cache=prompt_cache,
                                    )
                                    record["runtime"] = {
                                        "started_at": started_at,
                                        "finished_at": datetime.now(timezone.utc).isoformat(),
                                        "seconds": round(time.monotonic() - started, 3),
                                        "worker": owner,
                                        "video_session_reused": session_id is not None,
                                    }
                                    atomic_json(output_path, record)
                                    error_path.unlink(missing_ok=True)
                                    committed_by_worker += 1
                                    print(
                                        f"[committed] {unit['sample_id']} "
                                        f"disposition={record['disposition']} "
                                        f"tracklets={len(record['tracklets'])}",
                                        flush=True,
                                    )
                                except Exception as error:
                                    attempt = _error_attempts(error_path) + 1
                                    atomic_json(
                                        error_path,
                                        {
                                            "sample_id": unit["sample_id"],
                                            "dataset": unit["dataset"],
                                            "split": unit["split"],
                                            "cohort": unit["cohort"],
                                            "stage": "tracklet_generation",
                                            "attempt": attempt,
                                            "max_attempts": max_error_attempts,
                                            "error_type": type(error).__name__,
                                            "message": str(error),
                                            "traceback": traceback.format_exc(),
                                            "failed_at": datetime.now(timezone.utc).isoformat(),
                                        },
                                    )
                                    batch_failed = True
                                    print(
                                        f"[failed] {unit['sample_id']} attempt={attempt} "
                                        f"{type(error).__name__}: {error}",
                                        flush=True,
                                    )
                                    # A SAM exception can poison its session. Release the
                                    # video and retry remaining instructions in a fresh pass.
                                    break
                    finally:
                        if session_id is not None:
                            predictor.handle_request(
                                {
                                    "type": "close_session",
                                    "session_id": session_id,
                                    "run_gc_collect": False,
                                }
                            )
                        provider.finish_video()
                    if batch_failed:
                        continue
            if STOP_REQUESTED or (
                max_samples is not None and committed_by_worker >= max_samples
            ):
                break
            if not saw_pending:
                break
            if not claimed_video:
                # Other workers own the remaining leases. Recheck rather than
                # exiting early and leaving a tail after faster workers finish.
                time.sleep(10)
            else:
                time.sleep(0.2)
    finally:
        provider.close()
        if predictor is not None and hasattr(predictor, "shutdown"):
            predictor.shutdown()
    return 99 if STOP_REQUESTED else 0


def install_signal_handlers() -> None:
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
