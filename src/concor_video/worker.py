"""Checkpointable long-lived worker."""

from __future__ import annotations

import json
import os
import signal
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpointing import atomic_json, sample_claim
from .pipeline import DatasetProvider, build_predictor, process_unit


STOP_REQUESTED = False


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[drain] received signal {signum}; finishing the active sample", flush=True)


def _requires_sam(unit: dict[str, Any]) -> bool:
    return bool(
        not unit["negative"]
        and (
            unit["target_source_expected"] != "official_dataset_ground_truth"
            or unit.get("sam_prompt_groups")
        )
    )


def run_worker(
    *,
    worklist_path: Path,
    campaign_root: Path,
    checkpoint: Path,
    shard_index: int,
    shard_count: int,
    output_threshold: float,
    compile_model: bool,
    warm_up: bool,
    use_fa3: bool,
    retry_errors: bool,
    max_samples: int | None,
) -> int:
    worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    records_dir = campaign_root / "records"
    errors_dir = campaign_root / "errors"
    claims_dir = campaign_root / "claims"
    cache_root = campaign_root / "cache"
    units = [
        unit
        for index, unit in enumerate(worklist["units"])
        if index % shard_count == shard_index
    ]
    remaining = [
        unit
        for unit in units
        if not (records_dir / f"{unit['sample_id']}.json").is_file()
        and (retry_errors or not (errors_dir / f"{unit['sample_id']}.json").is_file())
    ]
    if max_samples is not None:
        remaining = remaining[:max_samples]
    print(
        f"[worker] shard={shard_index}/{shard_count} assigned={len(units)} "
        f"remaining={len(remaining)}",
        flush=True,
    )
    if not remaining:
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
    try:
        for ordinal, unit in enumerate(remaining, 1):
            if STOP_REQUESTED:
                break
            output_path = records_dir / f"{unit['sample_id']}.json"
            error_path = errors_dir / f"{unit['sample_id']}.json"
            with sample_claim(claims_dir / f"{unit['sample_id']}.claim", owner=owner) as owned:
                if not owned or output_path.is_file():
                    continue
                if _requires_sam(unit) and predictor is None:
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"missing SAM3.1 checkpoint: {checkpoint}")
                    print("[model] loading SAM3.1 Multiplex", flush=True)
                    predictor = build_predictor(
                        checkpoint_path=str(checkpoint),
                        compile_model=compile_model,
                        warm_up=warm_up,
                        use_fa3=use_fa3,
                    )
                print(
                    f"[sample {ordinal}/{len(remaining)}] {unit['sample_id']} "
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
                    )
                    record["runtime"] = {
                        "started_at": started_at,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "seconds": round(time.monotonic() - started, 3),
                        "worker": owner,
                    }
                    atomic_json(output_path, record)
                    error_path.unlink(missing_ok=True)
                    print(
                        f"[committed] {unit['sample_id']} disposition={record['disposition']} "
                        f"tracklets={len(record['tracklets'])}",
                        flush=True,
                    )
                except Exception as error:
                    atomic_json(
                        error_path,
                        {
                            "sample_id": unit["sample_id"],
                            "dataset": unit["dataset"],
                            "split": unit["split"],
                            "cohort": unit["cohort"],
                            "stage": "tracklet_generation",
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    print(
                        f"[failed] {unit['sample_id']} {type(error).__name__}: {error}",
                        flush=True,
                    )
    finally:
        provider.close()
        if predictor is not None and hasattr(predictor, "shutdown"):
            predictor.shutdown()
    return 99 if STOP_REQUESTED else 0


def install_signal_handlers() -> None:
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
