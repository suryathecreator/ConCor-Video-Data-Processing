#!/usr/bin/env python3
"""Bounded, density-stratified NLP-then-visual refinement pilot (never full corpus)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
from functools import lru_cache
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import cv2

from refine_hf import (APIClient, Ledger, MODEL, ROOT, Source, apply_result, atomic_json,
                       digest, discover_inputs, dumps, estimated_max_cost, export, loads,
                       inspect_row, prepare, prepare_nlp, read_prompt, release_memory, run_regular, track_signature, video_key)
from concor_video.verification import _candidate_media_sources

DENSITIES = ("0–2", "3–5", "6–12", "13+")


def candidate_count(row):
    return len({tid for group in loads(row["groups_json"], []) for tid in group.get("tracklet_ids", [])})


def density_bin(count):
    return DENSITIES[0 if count <= 2 else 1 if count <= 5 else 2 if count <= 12 else 3]


def available_rows(source, roots):
    """Restrict metadata sampling to available source archives or frame folders."""
    @lru_cache(maxsize=None)
    def archives(frame_source, dataset_root):
        row = {"frame_source": frame_source, "dataset_root": dataset_root}
        paths = _candidate_media_sources(roots, row)
        return any(p.is_file() and p.suffix.lower() in {".zip", ".tar"} for p in paths), [p for p in paths if p.is_dir()]
    @lru_cache(maxsize=None)
    def available(frame_source, dataset_root, video_id):
        archive, folders = archives(frame_source, dataset_root)
        return archive or any((p / prefix / video_id).is_dir() for p in folders
                              for prefix in ("", "JPEGImages", "train/JPEGImages", "valid/JPEGImages", "val/JPEGImages", "test/JPEGImages"))
    result = {}
    for sid, row in source.rows.items():
        if available(row.get("frame_source"), row.get("dataset_root"), row["video_id"]):
            result[sid] = row
    return result


def select_videos(source, per_dataset, seed, include_video, expressions_per_video=1):
    """Equal coverage of annotation candidate-count bins; featured videos count in quota."""
    if not 1 <= per_dataset <= 100 or expressions_per_video < 0:
        raise ValueError("sample wrapper allows 1..100 videos per dataset, never a full-corpus run")
    by_video = defaultdict(list)
    for row in source.rows.values():
        if row["dataset"] in {"ref_youtube_vos", "revos"}:
            by_video[video_key(row)].append(row)
    representatives = {}
    for key, rows in by_video.items():
        representatives[key] = sorted(rows, key=lambda r: (-candidate_count(r), digest([seed, r["sample_id"]])))
    missing = set(include_video) - {key[2] for key in by_video}
    if missing:
        raise ValueError(f"requested featured videos missing: {sorted(missing)}")
    records, selected_ids = [], []
    for dataset in ("ref_youtube_vos", "revos"):
        keys = [k for k in by_video if k[0] == dataset]
        if len(keys) < per_dataset:
            raise ValueError(f"not enough distinct {dataset} videos")
        chosen = sorted((k for k in keys if k[2] in include_video), key=lambda k: include_video.index(k[2]))
        if len(chosen) > per_dataset:
            raise ValueError("featured videos exceed dataset quota")
        for i, label in enumerate(DENSITIES):
            quota = per_dataset // 4 + (i < per_dataset % 4)
            have = sum(density_bin(candidate_count(representatives[k][0])) == label for k in chosen)
            options = sorted((k for k in keys if k not in chosen and
                              density_bin(candidate_count(representatives[k][0])) == label),
                             key=lambda k: digest([seed, k]))
            chosen.extend(options[:max(0, quota - have)])
        # A small/absent stratum borrows deterministically from the remaining pool.
        remaining = sorted((k for k in keys if k not in chosen), key=lambda k: digest([seed, k]))
        chosen = (chosen + remaining)[:per_dataset]
        for key in chosen:
            rows = representatives[key]
            rows = rows[:expressions_per_video] if expressions_per_video else rows
            selected_ids.extend(r["sample_id"] for r in rows)
            records.append({"dataset": key[0], "split": key[1], "video_id": key[2],
                            "candidate_tracks": candidate_count(representatives[key][0]),
                            "density_bin": density_bin(candidate_count(representatives[key][0])),
                            "featured": key[2] in include_video, "sample_ids": [r["sample_id"] for r in rows]})
    records.sort(key=lambda r: (not r["featured"], r["dataset"], DENSITIES.index(r["density_bin"]), r["video_id"]))
    selected_ids = [sid for record in records for sid in record["sample_ids"]]
    return selected_ids, records


class CleanedSource:
    def __init__(self, source, output):
        self.source, self.output = source, output
        self.rows, self.schema = source.rows, source.schema
        self.pairs = {}
        preflight = output.parent / "preflight"
        if (preflight / "requests.json").exists():
            for manifest in load_packets(preflight):
                packet = manifest["packet"]
                key = tuple(packet["video"]), tuple(sorted(t["id"] for t in packet["tracklets"]))
                for index, image, size in zip(packet["selected_frame_indices"], manifest["images"], manifest["dimensions"]):
                    self.pairs[key + (index,)] = (preflight / image, size)

    def cached_pair(self, key, used, index):
        # Reuse only an EXACT candidate set with stable IDs and the same frame.
        # Removed NLP candidates must never survive in the model's overlay.
        return self.pairs.get((key, tuple(used), index))

    def row(self, sid):
        result = loads((self.output / "decisions" / (digest(sid) + ".json")).read_text())
        if result["decision"] not in {"keep", "rewrite"}:
            raise ValueError("visual pass requires a validated NLP result")
        return apply_result(self.source.row(sid), result, result["mapping"])

    def visual_registry(self, key):
        # Keep numeric colors/IDs stable between the before and after audit views.
        registry = {}
        for sid in sorted(sid for sid, r in self.rows.items() if video_key(r) == key):
            row = self.source.row(sid)
            inspection, reason = inspect_row(row, 64)
            if reason:
                continue
            for track in inspection[0]:
                signature = track_signature(loads(row["frame_ids_json"]), track)
                if signature not in registry:
                    registry[signature] = len(registry) + 1
        return registry


def load_packets(output):
    manifests = [loads((output / "packets" / rid / "input.json").read_text())
                 for rid in loads((output / "requests.json").read_text())]
    return [m for m in manifests if not m.get("validation_retry_of") and not m.get("transport_retry_of")]


def main():
    # Rendering is sequential; large OpenCV thread pools waste memory on dense
    # clips, especially on shared machines with many visible CPU cores.
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, action="append", required=True)
    parser.add_argument("--videos-per-dataset", type=int, default=100)
    parser.add_argument("--expressions-per-video", type=int, default=1, help="1 gives exactly 200 examples; 0 explicitly selects all expressions of selected videos")
    parser.add_argument("--include-video", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--mode", choices=["prepare", "regular", "export"], default="prepare")
    parser.add_argument("--budget-usd", type=float, default=5.)
    parser.add_argument("--prior-spend-usd", type=float, default=0., help="prior charges/reservations outside this run, included in the $100 total cap")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recover-interrupted", action="store_true", help="after a confirmed interruption, allow one new request per uncertain reservation; retain the original maximum charge")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.budget_usd <= 0 or args.prior_spend_usd < 0 or args.budget_usd + args.prior_spend_usd > 100:
        parser.error("workers must be 1..8 and run budget plus prior spending must fit within $100")
    args.group_size, args.group_tracklets, args.max_tracklets, args.sample_count = 12, 24, 64, 12
    args.media_root = [p.resolve() for p in args.media_root]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata, prompt = read_prompt(ROOT / "prompts/video_refinement.md")
    nlp_metadata, nlp_prompt = read_prompt(ROOT / "prompts/nlp_cleanup.md")
    original = Source(discover_inputs(args.input), include_media=True)
    eligible = available_rows(original, args.media_root)
    from types import SimpleNamespace
    ids, records = select_videos(SimpleNamespace(rows=eligible), args.videos_per_dataset, args.seed, args.include_video, args.expressions_per_video)
    configuration = {"version": "concor-two-pass-sample-v1", "model": MODEL, "reasoning_effort": "medium",
                     "prompt_sha256": digest(prompt), "nlp_prompt_sha256": digest(nlp_prompt), "selected_ids": ids,
                     "sources": [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in original.paths],
                     "sampling": {"seed": args.seed, "videos_per_dataset": args.videos_per_dataset,
                                  "expressions_per_video": args.expressions_per_video, "include_video": args.include_video},
                     "prior_spend_usd": args.prior_spend_usd}
    path = output / "run.json"
    if path.exists() and loads(path.read_text()) != configuration:
        raise ValueError("output belongs to different inputs/prompts/selection; use a new directory")
    atomic_json(path, configuration)
    atomic_json(output / "selection.json", {"density_measure": "Maximum linked candidate-track count in an expression of the video, before cleanup; an annotation density proxy, not a verified count of real objects.",
                                           "media_eligibility": "Sampled only from locally available source archives or frame directories, before any paid calls.",
                                           "videos_without_local_source": len({video_key(r) for sid, r in original.rows.items() if sid not in eligible}),
                                           "videos": records})
    snapshot = output / "input" / "selected.parquet"
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(str(snapshot) + ".part", original.schema, compression="zstd") as writer:
            for sid in sorted(ids, key=lambda sid: original.locations[sid]):
                writer.write_table(pa.Table.from_pylist([original.row(sid)], schema=original.schema))
        Path(str(snapshot) + ".part").replace(snapshot)
    original.row_group.cache_clear()
    del original, eligible
    release_memory()
    source = Source([snapshot])
    ledger = Ledger(output / "budget.sqlite", args.budget_usd)
    client = None
    nlp_dir, preflight_dir = output / "nlp", output / "preflight"
    for directory, document, header in [(output, prompt, metadata), (nlp_dir, nlp_prompt, nlp_metadata)]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prompt_sent.txt").write_text(document)
        (directory / "prompt_metadata.txt").write_text(header)
    try:
        if (nlp_dir / "requests.json").exists() and (nlp_dir / "heuristics.json").exists():
            nlp_packets = load_packets(nlp_dir)
            early = loads((nlp_dir / "heuristics.json").read_text())
        else:
            nlp_packets, early = prepare_nlp(source, ids, args, nlp_prompt, nlp_dir)
        if (preflight_dir / "requests.json").exists():
            originals = load_packets(preflight_dir)
        else:
            # Human audit retains the original noisy candidates, including cases
            # rejected by the per-entity ceiling before any paid call.
            audit_args = SimpleNamespace(**vars(args), max_entity_tracks=None)
            originals, _ = prepare(source, ids, audit_args, prompt, preflight_dir)
        # Include the one allowed retry for every request, even though uncommon.
        bound = 2 * (sum(estimated_max_cost(p, nlp_prompt) for p in nlp_packets) +
                     sum(estimated_max_cost(p, prompt) for p in originals))
        uncertain = ledger.db.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE state='reserved'").fetchone()[0]
        if args.recover_interrupted:
            bound += uncertain
        preflight = {"videos": len(records), "expressions": len(ids), "nlp_requests": len(nlp_packets),
                     "visual_requests_before_cleanup": len(originals), "including_retry_upper_usd": round(bound, 6),
                     "uncertain_reservations_upper_usd": round(uncertain, 6),
                     "budget_usd": args.budget_usd, "prior_spend_usd": args.prior_spend_usd}
        atomic_json(output / "preflight.json", preflight)
        print(dumps(preflight), flush=True)
        if args.mode == "prepare":
            return
        if bound > args.budget_usd:
            raise RuntimeError("conservative two-pass preflight exceeds run budget; no new calls sent")
        if args.mode == "regular":
            client = APIClient()
            release_memory()
            run_regular(nlp_packets, nlp_prompt, source, nlp_dir, ledger, client, workers=args.workers,
                        recover_interrupted=args.recover_interrupted)
        ready, nlp_counts, removed_tracks, structural_removed = [], Counter(), 0, 0
        for sid in ids:
            decision = nlp_dir / "decisions" / (digest(sid) + ".json")
            result = loads(decision.read_text()) if decision.exists() else early.get(sid, {"sample_id": sid, "decision": "pending", "stage": "nlp", "reason_codes": []})
            nlp_counts[result["decision"]] += 1
            if result["decision"] in {"keep", "rewrite"}:
                cleaned = apply_result(source.row(sid), result, result["mapping"])
                inspected, _ = inspect_row(source.row(sid), args.max_tracklets)
                structural_removed += len(loads(source.row(sid)["tracklets_json"])) - len(inspected[0])
                removed_tracks += len(inspected[0]) - len(loads(cleaned["tracklets_json"]))
                ready.append(sid)
                cleaned = inspected = None
            else:
                early[sid] = result
        cleaned_source = CleanedSource(source, nlp_dir)
        if args.mode == "regular":
            if (output / "requests.json").exists() and (output / "heuristics.json").exists():
                packets = load_packets(output)
                # Recovery packets are reached via their original request, never
                # scheduled again as independent work on resume.
                packets = [p for p in packets if not p.get("transport_retry_of")]
                visual_early = loads((output / "heuristics.json").read_text())
            else:
                packets, visual_early = prepare(cleaned_source, ready, args, prompt, output)
            early.update(visual_early)
            atomic_json(output / "heuristics.json", early)
            release_memory()
            run_regular(packets, prompt, cleaned_source, output, ledger, client, workers=args.workers,
                        recover_interrupted=args.recover_interrupted)
        else:
            early.update(loads((output / "heuristics.json").read_text()) if (output / "heuristics.json").exists() else {})
        summary = export(cleaned_source, ids, output, early, ledger)
        summary.update(videos=len(records), videos_by_dataset=dict(Counter(r["dataset"] for r in records)),
                       nlp_counts=dict(nlp_counts), nlp_removed_tracks=removed_tracks,
                       structural_removed_tracks=structural_removed,
                       uncertain_cost_upper_usd=round(ledger.db.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE state='reserved'").fetchone()[0], 6),
                       total_with_prior_upper_usd=round(ledger.total() + args.prior_spend_usd, 6))
        atomic_json(output / "summary.json", summary)
        print(dumps(summary), flush=True)
    finally:
        if client is not None:
            client.close()
        ledger.db.close()
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, requests.RequestException) as exc:
        print("Sample refinement stopped: " + re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(exc)), file=sys.stderr)
        raise SystemExit(1)
