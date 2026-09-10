"""Command-line interface for preparation, processing, export, and verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .exporter import export_campaign
from .worker import install_signal_handlers, run_worker
from .worklist import (
    build_release_worklist,
    build_worklist,
    save_worklist,
)


def _positive_or_none(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="concor-video",
        description="Generate tracklet-text correspondences with SAM3.1",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("build-worklist", help="select and parse one split")
    prepare.add_argument("--dataset", choices=["refytvos", "revos"], required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--campaign-root", type=Path, required=True)
    prepare.add_argument("--campaign-id")
    prepare.add_argument(
        "--revos-categories",
        nargs="+",
        choices=["implicit", "explicit", "nonexistent"],
        default=["implicit", "explicit", "nonexistent"],
    )
    prepare.add_argument("--limit", type=_positive_or_none)
    prepare.add_argument("--limit-per-category", type=_positive_or_none)
    prepare.add_argument("--seed", type=int, default=20260909)
    prepare.add_argument("--one-expression-per-video", action="store_true")

    release = commands.add_parser(
        "build-release-worklist",
        help="build the exact public evaluation or train corpus",
    )
    release.add_argument("--scope", choices=["evaluation", "train"], required=True)
    release.add_argument("--refytvos-root", type=Path, required=True)
    release.add_argument("--revos-root", type=Path, required=True)
    release.add_argument("--campaign-root", type=Path, required=True)
    release.add_argument("--campaign-id")
    release.add_argument("--seed", type=int, default=20260909)
    release.add_argument(
        "--verify-official-counts", action=argparse.BooleanOptionalAction, default=True
    )

    mask_index = commands.add_parser(
        "index-revos-masks", help="build the shared read-only ReVOS mask index"
    )
    mask_index.add_argument("--source", type=Path, required=True)
    mask_index.add_argument("--output", type=Path, required=True)

    process = commands.add_parser("process", help="run one checkpointable worker")
    process.add_argument("--worklist", type=Path, required=True)
    process.add_argument("--campaign-root", type=Path, required=True)
    process.add_argument("--checkpoint", type=Path, required=True)
    process.add_argument("--frame-cache-root", type=Path)
    process.add_argument(
        "--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    process.add_argument("--shard-count", type=int, default=1)
    process.add_argument("--output-threshold", type=float, default=0.5)
    process.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    process.add_argument("--warm-up", action=argparse.BooleanOptionalAction, default=False)
    process.add_argument("--fa3", action=argparse.BooleanOptionalAction, default=False)
    process.add_argument("--retry-errors", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--max-error-attempts", type=int, default=3)
    process.add_argument("--max-samples", type=_positive_or_none)

    export = commands.add_parser("export", help="write Parquet tables and ledger")
    export.add_argument("--worklist", type=Path, required=True)
    export.add_argument("--campaign-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path)

    verify = commands.add_parser("verify", help="run the offline video verification UI")
    verify.add_argument("--parquet", type=Path, action="append", required=True)
    verify.add_argument("--decisions", type=Path)
    verify.add_argument("--media-root", type=Path, action="append", default=[])
    verify.add_argument("--host", default="127.0.0.1")
    verify.add_argument("--port", type=int, default=8000)
    verify.add_argument("--output", type=Path, default=Path("verified.parquet"))
    return parser


def _default_campaign_id(args: argparse.Namespace) -> str:
    cohort = "full-video" if args.dataset == "refytvos" else "-".join(args.revos_categories)
    return f"{args.dataset}-{args.split}-{cohort}-seed{args.seed}"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "build-worklist":
        campaign_root = args.campaign_root.resolve()
        campaign_id = args.campaign_id or _default_campaign_id(args)
        value = build_worklist(
            dataset=args.dataset,
            data_root=args.data_root,
            split=args.split,
            campaign_id=campaign_id,
            seed=args.seed,
            revos_categories=set(args.revos_categories),
            limit=args.limit,
            limit_per_category=args.limit_per_category,
            one_expression_per_video=args.one_expression_per_video,
        )
        path = campaign_root / "worklist.json"
        save_worklist(path, value)
        print(
            json.dumps(
                {"worklist": str(path), "counts": value["counts"], "videos": value["video_counts"]},
                indent=2,
            )
        )
        return
    if args.command == "build-release-worklist":
        campaign_root = args.campaign_root.resolve()
        campaign_id = args.campaign_id or f"public-{args.scope}-seed{args.seed}"
        value = build_release_worklist(
            scope=args.scope,
            refytvos_root=args.refytvos_root,
            revos_root=args.revos_root,
            campaign_id=campaign_id,
            seed=args.seed,
            verify_official_counts=args.verify_official_counts,
        )
        path = campaign_root / "worklist.json"
        save_worklist(path, value)
        print(
            json.dumps(
                {"worklist": str(path), "counts": value["counts"], "videos": value["video_counts"]},
                indent=2,
            )
        )
        return
    if args.command == "index-revos-masks":
        from .mask_index import index_revos_masks

        count = index_revos_masks(args.source.resolve(), args.output.resolve())
        print(json.dumps({"mask_sequences": count, "output": str(args.output.resolve())}))
        return
    if args.command == "process":
        if not 0 <= args.shard_index < args.shard_count:
            parser.error("--shard-index must satisfy 0 <= index < shard-count")
        install_signal_handlers()
        code = run_worker(
            worklist_path=args.worklist.resolve(),
            campaign_root=args.campaign_root.resolve(),
            checkpoint=args.checkpoint.resolve(),
            frame_cache_root=(args.frame_cache_root.resolve() if args.frame_cache_root else None),
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            output_threshold=args.output_threshold,
            compile_model=args.compile,
            warm_up=args.warm_up,
            use_fa3=args.fa3,
            retry_errors=args.retry_errors,
            max_error_attempts=args.max_error_attempts,
            max_samples=args.max_samples,
        )
        raise SystemExit(code)
    if args.command == "export":
        output_dir = args.output_dir or (args.campaign_root / "export")
        manifest = export_campaign(
            worklist_path=args.worklist.resolve(),
            campaign_root=args.campaign_root.resolve(),
            output_dir=output_dir.resolve(),
        )
        print(json.dumps(manifest, indent=2))
        return
    if args.command == "verify":
        from .verification import serve_verification

        serve_verification(
            parquet_paths=[path.resolve() for path in args.parquet],
            decisions_path=args.decisions.resolve() if args.decisions else None,
            media_roots=[path.resolve() for path in args.media_root],
            host=args.host,
            port=args.port,
            output_path=args.output.resolve(),
        )
        return
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
