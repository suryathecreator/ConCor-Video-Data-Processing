"""Command-line interface for worklist construction, processing, and export."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .exporter import export_campaign
from .worker import install_signal_handlers, run_worker
from .worklist import build_worklist, save_worklist


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

    prepare = commands.add_parser("build-worklist", help="select and parse dataset expressions")
    prepare.add_argument("--dataset", choices=["refytvos", "revos"], required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--campaign-root", type=Path, required=True)
    prepare.add_argument("--campaign-id")
    prepare.add_argument(
        "--ref-mode", choices=["full_video", "first_frame"], default="full_video"
    )
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

    process = commands.add_parser("process", help="run one checkpointable worker shard")
    process.add_argument("--worklist", type=Path, required=True)
    process.add_argument("--campaign-root", type=Path, required=True)
    process.add_argument("--checkpoint", type=Path, required=True)
    process.add_argument(
        "--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    process.add_argument("--shard-count", type=int, default=1)
    process.add_argument("--output-threshold", type=float, default=0.5)
    process.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    process.add_argument("--warm-up", action=argparse.BooleanOptionalAction, default=False)
    process.add_argument("--fa3", action=argparse.BooleanOptionalAction, default=False)
    process.add_argument("--retry-errors", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--max-samples", type=_positive_or_none)

    export = commands.add_parser("export", help="write normalized Parquet tables and ledger")
    export.add_argument("--worklist", type=Path, required=True)
    export.add_argument("--campaign-root", type=Path, required=True)
    export.add_argument("--output-dir", type=Path)
    return parser


def _default_campaign_id(args: argparse.Namespace) -> str:
    cohort = args.ref_mode if args.dataset == "refytvos" else "-".join(args.revos_categories)
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
            ref_mode=args.ref_mode,
            revos_categories=set(args.revos_categories),
            limit=args.limit,
            limit_per_category=args.limit_per_category,
            one_expression_per_video=args.one_expression_per_video,
        )
        path = campaign_root / "worklist.json"
        save_worklist(path, value)
        print(json.dumps({"worklist": str(path), "counts": value["counts"]}, indent=2))
        return
    if args.command == "process":
        if not 0 <= args.shard_index < args.shard_count:
            parser.error("--shard-index must satisfy 0 <= index < shard-count")
        install_signal_handlers()
        code = run_worker(
            worklist_path=args.worklist.resolve(),
            campaign_root=args.campaign_root.resolve(),
            checkpoint=args.checkpoint.resolve(),
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            output_threshold=args.output_threshold,
            compile_model=args.compile,
            warm_up=args.warm_up,
            use_fa3=args.fa3,
            retry_errors=args.retry_errors,
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
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
