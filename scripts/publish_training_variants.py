#!/usr/bin/env python3
"""Publish validated GPT training variants and archive the original training export."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from string import Template

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from publish_refined_hf import COLUMNS, public_table, read_token
from refine_hf import ROOT, atomic_json, loads, release_memory

REPO = "suryadv/concor-video-data-filtered"
MAPPING = {key: ("disposition" if key == "disposition" else value)
           for key, value in COLUMNS.items()}
ARCHIVE = {"train/verification.parquet": "train_old/verification.parquet",
           "train/manifest.json": "train_old/manifest.json"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publication_inputs(run, exclude_errors=False):
    summary = loads((run / "summary.json").read_text())
    audit = loads((run / "audit.json").read_text())
    config = loads((run / "run.json").read_text())
    decisions = {r["sample_id"]: r["decision"] for r in audit}
    counts = Counter(decisions.values())
    expected_counts = Counter({k: v for k, v in summary["counts"].items() if v})
    if (len(decisions) != len(audit) or len(audit) != summary["selected"]
            or len(config["selected_ids"]) != len(decisions)
            or set(decisions) != set(config["selected_ids"]) or counts != expected_counts):
        raise ValueError("audit coverage/count mismatch")
    if set(counts) - {"keep", "rewrite", "reject", "error"}:
        raise ValueError("pending/unknown outcomes cannot be published")
    if counts["error"] and not exclude_errors:
        raise ValueError("explicit --exclude-unresolved-errors is required")
    if not summary["complete"] and not counts["error"]:
        raise ValueError("run is incomplete for an unexplained reason")
    if config["model"] != "gpt-6-luna" or config["reasoning_effort"] != "medium":
        raise ValueError("unexpected model configuration")
    return summary, decisions, config


def training_table(table, decisions):
    result = public_table(table, decisions)
    names = ["disposition" if n == "annotation_status" else n for n in result.column_names]
    result = result.rename_columns(names)
    metadata = dict(result.schema.metadata or {})
    metadata[b"original_column_mapping"] = json.dumps(MAPPING, sort_keys=True).encode()
    return result.replace_schema_metadata(metadata)


def build_variants(run, destination, decisions):
    destination.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(run / "train.parquet")
    expected = {sid for sid, decision in decisions.items() if decision in {"keep", "rewrite"}}
    seen, writers = set(), {}
    stats = {name: Counter() for name in ["train_all", "train_filtered"]}
    try:
        for batch in source.iter_batches(batch_size=32, use_threads=False):
            table = pa.Table.from_batches([batch])
            ids = table["sample_id"].to_pylist()
            if (len(set(ids)) != len(ids) or seen.intersection(ids)
                    or not set(ids) <= expected
                    or any(s != "train" for s in table["split"].to_pylist())):
                raise ValueError("duplicate, unresolved, or non-training row in export")
            seen.update(ids)
            public = training_table(table, decisions)
            filtered = public.filter(pc.equal(public["disposition"], "complete_bcc"))
            for name, part in [("train_all", public), ("train_filtered", filtered)]:
                if name not in writers:
                    writers[name] = pq.ParquetWriter(destination / (name + ".parquet"),
                                                    public.schema, compression="zstd")
                writers[name].write_table(part, row_group_size=32)
                stats[name].update(rows=part.num_rows,
                                   tracklets=sum(part["object_count"].to_pylist()),
                                   span_links=sum(part["linked_phrase_count"].to_pylist()),
                                   negative=sum(part["is_nonexistent"].to_pylist()),
                                   literal_s_rows=sum("(s)" in t for t in part["caption"].to_pylist()))
            if len(seen) % 4096 == 0:
                print(f"Prepared {len(seen):,} validated expressions", flush=True)
            del batch, table, public, filtered, part
            release_memory()
    finally:
        for writer in writers.values():
            writer.close()
    if seen != expected or not stats["train_all"]["rows"]:
        raise ValueError("export does not exactly match all keep/rewrite IDs")
    return {k: dict(v) for k, v in stats.items()}


def source_statistics(source, retained, cache, source_hash):
    """Count source mask sequences; cache only against exact source bytes and IDs."""
    identity = {"source_sha256": source_hash,
                "retained_ids_sha256": hashlib.sha256(
                    json.dumps(sorted(retained), separators=(",", ":")).encode()).hexdigest()}
    if cache.exists():
        saved = loads(cache.read_text())
        if saved.get("identity") == identity:
            return saved["statistics"]
    total, kept = Counter(), Counter()
    columns = ["sample_id", "text", "negative", "tracklets_json", "span_links_json"]
    for batch in pq.ParquetFile(source).iter_batches(batch_size=32, columns=columns, use_threads=False):
        for row in batch.to_pylist():
            values = {"rows": 1, "tracklets": len(loads(row["tracklets_json"])),
                      "span_links": len(loads(row["span_links_json"])),
                      "negative": int(row["negative"]), "literal_s_rows": int("(s)" in row["text"])}
            total.update(values)
            if row["sample_id"] in retained:
                kept.update(values)
    if kept["rows"] != len(retained):
        raise ValueError("retained IDs are not represented exactly once in the source")
    statistics = {"train_old": dict(total), "source_rows_retained_after_gpt": dict(kept)}
    atomic_json(cache, {"identity": identity, "statistics": statistics})
    return statistics


def file_identity(f):
    return {"size": f.size, "blob_id": f.blob_id,
            "sha256": f.lfs.sha256 if f.lfs else None}


def costs(run, config):
    values = Counter()
    for path in [run / "budget.sqlite", *map(Path, config.get("prior_ledgers", []))]:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
            values.update({state: amount for state, amount in
                           db.execute("SELECT state, SUM(amount) FROM charges GROUP BY state")})
    return {"completed_usd": values["completed"],
            "accounted_usd": sum(values.values()),
            "reserved_usd": sum(v for k, v in values.items() if k != "completed")}


def make_card(original, manifest):
    header, _ = original[4:].split("\n---", 1)
    meta = yaml.safe_load(header) or {}
    meta["configs"] = [
        {"config_name": "gpt_refined", "default": True, "data_files": [
            {"split": n, "path": n + ".parquet"} for n in ["train_filtered", "train_all"]]},
        {"config_name": "original", "data_files": [
            {"split": "train_old", "path": "train_old/verification.parquet"},
            {"split": "eval_unfiltered", "path": "eval/unfiltered/verification.parquet"},
            {"split": "eval_filtered", "path": "eval/filtered/verified.parquet"}]}]
    for tag in ["gpt-refined", "video-captioning"]:
        if tag not in meta.setdefault("tags", []):
            meta["tags"].append(tag)
    s, m = manifest["statistics"], manifest
    substitutions = {}
    for split in ["train_old", "train_all", "train_filtered"]:
        for key in ["rows", "tracklets", "span_links", "literal_s_rows"]:
            substitutions[split + "_" + key] = f"{s[split][key]:,}"
    for key in ["completed_usd", "accounted_usd", "reserved_usd"]:
        substitutions[key] = f"{m['costs'][key]:.2f}"
    substitutions.update(
        run_cost=f"{m['summary']['cost_upper_usd']:.2f}",
        expected_cost=f"{m['estimate']['expected_batch_usd']:.2f}",
        buffered_cost=f"{m['estimate']['buffered_total_usd']:.2f}",
        excluded_errors=f"{m['summary']['counts'].get('error', 0):,}",
        rejected=f"{m['summary']['counts'].get('reject', 0):,}",
        removed_tracks=f"{s['train_old']['tracklets'] - s['train_all']['tracklets']:,}",
        removed_retained_tracks=f"{s['source_rows_retained_after_gpt']['tracklets'] - s['train_all']['tracklets']:,}",
        retained_before_tracks=f"{s['source_rows_retained_after_gpt']['tracklets']:,}",
        mask_reduction=f"{100 * (1 - s['train_all']['tracklets'] / s['train_old']['tracklets']):.1f}",
    )
    body = Template((ROOT / "docs/hf_training_card.md").read_text()).substitute(substitutions)
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n\n" + body


def prepare(run, source, repo=REPO, exclude_errors=False):
    from huggingface_hub import HfApi, hf_hub_download
    summary, decisions, config = publication_inputs(run, exclude_errors)
    token = read_token()
    api = HfApi(token=token)
    who = api.whoami()
    if who.get("auth", {}).get("accessToken", {}).get("role") == "read":
        raise ValueError("write access is required")
    info = api.dataset_info(repo, files_metadata=True)
    before = {f.rfilename: file_identity(f) for f in info.siblings}
    if not set(ARCHIVE) <= set(before) or set(ARCHIVE.values()) & set(before):
        raise ValueError("unexpected training archive layout")
    if any(n + ".parquet" in before for n in ["train_all", "train_filtered"]):
        raise ValueError("training variants already exist; use the saved publication manifest")
    source_hash = sha256(source)
    if (source.stat().st_size != before["train/verification.parquet"]["size"]
            or source_hash != before["train/verification.parquet"]["sha256"]):
        raise ValueError("local training source differs from the HF archive")
    destination = run / "publication_training_variants"
    destination.mkdir(parents=True, exist_ok=True)
    original = Path(hf_hub_download(repo, "README.md", repo_type="dataset", revision=info.sha,
                                   token=token)).read_text()
    (destination / "README.before.md").write_text(original)
    built = build_variants(run, destination, decisions)
    retained = {sid for sid, decision in decisions.items() if decision in {"keep", "rewrite"}}
    statistics = source_statistics(source, retained,
                                   destination / "source_statistics.json", source_hash)
    if statistics["train_old"]["rows"] != summary["selected"]:
        raise ValueError("source training count differs from the selected run")
    statistics.update(built)
    manifest = {"repo": repo, "parent_commit": info.sha, "before": before, "summary": summary,
                "statistics": statistics, "column_mapping": MAPPING, "costs": costs(run, config),
                "estimate": loads((run / "estimate.json").read_text()),
                "excluded_unresolved_errors": summary["counts"].get("error", 0),
                "source_sha256": source_hash,
                "prompt_sha256": config["prompt_sha256"], "nlp_prompt_sha256": config["nlp_prompt_sha256"]}
    (destination / "README.md").write_text(make_card(original, manifest))
    manifest["uploads"] = {name: {"size": (destination / name).stat().st_size,
                                 "sha256": sha256(destination / name)}
                           for name in ["train_all.parquet", "train_filtered.parquet", "README.md"]}
    atomic_json(destination / "manifest.json", manifest)
    print(json.dumps({"prepared": str(destination), "rows": {k: v["rows"] for k, v in built.items()},
                      "costs": manifest["costs"]}), flush=True)
    return manifest


def verify_remote(api, info, manifest, token, destination):
    from huggingface_hub import hf_hub_download
    files = {f.rfilename: file_identity(f) for f in info.siblings}
    for old, new in ARCHIVE.items():
        if old in files or files.get(new) != manifest["before"][old]:
            raise ValueError("archived training bytes differ")
    for name, identity in manifest["before"].items():
        if name not in ARCHIVE and name != "README.md" and files.get(name) != identity:
            raise ValueError("an unrelated remote file changed")
    for name, expected in manifest["uploads"].items():
        if name == "README.md":
            path = hf_hub_download(manifest["repo"], name, repo_type="dataset",
                                   revision=info.sha, token=token)
            if sha256(path) != expected["sha256"]:
                raise ValueError("published card differs")
        elif (files.get(name, {}).get("size") != expected["size"]
              or files.get(name, {}).get("sha256") != expected["sha256"]):
            raise ValueError("published Parquet bytes differ")
    receipt = {"repo": manifest["repo"], "commit": info.sha,
               "url": f"https://huggingface.co/datasets/{manifest['repo']}",
               "rows": {k: manifest["statistics"][k]["rows"]
                        for k in ["train_all", "train_filtered", "train_old"]}}
    atomic_json(destination / "publication.json", receipt)
    return receipt


def publish(run):
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
    destination = run / "publication_training_variants"
    manifest = loads((destination / "manifest.json").read_text())
    token = read_token()
    api = HfApi(token=token)
    info = api.dataset_info(manifest["repo"], files_metadata=True)
    if info.sha != manifest["parent_commit"]:
        # Recover a successful commit whose response was lost, without re-uploading.
        return verify_remote(api, info, manifest, token, destination)
    for name, expected in manifest["uploads"].items():
        if sha256(destination / name) != expected["sha256"]:
            raise ValueError("prepared publication changed")
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=destination / name)
                  for name in manifest["uploads"]]
    for old, new in ARCHIVE.items():
        operations.extend([CommitOperationCopy(src_path_in_repo=old, path_in_repo=new,
                                               src_revision=info.sha),
                           CommitOperationDelete(path_in_repo=old)])
    commit = api.create_commit(
        repo_id=manifest["repo"], repo_type="dataset", operations=operations,
        parent_commit=info.sha,
        commit_message="Publish GPT-refined train_all and complete-BCC train_filtered; archive train_old",
        commit_description="Two-pass GPT-6 Luna medium cleanup, sampled-video review, exact-span validation. "
                           "Keep disposition; exclude unresolved errors; preserve original training and evaluation bytes. "
                           "Document mask counts, costs, prompt sources and remaining improvements.")
    verified = api.dataset_info(manifest["repo"], revision=commit.oid, files_metadata=True)
    return verify_remote(api, verified, manifest, token, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "data/hf/train/verification.parquet")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--exclude-unresolved-errors", action="store_true")
    parser.add_argument("--publish", action="store_true", help="Publish already prepared, verified artifacts.")
    args = parser.parse_args()
    try:
        from refine_release import load_env
        load_env(ROOT / ".env")
        if args.publish:
            print(json.dumps(publish(args.run.resolve())))
        else:
            prepare(args.run.resolve(), args.source.resolve(), args.repo, args.exclude_unresolved_errors)
    except Exception as exc:
        # Never include response bodies, request headers, environment or credentials.
        raise SystemExit(f"Publication stopped ({type(exc).__name__}); credentials omitted.")
