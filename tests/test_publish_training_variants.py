from collections import Counter
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_training_variants as variants


def record(sid, negative=False):
    row = {key: "" for key in variants.COLUMNS}
    row.update(sample_id=sid, dataset="revos", split="train", video_id="v",
               text="a dog", negative=negative, runtime_seconds=1.0,
               disposition="negative_unsegmentable" if negative else "complete_bcc",
               frame_ids_json='["00001"]', frame_files_json='["00001.jpg"]',
               tracklets_json="[]" if negative else '[{"tracklet_id":"a","masks":[{"size":[1,1],"counts":[0,1]}]}]',
               groups_json="[]", span_links_json="[]" if negative else
               '[{"start":0,"end":5,"text":"a dog","tracklet_ids":["a"]}]')
    return row


def write_audit(run, decisions):
    audit = [{"sample_id": sid, "decision": decision} for sid, decision in decisions.items()]
    for name, value in {
        "audit.json": audit,
        "summary.json": {"complete": "error" not in decisions.values(),
                         "selected": len(decisions), "counts": dict(Counter(decisions.values()))},
        "run.json": {"selected_ids": list(decisions), "model": "gpt-6-luna", "reasoning_effort": "medium"},
    }.items():
        (run / name).write_text(json.dumps(value))


def test_only_explicitly_authorized_errors_can_be_excluded(tmp_path):
    write_audit(tmp_path, {"a": "keep", "b": "error"})
    with pytest.raises(ValueError, match="exclude-unresolved-errors"):
        variants.publication_inputs(tmp_path)
    assert variants.publication_inputs(tmp_path, True)[1] == {"a": "keep", "b": "error"}
    write_audit(tmp_path, {"a": "keep", "b": "pending"})
    with pytest.raises(ValueError, match="pending"):
        variants.publication_inputs(tmp_path, True)


def test_duplicate_audit_ids_are_not_silently_lost(tmp_path):
    write_audit(tmp_path, {"a": "keep", "b": "keep"})
    (tmp_path / "audit.json").write_text(json.dumps([
        {"sample_id": "a", "decision": "keep"}, {"sample_id": "a", "decision": "keep"}]))
    with pytest.raises(ValueError, match="coverage"):
        variants.publication_inputs(tmp_path)


def test_variants_keep_negatives_only_in_all_and_preserve_every_source_column(tmp_path):
    source = pa.Table.from_pylist([record("a"), record("b", True)])
    pq.write_table(source, tmp_path / "train.parquet")
    destination = tmp_path / "prepared"
    stats = variants.build_variants(tmp_path, destination,
                                     {"a": "rewrite", "b": "keep", "error": "error"})
    all_rows = pq.read_table(destination / "train_all.parquet")
    filtered = pq.read_table(destination / "train_filtered.parquet")
    for old, new in variants.MAPPING.items():
        assert source[old].equals(all_rows[new])
    assert all_rows.schema == filtered.schema
    assert filtered["sample_id"].to_pylist() == ["a"]
    assert filtered["disposition"].to_pylist() == ["complete_bcc"]
    assert stats["train_all"]["rows"] == 2
    assert stats["train_filtered"]["negative"] == 0
    assert json.loads(all_rows.schema.metadata[b"original_column_mapping"])["disposition"] == "disposition"


@pytest.mark.parametrize("decisions", [
    {"a": "keep", "b": "keep"},  # Export omitted a successful result.
    {"a": "error"},             # Export contains a failed result.
])
def test_export_identity_mismatch_blocks_publication(tmp_path, decisions):
    pq.write_table(pa.Table.from_pylist([record("a")]), tmp_path / "train.parquet")
    with pytest.raises(ValueError):
        variants.build_variants(tmp_path, tmp_path / "prepared", decisions)


def test_card_exposes_distinct_splits_and_documents_error_exclusions():
    stats = {name: {"rows": 2, "tracklets": 2, "span_links": 2, "literal_s_rows": 1}
             for name in ["train_old", "train_all", "train_filtered", "source_rows_retained_after_gpt"]}
    manifest = {"statistics": stats,
                "summary": {"cost_upper_usd": 1.0, "counts": {"reject": 1, "error": 1}},
                "costs": {"completed_usd": 1, "accounted_usd": 1.1, "reserved_usd": 0.1},
                "estimate": {"expected_batch_usd": 2, "buffered_total_usd": 3}}
    card = variants.make_card("---\nlanguage: en\n---\nOld card.", manifest)
    assert 'path: train_filtered.parquet' in card
    assert 'path: train_old/verification.parquet' in card
    assert '1 unresolved validation errors' in card
    assert '${' not in card
    assert 'USD2' not in card



def test_source_mask_counts_distinguish_row_exclusion_from_track_removal(tmp_path):
    path = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist([record('kept'), record('removed'), record('negative', True)]), path)
    cache = tmp_path / 'counts.json'
    stats = variants.source_statistics(path, {'kept', 'negative'}, cache, variants.sha256(path))
    assert stats['train_old']['rows'] == 3
    assert stats['train_old']['tracklets'] == 2
    assert stats['source_rows_retained_after_gpt']['rows'] == 2
    assert stats['source_rows_retained_after_gpt']['tracklets'] == 1
    changed = variants.source_statistics(path, {'kept', 'removed'}, cache, variants.sha256(path))
    assert changed['source_rows_retained_after_gpt']['tracklets'] == 2
