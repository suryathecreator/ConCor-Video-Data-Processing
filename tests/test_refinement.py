"""CPU-only regression tests for the paid refinement boundary and BCC export."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refine_hf", ROOT / "scripts/refine_hf.py")
refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refine)


def row(sid="a"):
    tracks = [
        {"tracklet_id": "target", "masks": [{"size": [2, 2], "counts": [0, 1, 3]}, None]},
        {"tracklet_id": "context", "masks": [None, {"size": [2, 2], "counts": [1, 1, 2]}]},
    ]
    groups = [
        {"group_id": "main", "role": "main_referent", "identity": "dog", "tracklet_ids": ["target"],
         "text_spans": [{"start": 0, "end": 5, "text": "a dog"}]},
        {"group_id": "ctx", "role": "context_entity", "identity": "dog", "tracklet_ids": ["context"],
         "text_spans": [{"start": 11, "end": 16, "text": "a dog"}]},
    ]
    value = {key: "" for key in refine.META}
    value.update(sample_id=sid, dataset="fixture", split="val", video_id="video", text="a dog with a dog",
                 negative=False, disposition="complete_bcc", target_source="official",
                 frame_ids_json='["00000","00001"]', frame_files_json='[]', tracklets_json=refine.dumps(tracks),
                 groups_json=refine.dumps(groups), span_links_json=refine.dumps(refine.rebuild_span_links(groups)),
                 extraction_json='{"original":true}', runtime_seconds=1.25)
    return value


MAPPING = {"visual_to_local": {1: "target", 2: "context"}, "main_ids": [1]}


def test_tag_parser_preserves_unicode_possessives_plural_and_references():
    text, spans = refine.parse_tags("[2][1]Zoë’s[/1] bag[/2] is beside [1]her[/1].", {1, 2})
    assert text == "Zoë’s bag is beside her."
    assert [text[a:b] for a, b in spans[1]] == ["Zoë’s", "her"]
    text, spans = refine.parse_tags("[1][2]two dogs[/2][/1] play.", {1, 2})
    assert spans[1] == spans[2] == [(0, 8)]


@pytest.mark.parametrize("caption", ["[3]a dog[/3]", "[1]a [2]dog[/1][/2]", "[1][1]a dog[/1][/1]",
                                   "[1][/1]", "[1]a dog", "[1]a dog [/1]", "a[1]dog[/1]"])
def test_malformed_or_foreign_links_fail_closed(caption):
    with pytest.raises(ValueError):
        refine.parse_tags(caption, {1, 2})


def test_rewrite_rebuilds_both_directions_and_preserves_full_track():
    original = row()
    output = refine.apply_result(original, {"decision": "rewrite", "tagged_caption": "[1]A dog[/1] runs."}, MAPPING)
    assert output["text"] == "A dog runs."
    assert json.loads(output["tracklets_json"]) == [json.loads(original["tracklets_json"])[0]]
    assert json.loads(output["span_links_json"]) == [{"start": 0, "end": 5, "text": "A dog", "tracklet_ids": ["target"]}]
    assert output["extraction_json"] == original["extraction_json"]
    assert original["text"] == "a dog with a dog"
    with pytest.raises(ValueError, match="intended main"):
        refine.apply_result(original, {"decision": "rewrite", "tagged_caption": "[2]A dog[/2] runs."}, MAPPING)


def test_shared_collective_is_one_span_to_two_tracks():
    output = refine.apply_result(row(), {"decision": "rewrite", "tagged_caption": "[1][2]Two dogs[/2][/1] run."}, MAPPING)
    assert json.loads(output["span_links_json"])[0]["tracklet_ids"] == ["context", "target"]


def test_negative_is_preserved_without_links():
    original = row()
    original["negative"] = True
    output = refine.apply_result(original, {"decision": "keep", "tagged_caption": original["text"]}, {})
    assert output["text"] == original["text"]
    assert output["tracklets_json"] == output["groups_json"] == output["span_links_json"] == "[]"
    assert output["disposition"] == "negative_unsegmentable"


def test_exact_dedup_and_corrupt_mask_checks():
    original = row()
    tracks = json.loads(original["tracklets_json"])
    duplicate = deepcopy(tracks[0])
    duplicate["tracklet_id"] = "duplicate"
    duplicate["confidence"] = .01
    original["tracklets_json"] = refine.dumps([duplicate] + tracks)
    (kept, aliases), reason = refine.inspect_row(original, 64)
    assert reason is None and len(kept) == 2 and aliases == {"duplicate": "target"}
    tracks[0]["masks"][0]["counts"] = [1, 3, 1]
    original["tracklets_json"] = refine.dumps(tracks)
    assert refine.inspect_row(original, 64)[1] == "invalid_mask_rle"


def test_sampling_includes_endpoints_and_late_first_appearance():
    tracks = [{"masks": [None] * 96 + [{"size": [1, 1], "counts": [0, 1]}] + [None] * 3}]
    selected = refine.sample_frames(100, tracks)
    assert {0, 96, 99}.issubset(selected)
    assert len(selected) == 13


def test_prompt_metadata_is_not_sent():
    metadata, prompt = refine.read_prompt(ROOT / "prompts/video_refinement.md")
    assert "visual_inputs:" in metadata
    assert "visual_inputs:" not in prompt and "metadata_only:" not in prompt


def test_response_schema_binds_tracklet_permissions_to_each_sample():
    import re
    manifest = {"packet": {"expressions": [
        {"sample_id": "a", "allowed_ids": [1, 2], "negative": False},
        {"sample_id": "b", "allowed_ids": [3], "negative": False},
    ]}}
    variants = refine.response_schema(manifest)["properties"]["results"]["items"]["anyOf"]
    first, second = (variant["properties"] for variant in variants)
    assert first["sample_id"]["enum"] == ["a"] and second["sample_id"]["enum"] == ["b"]
    assert re.fullmatch(first["tagged_caption"]["pattern"], "[1]a dog[/1]")
    assert not re.fullmatch(first["tagged_caption"]["pattern"], "[3]a dog[/3]")
    assert not re.fullmatch(second["tagged_caption"]["pattern"], "[1]a dog[/1]")


def test_grouping_shares_only_identical_tracks_and_keeps_permissions(tmp_path, monkeypatch):
    first, second = row("a"), row("b")
    second["tracklets_json"] = refine.dumps([json.loads(second["tracklets_json"])[0]])
    second["groups_json"] = refine.dumps([json.loads(second["groups_json"])[0]])
    source = SimpleNamespace(rows={"a": first, "b": second}, row=lambda sid: {"a": first, "b": second}[sid])
    args = SimpleNamespace(media_root=[], max_tracklets=64, group_size=12, group_tracklets=24, sample_count=12)
    monkeypatch.setattr(refine.Media, "frame", lambda *args: b"fixture")
    monkeypatch.setattr(refine, "render_pair", lambda *args: (b"jpeg-fixture", [4, 2]))
    packets, early = refine.prepare(source, ["a", "b"], args, "prompt", tmp_path)
    assert not early and len(packets) == 1
    entries = packets[0]["packet"]["expressions"]
    assert entries[0]["allowed_ids"] == [1, 2]
    assert entries[1]["allowed_ids"] == [1]
    assert len(packets[0]["packet"]["tracklets"]) == 2
    assert packets[0]["mapping"]["b"]["visual_to_local"] == {1: "target"}


def test_budget_is_durable_and_long_context_costs_more(tmp_path):
    ledger = refine.Ledger(tmp_path / "ledger.sqlite", 1)
    assert ledger.reserve("request", .8)
    assert not ledger.reserve("request", .8)
    with pytest.raises(RuntimeError, match="budget"):
        ledger.reserve("other", .3)
    ledger.settle("request", .1)
    ledger.db.close()
    ledger = refine.Ledger(tmp_path / "ledger.sqlite", 1)
    assert ledger.total() == .1
    assert ledger.reserve("other", .3)
    ledger.db.close()
    short = refine.usage_cost({"input_tokens": 270_000, "output_tokens": 1000})
    long = refine.usage_cost({"input_tokens": 273_000, "output_tokens": 1000})
    assert long > 1.99 * short


def test_parquet_schema_and_untouched_columns_survive_export(tmp_path):
    original = row()
    schema = pa.Table.from_pylist([original]).schema.with_metadata({b"fixture": b"preserve"})
    pq.write_table(pa.Table.from_pylist([original], schema=schema), tmp_path / "source.parquet")
    source = refine.Source([tmp_path / "source.parquet"])
    output = tmp_path / "result"
    output.mkdir()
    refine.atomic_json(output / "decisions" / (refine.digest("a") + ".json"), {
        "sample_id": "a", "decision": "rewrite", "tagged_caption": "[1]A dog[/1] runs.", "mapping": MAPPING})
    ledger = refine.Ledger(output / "ledger.sqlite", 1)
    summary = refine.export(source, ["a"], output, {}, ledger)
    assert summary["complete"]
    table = pq.read_table(output / "val.parquet")
    assert table.schema.equals(schema, check_metadata=True)
    assert table.to_pylist()[0]["runtime_seconds"] == original["runtime_seconds"]
    ledger.db.close()


def test_grouped_response_must_cover_every_sample(tmp_path):
    source = SimpleNamespace(row=lambda sid: row(sid))
    manifest = {"request_id": "packet", "mapping": {"a": MAPPING, "b": MAPPING}}
    response = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text",
        "text": refine.dumps({"results": [{"sample_id": "a", "decision": "reject", "tagged_caption": ""}]})}]}]}
    refine.save_response(manifest, response, source, tmp_path)
    for sid in ("a", "b"):
        decision = json.loads((tmp_path / "decisions" / (refine.digest(sid) + ".json")).read_text())
        assert decision["decision"] == "error"


def test_protocol_retry_only_resends_invalid_sample_and_resumes_without_charge(tmp_path):
    source = SimpleNamespace(row=lambda sid: row(sid))
    expressions = [{"sample_id": sid, "allowed_ids": [1, 2], "negative": False, "existing_text": row(sid)["text"]}
                   for sid in ("a", "b")]
    manifest = {"request_id": "packet", "mapping": {"a": MAPPING, "b": MAPPING}, "packet": {
        "expressions": expressions, "frame_ids": [], "selected_frame_indices": []},
        "images": [], "dimensions": [], "max_output_tokens": 100}
    def response(results):
        return {"status": "completed", "usage": {"input_tokens": 50, "output_tokens": 30},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": refine.dumps({"results": results})}]}]}
    first = response([{"sample_id": "a", "decision": "rewrite", "tagged_caption": "[3]A dog[/3] runs."},
                      {"sample_id": "b", "decision": "rewrite", "tagged_caption": "[1]A dog[/1] runs."}])
    retry = response([{"sample_id": "a", "decision": "rewrite", "tagged_caption": "[1]A dog[/1] runs."}])
    calls = []
    def call(method, endpoint, **kwargs):
        calls.append(kwargs["json"])
        return SimpleNamespace(json=lambda: first if len(calls) == 1 else retry)
    refine.atomic_json(tmp_path / "requests.json", ["packet"])
    ledger = refine.Ledger(tmp_path / "ledger.sqlite", 1)
    client = SimpleNamespace(call=call)
    refine.run_regular([manifest], "prompt", source, tmp_path, ledger, client)
    assert len(calls) == 2
    resent = json.loads(calls[1]["input"][0]["content"][-1]["text"])["expressions"]
    assert [e["sample_id"] for e in resent] == ["a"]
    cost = ledger.total()
    refine.run_regular([manifest], "prompt", source, tmp_path, ledger, client)
    assert len(calls) == 2 and ledger.total() == cost
    for sid in ("a", "b"):
        assert json.loads((tmp_path / "decisions" / (refine.digest(sid) + ".json")).read_text())["decision"] == "rewrite"
    ledger.db.close()


def test_terminal_batch_without_results_is_error_not_semantic_rejection(tmp_path):
    manifest = {"request_id": "r", "mapping": {"a": MAPPING}}
    ledger = refine.Ledger(tmp_path / "ledger.sqlite", 1)
    ledger.reserve("r", .1)
    client = SimpleNamespace(call=lambda *args: SimpleNamespace(json=lambda: {"status": "failed"}))
    refine.collect_batches([{"id": "batch-fixture", "requests": ["r"]}], [manifest], None, tmp_path, ledger, client, False)
    result = json.loads((tmp_path / "decisions" / (refine.digest("a") + ".json")).read_text())
    assert result["decision"] == "error" and ledger.total() == .1
    ledger.db.close()


def test_facing_is_not_an_entity_even_if_model_emits_it():
    with pytest.raises(ValueError, match="verb-only"):
        refine.parse_tags("a penguin [1]facing[/1] it", {1})


def test_nlp_preserves_words_and_all_candidates_for_valid_noun(tmp_path):
    original = row()
    original["text"] = "a dog facing a dog"
    groups = json.loads(original["groups_json"])
    groups[1]["text_spans"] = [{"start": 13, "end": 18, "text": "a dog"}]
    duplicate = deepcopy(groups[1])
    duplicate.update(group_id="ctx2", tracklet_ids=["context2"])
    verb = {"group_id": "verb", "role": "context_entity", "identity": "face",
            "tracklet_ids": ["face"], "text_spans": [{"start": 6, "end": 12, "text": "facing"}]}
    original["groups_json"] = refine.dumps(groups + [duplicate, verb])
    tracks = json.loads(original["tracklets_json"])
    tracks += [{"tracklet_id": "context2", "masks": [{"size": [2, 2], "counts": [2, 1, 1]}, None]},
               {"tracklet_id": "face", "masks": [{"size": [2, 2], "counts": [3, 1]}, None]}]
    original["tracklets_json"] = refine.dumps(tracks)
    source = SimpleNamespace(row=lambda _: original)
    packets, early = refine.prepare_nlp(source, ["a"], SimpleNamespace(max_tracklets=64, group_size=12), "prompt", tmp_path)
    assert not early
    manifest = packets[0]
    entry = manifest["packet"]["expressions"][0]
    assert len(entry["existing_groups"]) == 3
    assert entry["existing_groups"][1]["candidate_count"] == 2
    result = {"decision": "rewrite", "tagged_caption": "[1]a dog[/1] facing [2]a dog[/2]"}
    clean = refine.apply_result(original, result, manifest["mapping"]["a"])
    assert clean["text"] == original["text"]
    assert json.loads(clean["tracklets_json"]) == tracks[:3]
    assert "face" not in {t for g in json.loads(clean["groups_json"]) for t in g["tracklet_ids"]}
    assert json.loads(clean["groups_json"])[1]["text_spans"] == [{"start": 13, "end": 18, "text": "a dog"}]
    with pytest.raises(ValueError, match="exact original"):
        refine.apply_result(original, {"decision": "rewrite", "tagged_caption": "[1]A dog[/1]"}, manifest["mapping"]["a"])
    with pytest.raises(ValueError, match="main referent"):
        refine.apply_result(original, {"decision": "rewrite", "tagged_caption": "a dog facing [2]a dog[/2]"}, manifest["mapping"]["a"])
    with pytest.raises(ValueError, match="cannot semantically reject"):
        refine.apply_result(original, {"decision": "reject", "tagged_caption": ""}, manifest["mapping"]["a"])
    assert not manifest["images"]
    assert refine.response_schema(manifest)["properties"]["results"]["items"]["anyOf"][0]["properties"]["decision"]["enum"] == ["keep", "rewrite"]


def test_nlp_negative_has_no_candidate_groups(tmp_path):
    original = row()
    original["negative"] = True
    original["text"] = "a dog [that is absent]"
    packets, _ = refine.prepare_nlp(SimpleNamespace(row=lambda _: original), ["a"],
                                   SimpleNamespace(max_tracklets=64, group_size=12), "prompt", tmp_path)
    clean = refine.apply_result(original, {"decision": "keep", "tagged_caption": original["text"]}, packets[0]["mapping"]["a"])
    assert clean["text"] == original["text"] and clean["groups_json"] == clean["tracklets_json"] == "[]"


def test_density_sample_has_exact_video_quotas_and_featured_within_200(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from refine_sample import select_videos
    values = {}
    for dataset in ["ref_youtube_vos", "revos"]:
        for index in range(120):
            original = row(f"{dataset}-{index}")
            original.update(dataset=dataset, video_id=f"{dataset}-{index}")
            count = [1, 4, 8, 20][index % 4]
            original["groups_json"] = refine.dumps([{"tracklet_ids": [str(i) for i in range(count)]}])
            values[original["sample_id"]] = original
            second = dict(original, sample_id=original["sample_id"] + "-second", groups_json="[]")
            values[second["sample_id"]] = second
    source = SimpleNamespace(rows=values)
    ids, records = select_videos(source, 100, 123, ["ref_youtube_vos-3"])
    assert len(ids) == len(records) == len(set(ids)) == 200
    assert records[0]["featured"] and records[0]["video_id"] == "ref_youtube_vos-3"
    from collections import Counter
    assert set(Counter((r["dataset"], r["density_bin"]) for r in records).values()) == {25}
    assert (ids, records) == select_videos(source, 100, 123, ["ref_youtube_vos-3"])
    with pytest.raises(ValueError, match="never a full-corpus"):
        select_videos(source, 101, 123, [])
    with pytest.raises(ValueError, match="missing"):
        select_videos(source, 100, 123, ["missing"])


def test_parallel_calls_checkpoint_successful_siblings_after_failure(tmp_path):
    packets = [{"request_id": sid, "mapping": {sid: MAPPING}, "max_output_tokens": 100,
                "packet": {"frame_ids": [], "selected_frame_indices": [], "expressions": [
                    {"sample_id": sid, "allowed_ids": [1, 2], "negative": False}]}, "dimensions": [], "images": []}
               for sid in ["a", "b"]]
    def call(*args, **kwargs):
        sid = json.loads(kwargs["json"]["input"][0]["content"][-1]["text"])["expressions"][0]["sample_id"]
        if sid == "a":
            import time
            deadline = time.monotonic() + 5
            checkpoint = tmp_path / "packets/b/response.json"
            while not checkpoint.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert checkpoint.exists(), "completed peer must checkpoint while the first request is still running"
            raise RuntimeError("network failed")
        return SimpleNamespace(json=lambda: {"status": "completed", "usage": {"input_tokens": 5, "output_tokens": 5},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": refine.dumps({"results": [
                {"sample_id": sid, "decision": "rewrite", "tagged_caption": "[1]A dog[/1] runs."}]})}]}]})
    ledger = refine.Ledger(tmp_path / "budget.sqlite", 1)
    with pytest.raises(RuntimeError, match="network"):
        refine.run_regular(packets, "prompt", SimpleNamespace(row=row), tmp_path, ledger, SimpleNamespace(call=call), workers=2)
    assert (tmp_path / "packets/b/response.json").exists()
    assert dict(ledger.db.execute("SELECT id,state FROM charges")) == {"a": "reserved", "b": "completed"}
    ledger.db.close()


def test_media_eligibility_does_not_treat_empty_root_as_video(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from refine_sample import available_rows
    rows = {sid: dict(row(sid), frame_source=str(tmp_path / archive), dataset_root=str(tmp_path), video_id=sid)
            for sid, archive in [("present", "train.zip"), ("missing", "valid.zip"), ("folder", "missing.zip")]}
    (tmp_path / "train.zip").write_bytes(b"fixture: eligibility checks source existence only")
    (tmp_path / "JPEGImages/folder").mkdir(parents=True)
    assert set(available_rows(SimpleNamespace(rows=rows), [tmp_path])) == {"present", "folder"}


def test_stable_visual_ids_survive_nlp_removal(tmp_path, monkeypatch):
    original = row()
    sig = refine.track_signature(json.loads(original['frame_ids_json']), json.loads(original['tracklets_json'])[0])
    source = SimpleNamespace(rows={'a': original}, row=lambda _: original, visual_registry=lambda _: {"dropped": 1, sig: 2})
    args = SimpleNamespace(media_root=[], max_tracklets=64, group_size=12, group_tracklets=24, sample_count=12)
    monkeypatch.setattr(refine.Media, 'frame', lambda *args: b'fixture')
    monkeypatch.setattr(refine, 'render_pair', lambda *args: (b'jpeg-fixture', [4, 2]))
    packets, _ = refine.prepare(source, ['a'], args, 'prompt', tmp_path)
    assert packets[0]['mapping']['a']['visual_to_local'] == {2: 'target', 3: 'context'}


def recovery_packet():
    return {"request_id": "interrupted", "mapping": {"a": MAPPING}, "max_output_tokens": 100,
            "packet": {"frame_ids": [], "selected_frame_indices": [], "expressions": [
                {"sample_id": "a", "allowed_ids": [1, 2], "negative": False}]}, "dimensions": [], "images": []}


def test_explicit_interruption_recovery_keeps_old_charge_and_resumes_once(tmp_path):
    packet = recovery_packet()
    refine.atomic_json(tmp_path / "requests.json", [packet["request_id"]])
    ledger = refine.Ledger(tmp_path / "budget.sqlite", 1)
    ledger.reserve("interrupted", .1)
    calls = []
    def call(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(json=lambda: {"status": "completed", "usage": {"input_tokens": 5, "output_tokens": 5},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": refine.dumps({"results": [
                {"sample_id": "a", "decision": "rewrite", "tagged_caption": "[1]A dog[/1] runs."}]})}]}]})
    client = SimpleNamespace(call=call)
    with pytest.raises(RuntimeError, match="unresolved"):
        refine.run_regular([packet], "prompt", SimpleNamespace(row=row), tmp_path, ledger, client)
    assert not calls
    refine.run_regular([packet], "prompt", SimpleNamespace(row=row), tmp_path, ledger, client, recover_interrupted=True)
    assert len(calls) == 1 and ledger.total() > .1
    assert ledger.db.execute("SELECT amount,state FROM charges WHERE id='interrupted'").fetchone() == (.1, "reserved")
    decision = json.loads((tmp_path / "decisions" / (refine.digest("a") + ".json")).read_text())
    assert decision["transport_retry_of"] == "interrupted" and decision["decision"] == "rewrite"
    total = ledger.total()
    refine.run_regular([packet], "prompt", SimpleNamespace(row=row), tmp_path, ledger, client, recover_interrupted=True)
    assert len(calls) == 1 and ledger.total() == total
    ledger.db.close()


@pytest.mark.parametrize("limit,already_interrupted,reason", [(0.1, False, "budget"), (1, True, "no further")])
def test_interruption_recovery_never_bypasses_budget_or_retries_forever(tmp_path, limit, already_interrupted, reason):
    packet = recovery_packet()
    refine.atomic_json(tmp_path / "requests.json", [packet["request_id"]])
    ledger = refine.Ledger(tmp_path / "budget.sqlite", limit)
    ledger.reserve("interrupted", .1)
    if already_interrupted:
        ledger.reserve(refine.digest(["interrupted", "transport-recovery-v1"])[:24], .1)
    def call(*args, **kwargs):
        pytest.fail("No HTTP call is allowed")
    with pytest.raises(RuntimeError, match=reason):
        refine.run_regular([packet], "prompt", SimpleNamespace(row=row), tmp_path, ledger, SimpleNamespace(call=call), recover_interrupted=True)
    assert ledger.total() == (.2 if already_interrupted else .1)
    ledger.db.close()
