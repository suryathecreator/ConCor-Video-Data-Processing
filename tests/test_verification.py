from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from concor_video.verification import (
    REVOS_PREVIEW_PROTOCOL,
    SELECTION_PROTOCOL,
    VerificationState,
    _candidate_media_sources,
    _materialize_sampling,
    _revos_preview_order,
    _suggestion_order,
    apply_decisions,
    write_verified_parquet,
)


def _row():
    tracklets=[
        {"tracklet_id":"t1","masks":[None]},
        {"tracklet_id":"t2","masks":[None]},
    ]
    groups=[
        {"group_id":"g1","role":"main_referent","identity":"cats","text_spans":[{"start":0,"end":8,"text":"two cats"}],"tracklet_ids":["t1","t2"]}
    ]
    return {
        "sample_id":"s1","dataset":"revos","split":"val","video_id":"v1","text":"two cats",
        "tracklets_json":json.dumps(tracklets),"groups_json":json.dumps(groups),
        "span_links_json":"[]",
    }


def test_decisions_delete_tracklet_and_rebuild_bidirectional_links() -> None:
    decisions={"videos":{"revos::val::v1":{"status":"accepted","instructions":{"s1":{"deleted_tracklet_ids":["t2"]}}}}}
    output=apply_decisions([_row()],decisions)
    assert len(output)==1
    assert [x["tracklet_id"] for x in json.loads(output[0]["tracklets_json"])]==["t1"]
    assert json.loads(output[0]["span_links_json"])[0]["tracklet_ids"]==["t1"]
    assert output[0]["verification_status"]=="accepted"


def test_rejected_video_is_removed() -> None:
    decisions={"videos":{"revos::val::v1":{"status":"rejected","instructions":{}}}}
    assert apply_decisions([_row()],decisions)==[]


def test_discarded_instruction_is_removed_without_rejecting_video() -> None:
    decisions={"videos":{"revos::val::v1":{"status":"accepted","instructions":{"s1":{"discarded":True}}}}}
    assert apply_decisions([_row()], decisions)==[]


def test_media_root_rebases_processing_host_archive_path(tmp_path: Path) -> None:
    archive = tmp_path / "ref-youtube-vos" / "archives" / "valid.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"zip")
    row = {
        "frame_source": "/mmfs1/gscratch/datasets/concor-video/ref-youtube-vos/archives/valid.zip",
        "dataset_root": None,
    }
    assert archive in _candidate_media_sources([tmp_path], row)


def _selection_rows():
    first = _row()
    second = {**_row(), "sample_id": "s2", "text": "two dogs"}
    third = {**_row(), "sample_id": "s3", "video_id": "v2"}
    return [first, second, third]


def test_new_selection_protocol_exports_only_confirmed_accepted_instruction() -> None:
    decisions = {
        "selection_protocol": SELECTION_PROTOCOL,
        "videos": {
            "revos::val::v1": {
                "status": "accepted",
                "selected_sample_ids": ["s2"],
                "instructions": {},
            }
        },
    }
    assert [row["sample_id"] for row in apply_decisions(_selection_rows(), decisions)] == ["s2"]


def test_multi_selection_and_legacy_behavior() -> None:
    rows = _selection_rows()
    selected = {
        "selection_protocol": SELECTION_PROTOCOL,
        "videos": {
            "revos::val::v1": {
                "status": "accepted",
                "allow_multiple": True,
                "selected_sample_ids": ["s1", "s2"],
                "instructions": {},
            }
        },
    }
    output = apply_decisions(rows, selected)
    assert len(output) == 1
    assert output[0]["sample_id"] in {"s1", "s2"}
    assert apply_decisions(rows, {"videos": {}}) == []
    archived = _materialize_sampling(rows, selected)
    video = archived["videos"]["revos::val::v1"]
    assert set(video["accepted_sample_ids"]) == {"s1", "s2"}
    assert video["selected_sample_ids"] == ["s1", "s2"]
    assert video["sampled_sample_id"] == output[0]["sample_id"]


def test_legacy_acceptance_samples_one_and_archives_other_accepted_instructions() -> None:
    rows = _selection_rows()
    original = {
        "videos": {
            "revos::val::v1": {
                "status": "accepted",
                "instructions": {"s2": {"text": "edited two dogs"}},
            },
            "revos::val::v2": {"status": "rejected", "instructions": {}},
        }
    }
    prepared = _materialize_sampling(rows, original)
    video = prepared["videos"]["revos::val::v1"]
    assert set(video["accepted_sample_ids"]) == {"s1", "s2"}
    assert video["sampled_sample_id"] in {"s1", "s2"}
    assert video["instructions"]["s2"]["text"] == "edited two dogs"
    assert "accepted_sample_ids" not in original["videos"]["revos::val::v1"]
    assert [row["sample_id"] for row in apply_decisions(rows, original)] == [
        video["sampled_sample_id"]
    ]
    assert [row["sample_id"] for row in apply_decisions(rows, prepared)] == [
        video["sampled_sample_id"]
    ]


def test_explicit_override_keeps_legacy_accepted_alternatives_archived() -> None:
    rows = _selection_rows()
    original = {
        "videos": {
            "revos::val::v1": {
                "status": "accepted",
                "accepted_sample_ids": ["s1", "s2"],
                "selected_sample_ids": ["s2"],
                "instructions": {},
            }
        }
    }
    prepared = _materialize_sampling(rows, original)
    video = prepared["videos"]["revos::val::v1"]
    assert set(video["accepted_sample_ids"]) == {"s1", "s2"}
    assert video["sampled_sample_id"] == "s2"
    assert [row["sample_id"] for row in apply_decisions(rows, prepared)] == ["s2"]


def test_legacy_sampling_ignores_discarded_and_rejected_instructions() -> None:
    decisions = {
        "videos": {
            "revos::val::v1": {
                "status": "accepted",
                "instructions": {"s1": {"discarded": True}},
            },
            "revos::val::v2": {
                "status": "rejected",
                "sampled_sample_id": "s3",
                "instructions": {},
            },
        }
    }
    prepared = _materialize_sampling(_selection_rows(), decisions)
    assert prepared["videos"]["revos::val::v1"]["accepted_sample_ids"] == ["s2"]
    assert prepared["videos"]["revos::val::v1"]["sampled_sample_id"] == "s2"
    assert "sampled_sample_id" not in prepared["videos"]["revos::val::v2"]
    assert [row["sample_id"] for row in apply_decisions(_selection_rows(), prepared)] == ["s2"]


def test_old_decisions_file_loads_and_saves_all_accepted_alternatives(tmp_path: Path) -> None:
    rows = [
        {**row, "expression_id": row["sample_id"], "frame_ids_json": "[]"}
        for row in _selection_rows()
    ]
    parquet = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "schema_version": "concor-video-decisions-v1",
                "videos": {
                    "revos::val::v1": {
                        "status": "accepted",
                        "instructions": {"s2": {"text": "edited two dogs"}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    state = VerificationState([parquet], decisions_path, [], tmp_path / "verified.parquet")
    restored = state.decisions["videos"]["revos::val::v1"]
    assert set(restored["accepted_sample_ids"]) == {"s1", "s2"}
    assert restored["sampled_sample_id"] in {"s1", "s2"}
    state.save_decisions(state.decisions)
    persisted = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert set(persisted["videos"]["revos::val::v1"]["accepted_sample_ids"]) == {
        "s1", "s2"
    }
    assert persisted["videos"]["revos::val::v1"]["instructions"]["s2"]["text"] == "edited two dogs"
    assert len(apply_decisions(state.rows, persisted)) == 1
    write_verified_parquet(state.rows, persisted, tmp_path / "verified.parquet")
    exported = pq.read_table(tmp_path / "verified.parquet").to_pylist()
    assert len(exported) == 1
    assert exported[0]["sample_id"] == restored["sampled_sample_id"]


@pytest.mark.parametrize(
    "selection,allow_multiple",
    [([], False), (["s1", "s2"], False), (["s1", "s1"], True), (["s3"], False)],
)
def test_invalid_accepted_selections_are_rejected(selection, allow_multiple) -> None:
    decisions = {
        "selection_protocol": SELECTION_PROTOCOL,
        "videos": {
            "revos::val::v1": {
                "status": "accepted",
                "allow_multiple": allow_multiple,
                "selected_sample_ids": selection,
                "instructions": {},
            }
        },
    }
    with pytest.raises(ValueError):
        apply_decisions(_selection_rows(), decisions)


def test_discarded_selected_instruction_cannot_be_accepted() -> None:
    decisions = {
        "selection_protocol": SELECTION_PROTOCOL,
        "videos": {
            "revos::val::v1": {
                "status": "accepted",
                "selected_sample_ids": ["s1"],
                "instructions": {"s1": {"discarded": True}},
            }
        },
    }
    with pytest.raises(ValueError, match="discarded"):
        apply_decisions(_selection_rows(), decisions)


def test_new_protocol_requires_confirmation_even_if_selection_field_is_missing() -> None:
    decisions = {
        "selection_protocol": SELECTION_PROTOCOL,
        "videos": {"revos::val::v1": {"status": "accepted", "instructions": {}}},
    }
    with pytest.raises(ValueError, match="no confirmed instruction"):
        apply_decisions(_selection_rows(), decisions)


def test_suggestions_are_stable_and_put_masked_instructions_first() -> None:
    rows = [
        {"sample_id": "a", "tracklets_json": "[]", "negative": True},
        {"sample_id": "b", "tracklets_json": "[{\"tracklet_id\":\"t\"}]", "negative": False},
        {"sample_id": "c", "tracklets_json": "[{\"tracklet_id\":\"u\"}]", "negative": False},
        {"sample_id": "d", "tracklets_json": "[]", "negative": False},
    ]
    order = _suggestion_order(rows, "revos::val::v1")
    assert order == _suggestion_order(list(reversed(rows)), "revos::val::v1")
    assert set(order[:2]) == {"b", "c"}
    assert set(order) == {"a", "b", "c", "d"}


def _revos_rows() -> list[dict]:
    rows = []
    for sample_id, cohort in (
        ("n1", "nonexistent"),
        ("n2", "nonexistent"),
        ("e1", "explicit"),
        ("e2", "explicit"),
        ("i1", "implicit"),
        ("i2", "implicit"),
    ):
        negative = cohort == "nonexistent"
        rows.append(
            {
                **_row(),
                "sample_id": sample_id,
                "cohort": cohort,
                "negative": negative,
                "text": f"{cohort} {sample_id}",
                "tracklets_json": "[]" if negative else _row()["tracklets_json"],
                "groups_json": "[]" if negative else _row()["groups_json"],
            }
        )
    return rows


def test_revos_preview_is_all_nonexistent_plus_one_explicit_and_implicit() -> None:
    rows = _revos_rows()
    preview = _revos_preview_order(rows, "revos::val::v1")
    cohorts = {row["sample_id"]: row["cohort"] for row in rows}
    assert {sample_id for sample_id in preview if cohorts[sample_id] == "nonexistent"} == {
        "n1", "n2"
    }
    assert sum(cohorts[sample_id] == "explicit" for sample_id in preview) == 1
    assert sum(cohorts[sample_id] == "implicit" for sample_id in preview) == 1
    assert preview == _revos_preview_order(list(reversed(rows)), "revos::val::v1")


def test_revos_preview_defaults_to_accepted_and_exports_every_accepted_preview() -> None:
    rows = _revos_rows()
    prepared = _materialize_sampling(rows, {"videos": {}})
    video = prepared["videos"]["revos::val::v1"]
    assert video["review_mode"] == REVOS_PREVIEW_PROTOCOL
    assert len(video["preview_sample_ids"]) == 4
    assert video["accepted_sample_ids"] == video["preview_sample_ids"]
    assert video["status"] == "accepted"
    assert all(
        video["instructions"][sample_id]["status"] == "accepted"
        for sample_id in video["preview_sample_ids"]
    )
    output = apply_decisions(rows, {"videos": {}})
    assert {row["sample_id"] for row in output} == set(video["preview_sample_ids"])
    assert all(row["verification_status"] == "accepted" for row in output)


def test_revos_per_instruction_rejection_is_preserved_and_filtered() -> None:
    rows = _revos_rows()
    initial = _materialize_sampling(rows, {"videos": {}})
    video = initial["videos"]["revos::val::v1"]
    rejected = video["preview_sample_ids"][:2]
    for sample_id in rejected:
        video["instructions"][sample_id]["status"] = "rejected"
    prepared = _materialize_sampling(rows, initial)
    updated = prepared["videos"]["revos::val::v1"]
    assert not set(rejected) & set(updated["accepted_sample_ids"])
    assert {row["sample_id"] for row in apply_decisions(rows, prepared)} == set(
        updated["accepted_sample_ids"]
    )


def test_legacy_revos_rejection_and_edits_upgrade_without_loss() -> None:
    rows = _revos_rows()
    decisions = {
        "videos": {
            "revos::val::v1": {
                "status": "rejected",
                "instructions": {"e1": {"text": "manually edited"}},
            }
        }
    }
    prepared = _materialize_sampling(rows, decisions)
    video = prepared["videos"]["revos::val::v1"]
    assert video["status"] == "rejected"
    assert video["accepted_sample_ids"] == []
    assert all(
        video["instructions"][sample_id]["status"] == "rejected"
        for sample_id in video["preview_sample_ids"]
    )
    assert video["instructions"]["e1"]["text"] == "manually edited"
    assert apply_decisions(rows, decisions) == []


def test_mixed_dataset_keeps_standard_sampling_and_revos_adapter_separate() -> None:
    revos_rows = _revos_rows()
    ref_row = {
        **_row(),
        "sample_id": "ref-1",
        "dataset": "refytvos",
        "video_id": "ref-video",
        "cohort": "full_video",
        "negative": False,
    }
    decisions = {
        "selection_protocol": SELECTION_PROTOCOL,
        "videos": {
            "refytvos::val::ref-video": {
                "status": "accepted",
                "selected_sample_ids": ["ref-1"],
                "instructions": {},
            }
        },
    }
    output = apply_decisions([*revos_rows, ref_row], decisions)
    assert sum(row["dataset"] == "refytvos" for row in output) == 1
    assert sum(row["dataset"] == "revos" for row in output) == 4


def test_old_revos_decisions_file_is_adapted_only_when_saved(tmp_path: Path) -> None:
    rows = [
        {
            **row,
            "expression_id": row["sample_id"],
            "frame_ids_json": "[]",
        }
        for row in _revos_rows()
    ]
    parquet = tmp_path / "revos.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    decisions_path = tmp_path / "decisions.json"
    old = {
        "schema_version": "concor-video-decisions-v1",
        "videos": {
            "revos::val::v1": {
                "status": "accepted",
                "instructions": {"e1": {"text": "old edited expression"}},
            }
        },
    }
    decisions_path.write_text(json.dumps(old), encoding="utf-8")
    state = VerificationState([parquet], decisions_path, [], tmp_path / "verified.parquet")
    assert json.loads(decisions_path.read_text(encoding="utf-8")) == old
    adapted = state.decisions["videos"]["revos::val::v1"]
    assert adapted["review_mode"] == REVOS_PREVIEW_PROTOCOL
    assert adapted["instructions"]["e1"]["text"] == "old edited expression"
    state.save_decisions(state.decisions)
    saved = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert saved["videos"]["revos::val::v1"]["preview_sample_ids"]
    assert saved["videos"]["revos::val::v1"]["instructions"]["e1"]["text"] == "old edited expression"
