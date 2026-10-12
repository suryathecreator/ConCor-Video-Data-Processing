"""Release handoff: only validated NLP rows may reach paid visual refinement."""
from collections import Counter
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("valid,unresolved", [
    (("keep", "rewrite"), True),
    ((), True),
    (("keep", "rewrite"), False),
])
def test_visual_handoff_preserves_unresolved_audit_and_publication_gate(
        tmp_path, monkeypatch, valid, unresolved):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import refine_release as release

    output, input_path = tmp_path / "run", tmp_path / "input.parquet"
    input_path.touch()
    nlp = output / "nlp"
    ids = [*valid, "rejected", *(["error", "pending"] if unresolved else [])]
    source = SimpleNamespace(rows={sid: {"split": "train"} for sid in ids},
                             locations={sid: index for index, sid in enumerate(ids)})
    for sid in valid:
        release.atomic_json(nlp / "decisions" / (release.digest(sid) + ".json"),
                            {"sample_id": sid, "decision": sid, "stage": "nlp"})
    error = {"sample_id": "error", "decision": "error", "stage": "nlp",
             "reason_codes": ["NLP cleanup dropped the intended main referent"],
             "validation_retry_of": "original"}
    if unresolved:
        release.atomic_json(nlp / "decisions" / (release.digest("error") + ".json"), error)
    rejected = {"sample_id": "rejected", "decision": "reject", "stage": "heuristic",
                "reason_codes": ["excessive_entity_link"]}
    release.atomic_json(nlp / "heuristics.json", {"rejected": rejected})
    release.atomic_json(nlp / "requests.json", ["original"])
    release.atomic_json(nlp / "packets/original/input.json",
                        {"request_id": "original", "mapping": {sid: {} for sid in ids}})
    # A later successful resume must also clear an old unresolved sidecar.
    release.atomic_json(output / "unresolved_nlp.json", {"error": error})

    monkeypatch.setattr(release, "Source", lambda *a, **kw: source)
    monkeypatch.setattr(release, "available_rows", lambda *a: source.rows)
    monkeypatch.setattr(release, "ledger_total", lambda *a: 0.)
    monkeypatch.setattr(release, "estimate_release",
                        lambda *a: {"buffered_total_usd": 1., "training_videos": 1})
    monkeypatch.setattr(release, "read_prompt", lambda path: ("metadata", path.stem))
    monkeypatch.setattr(release, "freeze_execution", lambda *a: None)
    monkeypatch.setattr(release, "load_env", lambda *a: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-no-network")
    monkeypatch.setattr(release, "ReleaseAPIClient",
                        lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(release, "CleanedSource", lambda *a: source)

    submitted, exported, published = [], [], []

    def prepare(cleaned, selected, args, prompt, folder):
        assert cleaned is source and selected == list(valid)
        packets = ([{"request_id": "visual", "mapping": {sid: {} for sid in selected}}]
                   if selected else [])
        return packets, {}

    def run_stage(packets, prompt, cleaned, folder, ledger, client):
        submitted.append((folder, [sid for packet in packets for sid in packet["mapping"]]))

    def export(cleaned, selected, folder, early, ledger):
        exported.append((list(selected), dict(early)))
        counts = Counter(early[sid]["decision"] if sid in early else "keep" for sid in selected)
        return {"selected": len(selected), "counts": dict(counts),
                "complete": not (counts["error"] or counts["pending"])}

    monkeypatch.setattr(release, "prepare", prepare)
    monkeypatch.setattr(release, "run_stage", run_stage)
    monkeypatch.setattr(release, "export", export)
    publisher = ModuleType("publish_refined_hf")
    publisher.publish = lambda *args: published.append(args) or {"commit": "fixture"}
    monkeypatch.setitem(sys.modules, "publish_refined_hf", publisher)
    monkeypatch.setattr(sys, "argv", [
        "refine_release.py", "--input", str(input_path), "--output", str(output),
        "--media-root", str(tmp_path), "--pilot-run", str(tmp_path / "pilot"),
        "--mode", "run", "--publish",
    ])

    if unresolved:
        with pytest.raises(RuntimeError, match="release has unresolved errors/pending rows"):
            release.main()
    else:
        release.main()

    assert submitted == [(nlp, sorted(ids)), (output, list(valid))]
    assert exported[0][0] == ids  # Export audits all inputs, including skipped NLP rows.
    assert exported[0][1]["rejected"] == rejected
    remaining = json.loads((output / "unresolved_nlp.json").read_text())
    if unresolved:
        assert remaining["error"] == exported[0][1]["error"] == error
        assert remaining["pending"] == {"sample_id": "pending", "decision": "pending"}
        assert set(remaining) == {"error", "pending"}
        assert not published
        assert not json.loads((output / "summary.json").read_text())["complete"]
    else:
        assert remaining == {}
        assert len(published) == 1
