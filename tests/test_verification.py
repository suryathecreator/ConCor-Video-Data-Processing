from __future__ import annotations

import json
from concor_video.verification import apply_decisions


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
