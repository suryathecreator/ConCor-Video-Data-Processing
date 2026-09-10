from __future__ import annotations

from concor_video.tracklet_schema import rebuild_span_links


def test_one_span_can_link_to_multiple_tracklets() -> None:
    groups = [
        {
            "tracklet_ids": ["t1"],
            "text_spans": [{"start": 0, "end": 9, "text": "the balls"}],
        },
        {
            "tracklet_ids": ["t2"],
            "text_spans": [{"start": 0, "end": 9, "text": "the balls"}],
        },
    ]
    assert rebuild_span_links(groups)[0]["tracklet_ids"] == ["t1", "t2"]
