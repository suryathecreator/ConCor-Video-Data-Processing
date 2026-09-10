from __future__ import annotations

import csv
import json

import numpy as np
import pyarrow.parquet as pq

from concor_video.checkpointing import atomic_json
from concor_video.exporter import export_campaign
from concor_video.rle import encode_rle
from concor_video.tracklet_schema import SCHEMA_VERSION, rebuild_span_links


def test_export_is_flat_and_accounts_for_pending(tmp_path) -> None:
    campaign = tmp_path / "campaign"
    worklist = {
        "campaign_id": "test",
        "units": [
            {
                "sample_id": "done",
                "dataset": "ref_youtube_vos",
                "split": "train",
                "cohort": "first_frame",
                "video_id": "v",
                "expression_id": "0",
                "target_source_expected": "official_dataset_ground_truth",
                "sam_prompt_groups": [],
            },
            {
                "sample_id": "pending",
                "dataset": "ref_youtube_vos",
                "split": "train",
                "cohort": "first_frame",
                "video_id": "w",
                "expression_id": "0",
                "target_source_expected": "official_dataset_ground_truth",
                "sam_prompt_groups": [],
            },
        ],
    }
    worklist_path = campaign / "worklist.json"
    atomic_json(worklist_path, worklist)
    groups = [
        {
            "group_id": "main",
            "role": "main_referent",
            "identity": "owl",
            "text_spans": [{"start": 0, "end": 6, "text": "an owl"}],
            "tracklet_ids": ["gt-001"],
        }
    ]
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": "done",
        "dataset": "ref_youtube_vos",
        "split": "train",
        "cohort": "first_frame",
        "video_id": "v",
        "expression_id": "0",
        "text": "an owl",
        "negative": False,
        "frame_ids": ["00000"],
        "frame_files": ["v/00000.jpg"],
        "tracklets": [
            {
                "tracklet_id": "gt-001",
                "source": "ground_truth",
                "source_annotation_id": "1",
                "sam_prompt": None,
                "confidence": 1.0,
                "present_frames": 1,
                "masks": [encode_rle(np.ones((4, 4), dtype=bool))],
            }
        ],
        "groups": groups,
        "span_links": rebuild_span_links(groups),
        "extraction": {},
        "sam_prompt_audit": [],
        "pipeline": {"target_source": "official_dataset_ground_truth"},
        "unresolved_required_prompts": [],
        "disposition": "complete_bcc",
    }
    atomic_json(campaign / "records/done.json", record)

    manifest = export_campaign(
        worklist_path=worklist_path,
        campaign_root=campaign,
        output_dir=campaign / "export",
    )
    assert manifest["sample_rows"] == 1
    assert manifest["tracklet_rows"] == 1
    assert pq.read_table(campaign / "export/tracklets.parquet").num_rows == 1
    with (campaign / "export/run_ledger.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["completed", "pending"]
