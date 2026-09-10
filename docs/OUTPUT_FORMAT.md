# Output format

The canonical per-expression JSON record is preserved under `records/`. Export adds three Zstandard-compressed Parquet tables and a CSV ledger. JSON-valued columns use compact UTF-8 JSON strings so downstream tools do not need to reproduce a deeply nested Arrow schema.

## `samples.parquet`

One row per completed expression, including nonexistent-object negatives.

Important fields:

- identity: `sample_id`, `dataset`, `split`, `cohort`, `video_id`, `expression_id`;
- language: `text`, `span_links_json`, `extraction_json`;
- status: `negative`, `target_source`, `disposition`, `unresolved_context_count`;
- media: `frame_count`, `frame_ids_json`, `frame_files_json`;
- audit: `sam_prompt_audit_json`, `pipeline_json`, `runtime_seconds`.

## `tracklets.parquet`

One row per unique temporal instance. This is the simplest input for a future tracklet verification interface: group rows by `sample_id`, select one `tracklet_id`, parse `masks_rle_json`, and align each list element with `frame_ids_json`.

In addition to the sample identity and text, each row has:

- `tracklet_id`, `role`, `identity`, and `source`;
- `source_annotation_id` for official targets, or `sam_prompt` for predictions;
- `confidence`, `max_confidence`, and `present_frames`;
- `text_spans_json`, containing every exact linked span;
- `masks_rle_json`, containing one uncompressed COCO RLE object or `null` per frame.

No row is emitted for a nonexistent-object negative because it has no tracklet. Its expression remains in `samples.parquet`.

## `links.parquet`

One row per exact text span, with `tracklet_ids_json`. This is the canonical inverse relation and naturally represents one span linked to several tracks.

## `run_ledger.csv`

One row per selected worklist unit, including incomplete computation. `status` is `completed`, `failed`, or `pending`; completed rows also include disposition and tracklet count. This makes campaign accounting independent of model success.

## `manifest.json`

Contains schema version, campaign ID, row totals, status/disposition histograms, and relative table names. Export files are written through temporary files and renamed only after completion.
