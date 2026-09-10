# Output format

~~~text
campaign/
  worklist.json
  worklist-extraction-audit.md
  records/<sample_id>.json
  errors/<sample_id>.json
  claims/
  export/
    samples.parquet
    tracklets.parquet
    links.parquet
    verification.parquet
    run_ledger.csv
    manifest.json
~~~

- samples.parquet: one compact row per completed instruction.
- tracklets.parquet: one row per temporal instance with aligned uncompressed COCO RLE masks.
- links.parquet: one row per exact text span and its one-or-many tracklet IDs.
- verification.parquet: one self-contained row per instruction, including tracklets, groups, inverse links, extraction decisions, and SAM audit as JSON columns.
- run_ledger.csv: every selected instruction, including failed and pending work.
- manifest.json: row/status/disposition counts and table names.

Within a record, groups maps tracklet IDs to text spans; span_links is the deterministic inverse. Character intervals are half-open. Tracklet masks align exactly with frame_ids; absent frames use null.

The verifier groups rows by dataset, split, and video_id. Rejected videos are omitted from verified export. Deleted tracklets are removed, invalid/deleted links are removed, and edited exact spans rebuild span_links. Source Parquet is never overwritten.
