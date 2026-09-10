# Pipeline

## Extraction

spaCy parses each official expression. The loader stores exact half-open Unicode character spans independently from normalized SAM prompts. Concrete visible people, objects, body parts, garments, surfaces, and stuff are eligible. Abstract concepts, taxonomy words, viewpoint-camera mentions, quoted titles, and unsafe generic thing/object prompts are audited but not segmented.

A single span may link to several tracklets. Overlapping spans are legal, such as a hand and its possessor.

## Tracking

Work is dynamically claimed by video. A worker materializes frames once, initializes one SAM3.1 Object Multiplex state, and processes every pending instruction for that video. Repeated normalized prompts share one result inside the video batch. Multiplex output discovers several matching instances in one prompt call.

Official target sequences are inserted first when public. Ref-YT-VOS public val/test masks are withheld, so SAM3.1 supplies the main tracklet and provenance stays model-generated. ReVOS nonexistent descriptions become zero-tracklet negatives.

Tracks are removed when too short, under 64 pixels over the sequence, or below confidence thresholds. Context tracks at volume IoU >= 0.65 with a main track are discarded; context-context tracks at IoU >= 0.80 merge. Volume IoU sums intersection and union over aligned frames. A missing or empty official target annotation is preserved as an audited `missing_main_referent` record instead of aborting a worker. A malformed zero-length language span is repaired from its extracted head and recorded in the extraction notes.

## Checkpointing and races

Thirty-two array tasks inspect the same sorted video queue from different rotations. Exclusive video leases prevent simultaneous sessions for one video. Nested sample leases and atomic rename commits prevent duplicate/corrupt records. Leases heartbeat and become recoverable after a dead worker. Persistent errors stop after a bounded number of attempts.

SIGUSR1 requests a drain: finish the active instruction, fsync its JSON, close the session, then requeue. Records, errors, and worklists live outside node-local storage. Reproducible decoded-frame scratch and mutable Torch/Triton/CUDA caches are job/task/restart-specific and removed on exit. Immutable checkpoint/runtime staging is node-shared under `flock`. Before staging, workers remove abandoned atomic partial copies under the same locks. If node scratch has less than 5 GiB free, a worker skips local model staging and uses an isolated campaign-side fallback cache, so a full `/tmp` cannot corrupt or indefinitely fail a shard.

Eager SAM execution is the default because measured max-autotune warmup took roughly 23 minutes for this workload. Optional compiled mode remains available for a separately benchmarked long-lived campaign.

## Output

Every selected instruction is represented in run_ledger.csv as completed, failed, or pending. Export validates completed records and writes atomic Parquet outputs. verification.parquet is deliberately denormalized so a small offline verifier can load it without table joins.
