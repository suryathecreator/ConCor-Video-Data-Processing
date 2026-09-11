# Offline video verifier

Start from a clone with a generated `verification.parquet` and locally available media:

```bash
concor-video verify \
  --parquet data/verification.parquet \
  --media-root /data/concor-video \
  --decisions edits/decisions.json \
  --output exports/verified.parquet
```

The server binds to `127.0.0.1:8000` by default and uses no remote services. Add `--parquet` more than once to review several non-overlapping exports together. Processing-host paths stored in the Parquet are automatically rebased beneath each `--media-root`, so copied media does not require a rewritten Parquet. A legacy or stale ReVOS tar index is rebuilt automatically once using the local tar; new indexes use a portable sampled-content fingerprint rather than filesystem modification time.

Review is video-level: every instruction for the current video is visible at once while the video panel stays fixed on wider screens. Exact linked text spans and their tracklet IDs share colors. Click a linked phrase to isolate its overlays, or click tracklet names to show/hide overlays. You may play/pause, enable or disable looping, or step through frames manually. Edits include changing language, linking one exact span to one or many tracklets, deleting spans or tracklets, discarding/restoring an individual instruction, reverting an instruction or whole video, and accepting/rejecting a video. Optional quick keys use `1` for accept and `2` for reject. Either decision advances to the immediately following video, including videos that already have decisions, so reviewing an existing decision never skips it. **Browse all videos** opens a numbered status grid; accepted, rejected, undecided, and current videos are visually distinct, and any box jumps directly to that video.

Every edit checkpoints immediately in browser local storage. The **Saving** menu shows the exact browser key, server-side decisions path, and Parquet output path. Optional periodic saving atomically writes `--decisions`; **Save + export both files** first saves current edits, then downloads both `decisions.json` and the updated Parquet. The same files are also retained at `--decisions` and `--output`. The source Parquet is never changed.

Cross-host issues encountered during the first local verification and their temporary workarounds are recorded in [verification reproducibility notes](../docs/VERIFICATION_REPRODUCIBILITY.md).
