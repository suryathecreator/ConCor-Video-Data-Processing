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

Review is video-level. By default, three deterministic, tracklet-bearing suggestions appear first for each video; the top selector can show 1, 3, 5, 10, or all. The full instruction and all linked spans appear on every suggestion and override card. Click any colored span or link chip to isolate its corresponding tracklet mask(s) in the video; **Preview all masks** restores every tracklet for that instruction. Then click **Confirm to annotate** to open its editor. The other instructions remain below as overrides. A per-video **Select multiple instructions / target masks for this video** switch lets you confirm several. An instruction can also have several tracklets, and one text span can link to several of them. Suggestions are stable across reloads and prefer tracklet-bearing instructions; they never remove access to the rest.

Exact linked text spans and tracklet IDs share colors. Click a linked phrase to isolate its overlays, or click tracklet names to show/hide overlays. You may play/pause, enable or disable looping, or step through frames manually. Edits include changing language, linking one exact span to one or many tracklets, deleting spans or tracklets, discarding/restoring an individual instruction, reverting an instruction or whole video, and accepting/rejecting a video. Accepting a video now requires at least one confirmed, non-discarded instruction. Optional quick keys use `1` for accept and `2` for reject. Either decision advances to the immediately following video, including videos that already have decisions, so reviewing an existing decision never skips it. **Browse all videos** opens a numbered status grid; accepted, rejected, undecided, and current videos are visually distinct, and any box jumps directly to that video.

Every edit checkpoints immediately in browser local storage. The **Saving** menu shows the exact browser key, server-side decisions path, and Parquet output path. Optional periodic saving atomically writes `--decisions`; **Save + export both files** first saves current edits, then downloads both `decisions.json` and the updated Parquet. The same files are also retained at `--decisions` and `--output`. The source Parquet is never changed.

New decisions use `selection_protocol: selected_instructions_v1`; each video stores `selected_sample_ids` and optional `allow_multiple`. From each accepted video, exactly one eligible accepted instruction is chosen in a stable order (preferring tracklet-bearing instructions) for the verified Parquet. `sampled_sample_id` records that choice. All accepted alternatives stay in `accepted_sample_ids` and any edits remain in `instructions`, so nothing is lost if you later want a different example. Undecided/rejected videos are omitted. Older video-level decisions are read without changing the source file: all non-discarded instructions count as previously accepted, one is sampled, and the remaining IDs are materialized in `accepted_sample_ids` at the next save. Older per-instruction edits are preserved.

Cross-host issues encountered during the first local verification and their temporary workarounds are recorded in [verification reproducibility notes](../docs/VERIFICATION_REPRODUCIBILITY.md).
