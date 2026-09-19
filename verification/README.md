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

For Ref-YT-VOS, review is video-level: three deterministic, tracklet-bearing suggestions appear first by default, and the top selector can show 1, 3, 5, 10, or all. For ReVOS only, the fixed preview contains all nonexistent-object descriptions plus one deterministic explicit and one deterministic implicit description. Every ReVOS preview instruction starts accepted and has separate one-click instruction controls. The top **1 | Accept video** and **2 | Reject video** buttons—and the `1`/`2` keyboard keys when quick keys are enabled—decide the entire video and advance to the next video. Every unsampled ReVOS instruction remains visible below; **Add to preview + accept** adds it as an override, and **Remove override** removes it from export without deleting saved edits. An accepted video's accepted base and override preview rows enter verified export; rejected and undecided videos export no rows. The full instruction and all linked spans appear on every card. Click any colored span or link chip to isolate its corresponding tracklet mask(s) in the video; **Preview all masks** restores every tracklet for that instruction. Nonexistent-object descriptions intentionally have no masks.

Exact linked text spans and tracklet IDs share colors. Click a linked phrase to isolate its overlays, or click tracklet names to show/hide overlays. You may play/pause, enable or disable looping, or step through frames manually. Edits include changing language, linking one exact span to one or many tracklets, deleting spans or tracklets, discarding/restoring an individual instruction, reverting an instruction or whole video, and accepting/rejecting a video. Accepting a video now requires at least one confirmed, non-discarded instruction. Optional quick keys use `1` for accept and `2` for reject. Either decision advances to the immediately following video, including videos that already have decisions, so reviewing an existing decision never skips it. **Browse all videos** opens a numbered status grid; accepted, rejected, undecided, and current videos are visually distinct, and any box jumps directly to that video.

Every edit checkpoints immediately in browser local storage. The **Saving** menu shows the exact browser key, server-side decisions path, and Parquet output path. Optional periodic saving atomically writes `--decisions`; **Save + export both files** first saves current edits, then downloads both `decisions.json` and the updated Parquet. The same files are also retained at `--decisions` and `--output`. The source Parquet is never changed.

Non-ReVOS decisions use `selection_protocol: selected_instructions_v1`; each video stores `selected_sample_ids` and optional `allow_multiple`. Exactly one eligible accepted instruction is chosen in a stable order for verified export, with `sampled_sample_id` recording the choice. ReVOS videos instead store the video-level decision in `status` with `video_decision_explicit`, plus `review_mode: revos_preview_v1`, the combined `preview_sample_ids`, user-added `preview_override_sample_ids`, `accepted_sample_ids`, and `instructions[ID].status`. Only accepted videos export their accepted ReVOS preview instructions. Older files load unchanged: old edits and genuine video accept/reject decisions are retained, preview instructions without an explicit instruction-level decision initialize accepted, and statuses automatically derived by the first ReVOS adapter revision migrate to undecided. These adapter fields appear the next time decisions are saved; the input Parquet and original file are never rewritten merely by loading.

Cross-host issues encountered during the first local verification and their temporary workarounds are recorded in [verification reproducibility notes](../docs/VERIFICATION_REPRODUCIBILITY.md).
