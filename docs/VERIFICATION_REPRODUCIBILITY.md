# Verification reproducibility notes

This note records two cross-host issues found while moving the evaluation verifier from Hyak to macOS on 2026-09-10. Neither workaround changed annotations, masks, or the source media.

## Archived source paths

**Observed.** `frame_source` retained an absolute Hyak path such as `.../ref-youtube-vos/archives/valid.zip`. Initially, `--media-root` searched for frames inside its arguments but did not rebase the archived source path, so existing local archives still returned `404`.

**Temporary wrapper used during verification.** We wrote a separate `verification-local.parquet` whose `frame_source` values pointed directly to the copied local `valid.zip` and `ReVOS.tar`. The original Parquet was left intact.

**Repository fix.** The verifier now tries suffixes of each recorded provenance path beneath every `--media-root`. A single local media root therefore resolves both archives naturally; no Parquet wrapper or rewrite is needed.

## ReVOS tar index portability

**Observed.** The copied `ReVOS.tar.sqlite` stored the Hyak tar's byte size and nanosecond modification time. The tar contents copied correctly, but the destination filesystem assigned a different timestamp, so the old index was reported as stale.

**Temporary workaround used during verification.** We preserved the copied sidecar, then rebuilt offsets locally without extracting the 46 GB tar:

```bash
REV="$HOME/concor-video-data/media/ReVOS/ReVOS.tar"
mv "$REV.sqlite" "$REV.sqlite.from-hyak"
concor-video index-revos-tar --source "$REV" --output "$REV.sqlite"
```

**Repository fix.** New indexes validate archive size plus deterministic SHA-256 samples from the beginning, middle, and end; modification time is informational only. They remain valid after an unchanged tar and its index are copied to another filesystem. The verifier atomically rebuilds a missing, legacy, or genuinely stale index on first use. For a large local archive, prebuilding it before opening the UI avoids that one-time pause:

```bash
concor-video index-revos-tar \
  --source "$HOME/concor-video-data/media/ReVOS/ReVOS.tar" \
  --output "$HOME/concor-video-data/media/ReVOS/ReVOS.tar.sqlite"
```
