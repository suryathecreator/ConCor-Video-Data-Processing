# Offline video verifier

Start from a clone with a generated verification.parquet and locally available frames:

~~~bash
concor-video verify \
  --parquet data/verification.parquet \
  --media-root /data/concor-video \
  --decisions edits/decisions.json \
  --output exports/verified.parquet
~~~

The server binds to 127.0.0.1:8000 by default and uses no remote services. Add --parquet more than once to review several non-overlapping exports together.

Review is video-level: every instruction for the current video is visible at once. Click a card to inspect its colored temporal overlays. Select text inside its editor, check one or more tracklets, and add a link. You may remove spans or tracklets, revert one instruction, revert the whole video, and accept/reject the whole video. Turn quick keys on to use 1 for accept and 2 for reject.

Decisions autosave to browser local storage. Download decisions.json creates a portable checkpoint; --decisions resumes from it. Export updated Parquet applies the current decisions on the local server and downloads verified.parquet. The source file is never changed.
