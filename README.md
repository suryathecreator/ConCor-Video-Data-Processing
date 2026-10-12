# ConCor Video Data Processing

ConCor Video converts referring-video annotations into text-span <-> instance-tracklet correspondences. It preserves an official target tracklet when one is public, extracts independently visible context entities with spaCy, and uses SAM 3.1 Object Multiplex to segment and track additional entities. A plural text span may link to several tracklets, and every relationship is represented in both directions.

This is a clean, checkpointable research pipeline for Ref-YouTube-VOS and ReVOS. It includes a completely local browser interface for video-level verification and Parquet editing.

## Ref-YouTube-VOS terminology

"First-frame expression" describes how the language was authored: an annotator saw only the first frame when writing the expression. It does **not** mean running a full-video expression on frame zero. The original benchmark reports 3,978 videos (3,471/202/305 train/val/test), about 15k full-video-authored expressions, about 13k first-frame-authored expressions, and about 131k frame masks.

Only the **full-video-authored language** is in the current public release. This repository therefore has no pseudo-first-frame mode and processes complete videos only. It never relabels frame-zero inference as the retired first-frame annotation cohort.

## Public corpus covered

Counts are checked before GPU work. The current public full-video release and ReVOS metadata produce:

| Priority | Dataset / split / type | Videos | Instructions |
|---|---|---:|---:|
| evaluation | Ref-YT-VOS val, full video | 202 | 834 |
| evaluation | Ref-YT-VOS test, full video | 305 | 1,262 |
| evaluation | ReVOS val, explicit | 416 shared videos | 3,130 |
| evaluation | ReVOS val, implicit | 416 shared videos | 2,475 |
| evaluation | ReVOS val, nonexistent | 416 shared videos | 217 |
| train | Ref-YT-VOS train, full video | 3,471 | 12,913 |
| train | ReVOS train, explicit | 626 shared videos | 16,941 |
| train | ReVOS train, implicit | 626 shared videos | 12,203 |
| train | ReVOS train, nonexistent | 626 shared videos | 108 |

Evaluation totals 923 dataset/split videos and 7,918 instructions. Train totals 4,097 dataset/split videos and 42,165 instructions. ReVOS category video counts overlap, so they must not be summed.

The public Ref-YT-VOS valid metadata is the original 507-video pool. The loader removes the public 305-video competition-test subset to recover the disjoint 202-video validation split.

## Setup

You need Python 3.11+, a compatible NVIDIA GPU, the SAM 3.1 code/checkpoint, and dataset access under upstream terms.

~~~bash
git clone https://github.com/suryathecreator/ConCor-Video-Data-Processing.git
cd ConCor-Video-Data-Processing
bash scripts/setup.sh
source .venv/bin/activate
hf auth login
bash scripts/download_model.sh
~~~

To stage the public Ref-YT-VOS and ReVOS mirrors resumably:

~~~bash
bash scripts/prepare_public_datasets.sh /data/concor-video
~~~

The script keeps Ref-YT-VOS in validated ZIP archives. It also keeps the uncompressed ReVOS tar intact, builds a tiny byte-offset index for direct random frame reads, extracts only ReVOS metadata, and builds a shared read-only SQLite index for the large mask dictionary. This avoids a slow small-file expansion on parallel filesystems. Large archives use parallel range downloads when aria2 is available, and resume after interruption.

## Process one split

Ref-YT-VOS always processes full videos:

~~~bash
DATASET=refytvos DATA_ROOT=/data/concor-video/ref-youtube-vos SPLIT=val \
CAMPAIGN_ROOT=$PWD/outputs/ref-val bash scripts/run_local.sh
~~~

Choose any ReVOS reasoning categories:

~~~bash
DATASET=revos DATA_ROOT=/data/concor-video/ReVOS SPLIT=val \
REVOS_CATEGORIES="implicit explicit nonexistent" \
CAMPAIGN_ROOT=$PWD/outputs/revos-val bash scripts/run_local.sh
~~~

Use LIMIT for Ref-YT-VOS or LIMIT_PER_CATEGORY for ReVOS. SPLIT supports train/val/test for Ref-YT-VOS and train/val for ReVOS.

## Submit the complete public corpus

This is the exact reproducible command shape used for the full campaign:

~~~bash
DATA_ROOT=/data/concor-video \
CAMPAIGN_BASE=/outputs/concor-video-full \
PYTHON_BIN=$PWD/.venv/bin/python \
SAM31_REPO_ROOT=$PWD/external/sam3 \
SAM31_CHECKPOINT=$PWD/checkpoints/sam3.1/sam3.1_multiplex.pt \
SLURM_PARTITION=ckpt SLURM_ACCOUNT=your-account GPU_GRES=gpu:a40:1 \
NUM_WORKERS=32 bash scripts/submit_public_corpus.sh
~~~

The launcher submits a 32-task, one-A40 evaluation array with no percent concurrency cap; Slurm decides how many run. The train array has lower priority and starts after evaluation. Both are preemptible and resume from atomic per-instruction records.

For one custom split, use scripts/submit_slurm.sh. Change SLURM_PARTITION, SLURM_ACCOUNT, and GPU_GRES for another cluster/GPU.

## Pipeline

1. Select official expressions and assert release counts.
2. Parse exact referring/context spans with spaCy; keep the display span separate from the short SAM prompt.
3. Anchor the main referent to public ground truth, or label it explicitly as a SAM3.1 prediction when challenge masks are withheld.
4. Open one SAM3.1 session per video; reuse decoded frames, video state, and repeated prompt results across all its instructions.
5. Filter tiny/short/low-confidence tracks and merge duplicate temporal tracks by volume IoU.
6. Validate both tracklet-to-span groups and span-to-tracklet links.
7. Atomically checkpoint each instruction and export flat tables, a complete ledger, and one verifier-ready Parquet.
8. **Push the generated data to Hugging Face first.** Then optionally run the API refinement below on a downloaded HF snapshot plus local media. This stage is **optional for generation, required for our higher-quality release**. Inspect its pilot before publishing refined split Parquets.

Nonexistent ReVOS descriptions preserve the intentionally absent target; valid segmented context may retain its own links during refinement.

## Optional API refinement after HF upload

[`scripts/refine_hf.py`](scripts/refine_hf.py) reads the [HF annotations](https://huggingface.co/datasets/suryadv/concor-video-data-filtered) and your existing local frames/ZIPs/indexed tar. It uses **`gpt-6-luna`, medium reasoning** for every expression. No SAM rerun or GPU is needed. Downloading this HF repository provides annotations; stage video media separately with the dataset preparation script above.

The [separate prompt](prompts/video_refinement.md) combines the initial-write, audit and rewrite correspondence rules from [ConCor Training Data Generation](https://github.com/suryathecreator/ConCor-Training-Data-Generation) with temporal, coreference and phrase-boundary rules adapted from [Weiyang Huang's VIPOSeg](https://github.com/Weiyang-Huang/VIPOSeg-Grounded-Caption). Its front matter documents the accompanying images, metadata and output contract; **only the body is sent as the prompt**.

Visual presentation follows VIPOSeg: original frame on the left, colored masks and stable numeric labels on the right, up to 540 pixels high, JPEG quality 88. Sampling uses **12 evenly spaced frame indices including both endpoints, plus every tracklet's first visible frame**. It is not 1/2 FPS and does not send every frame. There are no extra crops, inverse crops or visual passes. Full mask sequences remain unchanged in retained output tracklets; sparse visual review cannot certify every intermediate event.

Up to **12 expressions from one dataset/split/video** share their images and tracklet inventory. Only exactly equal complete frame-aligned mask sequences share a visual ID. Each expression keeps its own allowed IDs and intended target, enforced in the response schema and local validation. Groups split above 24 distinct candidate tracks instead of making objects smaller. One severely noisy expression with more than 64 nonempty, distinct tracks is rejected by the configurable cheap filter. Invalid mask sequences, missing main tracks and empty main tracks are also rejected; empty context tracks and exact duplicates are removed. Similar-looking or high-IoU tracks are never merged by this wrapper.

One visual reasoning call chooses relevant context, repairs missing correspondences, rewrites the caption when needed, or rejects an irreparable sample. Balanced inline tags become exact Unicode character spans locally, then both BCC directions are rebuilt and checked. Every retained positive keeps its intended main track(s). Intentional nonexistent targets are **accepted by default**, preserving the exact caption and no target links. Visual resemblance does not override the negative label. Reject a negative only when its caption includes concrete context/background entities without valid supplied masks; segmented context may retain links. Regular mode and the two-pass release wrapper allow one validation retry for affected expressions only. Persistent validation/API/media failures remain errors, distinct from semantic rejection.

Release rules: **reject an entire example if one entity mention links to more than 10 distinct nonempty tracks**, including unions across fragmented groups. Exactly 10 is allowed; the limit is per entity, not per scene. Do not rescue an overfull flower/object group by selecting a smaller subset. Taxonomy and explanatory glosses are not extra referential mentions: in the flightless-bird question, link “Which animals,” leaving “Sphenisciformes,” “Spheniscidae,” and “(penguins)” unhighlighted. Positive captions omit concrete unsegmented background entities: a fox/dog caption cannot add “through the snow” without a supplied snow mask.

Manual observation from earlier dataset inspection: **Ref-YouTube-VOS more often has problematic context links; ReVOS is generally better but includes severe anomalies.** These observations motivate review; they are not a measured quality ranking. Model refinement and a 20-expression pilot also do not prove the entire release correct.

### Install, download and pilot

From this checkout on Linux/macOS with Python 3.11+:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[refine]'
hf download suryadv/concor-video-data-filtered --repo-type dataset --local-dir data/hf

python scripts/refine_hf.py \
  --input data/hf --media-root /data/concor-video \
  --output outputs/luna-pilot --pilot-per-dataset 10 \
  --mode regular --budget-usd 5
~~~

The HF root selects `train/verification.parquet` and `eval/unfiltered/verification.parquet`: **50,083 expressions across 5,020 dataset/split videos**. It deliberately excludes the overlapping manually verified subset. A single Parquet or split directory also works. Repeat `--media-root` for multiple locations. The deterministic pilot includes related expressions to exercise shared inputs and a nonexistent example when available.

The script asks for the API key with hidden terminal input. It keeps the key in process memory, clears its session credential on exit and never writes it to checkpoints, command lines or logs. For automation it can consume `OPENAI_API_KEY` from the environment; remove any parent-shell copy yourself. Deleting a local copy does not revoke a key or erase a previously sent chat message. Resume with the same command and output directory; completed API responses are reused. Do not change the input or prompt within an existing run.

### Bounded two-pass density review

[`scripts/refine_sample.py`](scripts/refine_sample.py) runs the new language pass
before the standard visual pass. Its default is **100 Ref-YT-VOS videos + 100 ReVOS
videos, one expression per video: 200 examples total**. It refuses more than 100
videos per dataset. `--include-video 003234408d` counts within that limit and pins
the penguin example separately in the viewer.

1. Select 25 videos from each candidate-count range (0–2, 3–5, 6–12, 13+) per
   dataset, with a deterministic seed. The densest annotated expression represents
   each video. Videos without a local source archive/frame folder are excluded
   before sampling and counted in `selection.json`. This is a density stress sample, not a verified count of
   real objects or a random estimate of corpus error rates.
2. Apply structural checks, remove empty context, deduplicate exactly equal mask
   sequences, and reject entity links with more than 10 distinct tracks.
   Structural failures remain visible as rejections with original audit images.
3. Run a **text-only GPT-6 Luna / medium** pass using
   [`prompts/nlp_cleanup.md`](prompts/nlp_cleanup.md). Preserve every original
   character and the intended target; repair noun boundaries and remove bogus
   entity links such as face masks attached to the verb “facing.” Candidate masks
   for a genuine noun remain together for later visual selection. This pass cannot
   reject a sentence or infer visual correctness. Local validation also forbids a
   standalone “facing” entity span in both passes.
4. Render the cleaned candidates and run the standard **GPT-6 Luna / medium**
   visual refinement, with the same VIPOSeg presentation and sampling policy.
5. Validate exact spans, target preservation and both BCC directions, then export
   the original 22-column Arrow schema and full retained mask sequences. Each
   stage allows one validation retry; errors remain distinct from rejection.

~~~bash
python scripts/refine_sample.py --input data/hf --media-root /data/concor-video \
  --output outputs/luna-density-200 --include-video 003234408d \
  --mode prepare --budget-usd 20 --prior-spend-usd 0.02981

# Load OPENAI_API_KEY privately into the environment; .env is not auto-loaded.
python scripts/refine_sample.py --input data/hf --media-root /data/concor-video \
  --output outputs/luna-density-200 --include-video 003234408d \
  --mode regular --workers 4 --budget-usd 20 --prior-spend-usd 0.02981

python scripts/build_refinement_viewer.py --input data/hf \
  --run outputs/luna-density-200 --output outputs/pilot-viewer/dist --webp-quality 76
~~~

Preparation sends **no API requests**. It saves the exact selection and original
input snapshot, prepares evidence, and bounds both stages including one retry.
The two stages share one durable budget ledger. Set `--prior-spend-usd` to all
other charges/reservations; the wrapper enforces a combined $100 ceiling. Resume
with the same inputs, prompts, arguments and output directory. Four HTTP calls may
run concurrently; checkpoint validation and ledger writes remain serialized.
Responses checkpoint immediately, and temporary Arrow/RLE buffers are released
between request waves to keep memory bounded. After a confirmed process interruption,
`--recover-interrupted` permits at most one replacement request under a new ID for
each unresolved reservation. Its original maximum charge remains in the ledger;
replacement calls also reserve budget, and unresolved replacements are never
automatically repeated. Without this explicit option, uncertain calls still stop
the run. `summary.json` reports their conservative cost separately.
`--expressions-per-video 0` explicitly expands to every expression of the selected
videos; the default stays at 200 expressions. The viewer shows original → NLP →
visual links, density filters, removed groups, exact visual inputs, and outcomes
of the early filters. The original-candidate view is for human audit only and is
not sent in an additional visual model call. No HF upload or full run occurs.

The viewer uses original JPEGs by default. Optional `--webp-quality 76` makes compact
display copies at the same dimensions, keeping every sampled frame and overlay.
This lets the 200-example viewer fit Sites' 256 MiB upload limit; it labels the
display encoding explicitly. The model still receives the original quality-88
JPEGs, which remain saved in the run checkpoints. No model calls are repeated.

Completed **2026-10-10**, seed `20261010`, in
`outputs/luna-density-200-20261010-available`: exactly **200 videos / expressions**,
100 per dataset and 25 per candidate-density range in each dataset. The 507 videos
whose local source archive was unavailable were excluded before sampling.
NLP cleanup kept 151 expressions and repaired 49, removing 57 candidate tracks;
the structural checks removed no additional tracks in this sample. Visual results:
**163 rewrites, 25 keeps, 11 rejections, 1 validation error**. The invalid result
dropped intended target IDs even after its one retry and remains excluded from
the export, separately from semantic rejections. The 188 retained rows export as
155 train and 33 val rows, with exact source schemas and unchanged retained mask
sequences verified. The featured `003234408d` goes
from **35 → 27 → 2 tracks**: NLP removes eight face candidates incorrectly linked
to “facing,” then visual refinement retains the target and neighboring penguin.

The [hosted review viewer](https://concor-video-refinement-pilot.daisydv.chatgpt.site)
is public and shows all 200 outcomes from the latest completed review. Click any
caption to show only its linked masks; click a colored span to isolate its BCC
link, or an ID chip to isolate one mask. Caption selection switches to the
matching before/after-NLP evidence. Click elsewhere or press Escape to restore
all masks for that view. Overlapping masks remain independently selectable.
Conservative run cost is
**$0.651928**, including **$0.053501** reserved for three calls interrupted by a
local memory-limit kill; their replacements were charged separately. Including
the earlier $0.02981 pilot, the conservative total is **$0.681738**. Checkpoints
and reservations are preserved. These are review results, not a measured accuracy
score or a full-corpus release; that review did not change HF.

Revised **2026-10-11**, using the same 200 expressions in
`outputs/luna-policy-v2-review-20261011`: **131 rewrites, 24 keeps, 45 rejections,
zero unresolved errors**. The stricter structural rules reject 37 before paid
review; NLP repairs 50 of the remaining 163 and removes 46 candidate tracks.
“(penguins)” remains unhighlighted in the taxonomy question, the fox/dog caption
omits unsupported snow, and the nonexistent car is kept unchanged without links.
The featured `003234408d` now fails the 10-track rule because “another penguin”
has 26 distinct tracks; its original masks remain inspectable. This review costs
**$0.466745**, bringing all review charges and outstanding earlier reservations to
**$1.148482**. The density-weighted training Batch estimate is **$53.32**, or
**$87.75 total including prior spending and the uncertainty/safety margin**;
the release ledger caps the aggregate at **$99**. These are cost projections,
not a promised final price or a measured semantic accuracy score.

### Full training release with automatic HF publication

[`scripts/refine_release.py`](scripts/refine_release.py) processes **only the training Parquet**, through both NLP and visual passes. The separate sample wrapper remains capped at 200 videos. A completed review using the same prompts is required before any full-run API submission. The estimate weights measured review costs by dataset and candidate-density range, applies the Batch discount, adds two standard errors and a 50% margin, and takes no credit for sharing full-run visual requests. It is a reasonable estimate, not a quote; the durable ledger independently caps reservations and charges below $100, including all supplied prior ledgers.

Only validated NLP results with **keep/rewrite** decisions proceed to visual refinement. Remaining NLP errors or pending results do not block valid expressions: they stay in unresolved_nlp.json and the final audit with their original status. Structural rejections stay excluded. The controller still requires all input outcomes to be resolved before automatic HF publication; forwarding valid rows does not relabel failures as semantic rejections.

~~~bash
python scripts/refine_release.py \
  --input data/hf/train/verification.parquet --media-root /data/concor-video \
  --output outputs/luna-training-v2 --pilot-run outputs/luna-policy-v2-review-20261011 \
  --prior-ledger outputs/luna-pilot-20261007/budget.sqlite \
  --prior-ledger outputs/luna-density-200-20261010-available/budget.sqlite \
  --total-budget-usd 99 --mode run --publish

# Recommended: start with --mode run --publish directly. It performs preparation
# and the cost check before submitting, then monitors both passes automatically.
# Use --mode prepare only if you want to stop after the free preflight.
# Credentials are read from .env (OPENAI_API_KEY / HF_TOKEN), HF's token cache,
# or --hf-token-file /path/to/a/local/credential. Never put tokens in arguments.
~~~

`--mode run` submits all work that the API queue and budget can admit, collects completed batches without waiting for earlier batches, and repeats for the dependent visual pass. **Use this faster default directly for new runs.** Run it in a durable terminal or scheduler job using `slurm/refine_release.slurm`. Optional `--mode submit` submits currently admissible NLP work and exits; `--mode run` resumes monitoring. `submit-first` remains a legacy alias for `submit`, and no longer restricts submission to one batch. Resume with the same arguments and output directory; requests, responses, batch IDs, cost reservations, and status checkpoints are preserved. On restart, collection also loads saved manifests for previously submitted validation retries, so their completed responses are recovered without resubmission. `status.json`, `estimate.json`, `summary.json`, and `publication.json` record progress and the verified HF commit. Uncertain charges remain reserved. A stopped run never mislabels pending examples as rejections and never publishes partial training output. Review/export errors require recovery before publication.

With `--publish`, [`scripts/publish_refined_hf.py`](scripts/publish_refined_hf.py) adds **`gpt_refined` → `train`** to `suryadv/concor-video-data-filtered` in one atomic commit. It preserves existing annotations and card content, documents both processing passes and limitations, and uploads prompts and per-expression decisions. All 22 source fields are retained under readable names (including `caption`, `is_nonexistent`, `source_dataset`, `object_tracks_json`, `text_entities_json`, and `caption_object_links_json`). Native `frame_ids` and `caption_links`, object/phrase counts, and `refinement_decision` make browsing easier. Full mask JSON is retained exactly. The processing manifest gives the complete column mapping. A write-enabled HF credential is required; credentials are never uploaded.

The original $5 pilot / $95 full-run split was a planning allowance. Include **all** prior actual charges and outstanding reservations; the current review ledger is included automatically. A reservation is conservative, not an OpenAI account spending limit: keep prices current and do not bypass the ledger by copying runs. After a submission/network interruption, collect known batches before investigating unresolved reservations; the script refuses to blindly repeat potentially billed calls.

Pricing checked 2026-10-11: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna) costs $0.10 input / $0.50 output per million standard tokens ($0.125 cache writes; $0.01 cache hits); [Batch](https://developers.openai.com/api/docs/guides/batch) halves those rates and has a 24-hour completion window. The ledger includes reasoning output, reserves the output ceiling, assumes no cache hits, accounts for cache writes and long-context premiums, and settles from returned usage. Image billing estimates deliberately overestimate patch tokens. Input rendering time and your account's Batch queue limit also affect ETA; multiple queue waves can take longer than 24 hours. Shared evidence reduces cost without lowering the requested model or visual resolution.

Batch packing uses the [documented API limits](https://developers.openai.com/api/docs/guides/batch): **50,000 requests and 200 MB per file**, measured after JSON/base64 serialization. There is no additional 1,000-request cap, 180 MB margin, guessed 16-million-token queue cap, or one-active-batch restriction. The API's actual account/model queue allowance controls admission. Project keys may not have access to admin rate-limit endpoints (the current key receives HTTP 403), so the scheduler uses real admission/rejection responses rather than assuming an account tier. It records any token limit reported by the service in `queue-capacity.json`. Only an explicit queue rejection before inference triggers smaller submissions; available capacity is retried automatically and completed batches free capacity. The 24-hour window starts separately for each submitted batch; the visual pass depends on completed NLP cleanup.

The **aggregate $99 ledger cap remains independent of batch size and concurrency**. Completed charges plus all outstanding maximum-cost reservations must fit before submission. If reservations temporarily occupy the budget, the scheduler collects results and uses the released capacity. Packing the same model requests into larger files does not add paid requests or change their prompts, grouping, images, or model. Submission journals are saved before paid POSTs, carry unique metadata tickets, and recover accepted batches after a lost response. A submission whose acceptance remains uncertain keeps its reservation and is never blindly repeated. Explicit pre-inference queue rejections are recorded separately from annotation errors and semantic rejections.

**Historical split and migration (2026-10-11):** the initial training controller used the overly conservative 1,000-request / 180 MB / 16M-token settings and waited for each batch. It planned NLP batches of **881, 892, 899, and 755 requests**. The first two were already submitted when scheduling was upgraded; their original IDs, results, and reservations are retained. The **1,654 unsubmitted requests** can now share one file and run alongside the existing batch, subject to actual API admission and the budget. The controller upgrade archives its old execution record and ledger before restarting; it does not cancel or duplicate remote work. **Do not reproduce that four-way split for future runs; use the current `--mode run` default.** The lower-level `refine_hf.py` also uses the faster scheduler for standalone visual-only `batch` / `--wait` workflows; use `refine_release.py` for the required two-pass training release.

Outputs are **`train.parquet`, `val.parquet`, `test.parquet`** as applicable, with the **same column order, Arrow types and schema metadata as the input**. Dataset identity remains in each row. Kept/rewritten rows retain `sample_id`, frame lists, full retained mask sequences and original extraction/SAM provenance; only `text`, `tracklets_json`, `groups_json`, `span_links_json` and `disposition` can change. Rejected rows are omitted. Empty splits still have the original schema. Check `summary.json` for `complete: true` before publishing: error/pending rows are omitted from partial exports, with their status retained in `audit.json`. Per-request manifests and raw model responses stay local. The release publisher adds readable columns and separate public processing audits only after the complete training run passes validation.

### Inspect the pilot in a browser

~~~bash
python scripts/build_refinement_viewer.py --input data/hf \
  --run outputs/luna-pilot --output outputs/pilot-viewer
python -m http.server 8000 --directory outputs/pilot-viewer
~~~

The static viewer includes exact shared input images, original/refined spans, retained IDs, rejection reasons and split Parquet downloads. It can be hosted without an API key. Dataset media keeps its upstream terms; do not commit generated pilot assets to the source branch.

To build interactive masks for an existing run, supply local media:

~~~bash
python scripts/build_refinement_viewer.py --input data/hf \
  --run outputs/luna-density-200-20261010-available \
  --output outputs/pilot-viewer/dist --media-root /data/concor-video \
  --webp-quality 76
~~~

This reads only the run's selected examples and sampled frame indices. It makes
no API calls and preserves all model results. The interactive display combines
compact original frames with separate lossless mask layers at up to 540px
height; selecting an absent mask displays no overlay. The original quality-88
model-input JPEGs remain in the run checkpoints. Per-example mask assets resume
on rebuild, and the static site contains no credentials.

Pilot completed **2026-10-07** with seed `20261007`: 10 Ref-YouTube-VOS and 10 ReVOS expressions, 10 shared initial requests and one validation retry. Results: **11 rewrites, 4 keeps, 5 rejections, 0 unresolved errors**; `train.parquet` has 9 retained rows and `val.parquet` has 6, both with the exact source schema. Usage was 211,515 input and 10,114 output tokens, costing approximately **$0.02981** including the retry. Scaling the observed cost per expression to 50,083 expressions at Batch prices gives **about $37**, before any further sharing benefit from larger groups. This small-pilot projection is not a quote; the full workload's preflight and $95 ledger allowance govern submission. The pilot verifies the workflow and surfaces examples for inspection, not a corpus-level quality score.

## Offline verification

~~~bash
concor-video verify \
  --parquet outputs/my-run/export/verification.parquet \
  --media-root /data/concor-video \
  --decisions edits/decisions.json \
  --output exports/verified.parquet \
  --port 8000
~~~

Open http://127.0.0.1:8000. For Ref-YT-VOS, the verifier suggests three tracklet-bearing instructions per video by default (change the count at the top), and verified export samples exactly one accepted instruction per accepted video. For ReVOS, a dedicated preview instead contains every nonexistent-object description plus one stable explicit and one stable implicit description for that video. ReVOS preview instructions start accepted and have their own per-instruction controls. The top `1`/`2` controls (and keyboard keys when enabled) accept or reject the **entire video** and advance to the next video. All remaining ReVOS instructions stay visible below; **Add to preview + accept** includes one as a persistent override. For an accepted video, every accepted base or override preview instruction enters export; a rejected or undecided video exports none. Each card shows its complete instruction with all exact linked spans. Click a colored span (or its link chip) to isolate its corresponding mask(s) in the video; **Preview all masks** restores the full instruction overlay. Existing `decisions.json` files remain compatible and retain all existing status values and edits. The adapter stores `review_mode`, `preview_sample_ids`, `preview_override_sample_ids`, per-instruction `status`, and `accepted_sample_ids`. Before the first server-side save of every verifier session, the exact input decisions file is copied to a timestamped `*.session-start.*.bak` file. Decisions then checkpoint in browser storage and, at a selectable interval, to `--decisions`; export saves and downloads both the decisions and updated Parquet without overwriting the source Parquet.

For compact storage, current campaign exports combine all training rows in one file and all evaluation rows (validation/test where publicly available) in another. Every row retains explicit `dataset` and `split` fields, so a small dataset/split filter can separate them losslessly; the final Hugging Face publishing wrapper will emit those splits.

See [verification details](verification/README.md), [pipeline notes](docs/PIPELINE.md), [dataset layouts](docs/DATASETS.md), [output contract](docs/OUTPUT_FORMAT.md), and [Slurm notes](docs/SLURM.md).

## Layout

~~~text
src/concor_video/  adapters, extraction, SAM3.1 processing, validation, export/server
verification/      offline browser UI and decision schema
scripts/           setup, staging, local and Slurm launchers
slurm/             portable worker, preparation, worklist, and export jobs
schema/            machine-readable record contracts
tests/             CPU-only unit/integration tests
~~~

Generated data, credentials, checkpoints, caches, and media are ignored. SAM/spaCy output remains fallible; complete_bcc means the deterministic representation checks passed, not that a human verified the semantic target. The verifier exists to correct those residual errors.

Code is MIT licensed. Ref-YouTube-VOS, ReVOS, and SAM 3.1 retain their own terms. ReVOS is non-commercial CC BY-NC-SA 4.0. Please cite the upstream Grounding as Concept Correspondence, SAM 3, URVOS/Ref-YouTube-VOS, and VISA/ReVOS work as applicable.
