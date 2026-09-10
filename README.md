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

The script keeps Ref-YT-VOS in validated ZIP archives, extracts ReVOS, and builds a shared read-only SQLite index for its large mask dictionary. Large archives use parallel range downloads when aria2 is available, and resume after interruption.

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

Nonexistent ReVOS descriptions are preserved as intentional zero-tracklet negatives.

## Offline verification

~~~bash
concor-video verify \
  --parquet outputs/my-run/export/verification.parquet \
  --media-root /data/concor-video \
  --decisions edits/decisions.json \
  --output exports/verified.parquet \
  --port 8000
~~~

Open http://127.0.0.1:8000. All instructions for one video appear together. You can edit language, create a span by selecting text, link it to one or many tracklets, delete spans/tracklets, revert an instruction or entire video, and accept/reject by video. Optional 1/2 quick keys accept/reject and advance. Decisions autosave in browser storage; import/export decisions.json works across sessions, and **Export updated Parquet** downloads a new file without overwriting the source.

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
