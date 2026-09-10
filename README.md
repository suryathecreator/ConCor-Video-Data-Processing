# ConCor Video Data Processing

ConCor Video turns referring-video annotations into **text span ↔ instance tracklet** correspondences. It keeps an official target tracklet when the dataset provides one, extracts independently visible context entities with spaCy, and uses SAM 3.1 Object Multiplex to segment and track those additional entities. The result preserves both directions of the relationship: every tracklet knows its exact text spans, and every span knows all of its tracklets.

This is the cleaned research release of the Ref-YouTube-VOS/ReVOS pipeline. It is deterministic at selection time, checkpointed per expression, safe to shard across preemptible workers, and exports ordinary Parquet tables plus a full run ledger for a future verification interface.

## Quick setup

You need Python 3.12+, CUDA 12.6+, a compatible NVIDIA GPU, access to the gated SAM 3.1 checkpoint, and the upstream datasets under their own terms.

```bash
git clone https://github.com/suryathecreator/ConCor-Video-Data-Processing.git
cd ConCor-Video-Data-Processing
bash scripts/setup.sh
source .venv/bin/activate
hf auth login
bash scripts/download_model.sh
```

`scripts/setup.sh` installs PyTorch CUDA 12.8 by default and clones the official SAM 3 repository. Set `INSTALL_GPU=0` for a CPU-only preparation/export environment, or set `TORCH_INDEX_URL` and `SAM31_REPO_ROOT` for an existing cluster runtime.

## Run locally

Ref-YouTube-VOS train, full sequence:

```bash
DATASET=refytvos DATA_ROOT=/data/ref-youtube-vos SPLIT=train \
REF_MODE=full_video LIMIT=100 CAMPAIGN_ROOT=$PWD/outputs/ref-train-100 \
bash scripts/run_local.sh
```

Process only the first frame of each selected Ref-YouTube-VOS expression:

```bash
DATASET=refytvos DATA_ROOT=/data/ref-youtube-vos SPLIT=train \
REF_MODE=first_frame LIMIT=100 CAMPAIGN_ROOT=$PWD/outputs/ref-first-100 \
bash scripts/run_local.sh
```

The public Ref-YouTube-VOS language release contains **full-video-authored expressions only**. `REF_MODE=first_frame` therefore limits model processing to frame zero; it does not claim to recover the retired first-frame-language annotations. `SPLIT` may be `train`, `val`, or `test`. Train uses official target masks when they are present. Public val/test data do not include target masks, so their main referent is explicitly stored as a SAM3.1 prediction rather than mislabeled as ground truth.

ReVOS train or validation, with any combination of categories:

```bash
DATASET=revos DATA_ROOT=/data/ReVOS SPLIT=train \
REVOS_CATEGORIES="implicit explicit" LIMIT_PER_CATEGORY=100 \
CAMPAIGN_ROOT=$PWD/outputs/revos-train-200 bash scripts/run_local.sh
```

Use `SPLIT=val` for validation. Add `nonexistent` to retain nonexistent-object descriptions as intentional zero-tracklet negatives. Omit `LIMIT`/`LIMIT_PER_CATEGORY` to process every matching expression.

## Submit to Slurm

The same variables work with the checkpoint-friendly array launcher:

```bash
DATASET=refytvos DATA_ROOT=/data/ref-youtube-vos SPLIT=train \
REF_MODE=full_video LIMIT=1000 CAMPAIGN_ROOT=$PWD/outputs/ref-train-1k \
NUM_WORKERS=8 SLURM_PARTITION=your_gpu_partition \
SLURM_ACCOUNT=your_account GPU_GRES=gpu:a40:1 \
bash scripts/submit_slurm.sh
```

For another partition or GPU, change only `SLURM_PARTITION` and `GPU_GRES` (for example `gpu:h200:1` or your site's generic `gpu:1`). The worker asks for one GPU and has no cluster-specific path or account baked in. Set `TIME_LIMIT`, `MEMORY`, `CPUS_PER_TASK`, or `EXPORT_PARTITION` when your scheduler needs different resources.

Workers keep SAM3.1 loaded, assign samples deterministically by array index, stage weights and compilation caches in a job-specific node-local directory, and atomically commit one JSON record per expression. On `USR1`, a worker finishes its current expression and requeues. Rerunning the same command reuses the worklist and completed records.

## Pipeline

1. **Select expressions** — dataset adapters select a seeded subset from the requested split, view, and reasoning categories.
2. **Extract entities** — spaCy separates exact text spans from short semantic prompts. Concrete context such as `arm`, `book`, or `road` is eligible; abstract/meta language and unsafe generic `thing(s)` are retained in the audit but not prompted.
3. **Anchor the referent** — official dataset tracklets are used when available. Otherwise SAM3.1 predicts the main referent and the provenance says so.
4. **Track context** — one warm SAM3.1 Multiplex predictor grounds each unique prompt and tracks all matching instances. A plural span can link to several tracklets.
5. **Filter and link** — low-confidence/short/tiny tracks are removed; temporal IoU merges duplicates. Exact Unicode character spans are linked in both directions.
6. **Export** — records become `samples.parquet`, `tracklets.parquet`, `links.parquet`, and `run_ledger.csv`.

See [the pipeline notes](docs/PIPELINE.md), [dataset layouts](docs/DATASETS.md), [output contract](docs/OUTPUT_FORMAT.md), and [cluster notes](docs/SLURM.md).

## Output at a glance

```text
outputs/my-run/
  worklist.json
  worklist-extraction-audit.md
  records/<sample_id>.json
  errors/<sample_id>.json
  cache/frames/<sample_id>/*.jpg
  export/
    samples.parquet
    tracklets.parquet
    links.parquet
    run_ledger.csv
    manifest.json
```

`tracklets.parquet` is the direct verification input: one row per temporal instance, with scalar metadata and JSON columns for aligned COCO RLE masks, frame IDs, frame files, and linked spans. `samples.parquet` retains completed nonexistent-object negatives even though they have no tracklet rows. The ledger covers every selected expression and labels it `completed`, `failed`, or `pending`.

## Repository layout

```text
src/concor_video/   dataset adapters, extraction, SAM3.1 processing, validation, export
scripts/            setup, checkpoint download, local and Slurm launchers
slurm/              portable worker and export jobs
schema/             record and Parquet field contracts
docs/               data, pipeline, output, and cluster details
tests/              CPU-only unit and integration tests
```

## Scope and limitations

SAM masks and spaCy extraction are useful but not perfect. `complete_bcc` means every required prompt returned a retained tracklet and the representation passed deterministic integrity checks; it does not replace human verification. In particular, a strong semantic prompt can still select the wrong instance, and language heuristics can over- or under-extract contextual entities. The audit fields and flat Parquet layout are intentionally kept so those mistakes can be reviewed later.

Code is MIT licensed. Ref-YouTube-VOS, ReVOS, SAM 3.1, and their media/checkpoints retain their own licenses and access terms. ReVOS is non-commercial CC BY-NC-SA 4.0. No dataset media, model weights, generated outputs, private paths, or credentials are included here.

If this is useful, please cite the upstream *Grounding as Concept Correspondence*, SAM 3, URVOS/Ref-YouTube-VOS, and VISA/ReVOS work as applicable.
