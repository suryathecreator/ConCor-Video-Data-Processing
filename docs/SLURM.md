# Slurm and arbitrary GPU partitions

The launcher contains no user, account, filesystem, partition, or GPU-model defaults. Supply your site values at submission time:

```bash
DATASET=revos DATA_ROOT=/datasets/ReVOS SPLIT=val \
REVOS_CATEGORIES="implicit explicit nonexistent" \
CAMPAIGN_ROOT=$PWD/outputs/revos-val \
NUM_WORKERS=4 SLURM_PARTITION=gpu SLURM_ACCOUNT=my_account \
GPU_GRES=gpu:1 bash scripts/submit_slurm.sh
```

Named resources work too:

```bash
GPU_GRES=gpu:a40:1   # one A40 per worker
GPU_GRES=gpu:h200:1  # one H200 per worker
```

SAM3.1 runs on one sufficiently capable GPU per worker. Increase `NUM_WORKERS` for data parallelism; do not assign the same physical GPU to multiple workers. Defaults are 16 CPU cores, 120 GB RAM, and 12 hours. Override `CPUS_PER_TASK`, `MEMORY`, and `TIME_LIMIT` as needed.

The export job is CPU-only. Set `EXPORT_PARTITION` if your cluster has a dedicated CPU partition. It uses an `afterany` dependency so even a partially failed worker array produces a truthful ledger and Parquet files for completed records. Resubmit the same campaign to fill pending/failed records, then rerun `concor-video export`.

For short campaigns, eager inference is usually faster because compilation may not amortize. For a long campaign:

```bash
COMPILE_MODEL=1 ... bash scripts/submit_slurm.sh
```

Each task uses a job-and-array-specific node-local cache and checkpoint copy, which prevents workers from corrupting one another. Model source and weights remain shared read-only inputs.
