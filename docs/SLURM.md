# Slurm

## Full public campaign

~~~bash
DATA_ROOT=/data/concor-video \
CAMPAIGN_BASE=/outputs/concor-video-full \
PYTHON_BIN=$PWD/.venv/bin/python \
SAM31_REPO_ROOT=$PWD/external/sam3 \
SAM31_CHECKPOINT=$PWD/checkpoints/sam3.1/sam3.1_multiplex.pt \
SLURM_PARTITION=ckpt SLURM_ACCOUNT=your-account GPU_GRES=gpu:a40:1 \
NUM_WORKERS=32 bash scripts/submit_public_corpus.sh
~~~

The DAG is:

~~~text
prepare data -> build evaluation worklist -> evaluation GPU[0-31] -> evaluation export
             -> build train worklist -------> train GPU[0-31] -----> train export
~~~

Train GPU work additionally waits for the evaluation array and is submitted with a positive nice value. Arrays use 0-31, not an application-side percent cap; available resources and site policy provide scheduling limits.

## Portability

Required settings are SLURM_PARTITION, SLURM_ACCOUNT, and GPU_GRES. One worker requires one GPU. Change gpu:a40:1 to gpu:h200:1, gpu:a100:1, or your site's generic request without code changes. Tune CPU, memory, and time through CPUS_PER_TASK, MEMORY, and TIME_LIMIT. The default host-memory request is 64 GB; the full evaluation campaign peaked at 11.4 GB, leaving substantial headroom while permitting all eight A40s on a 512+ GB node to be scheduled.

PYTHON_RUNTIME_ARCHIVE may point to a tarball whose top level is site-packages/. Jobs unpack it once per node under a lock and prepend it to PYTHONPATH. The SAM checkpoint is likewise copied once per node. This avoids thousands of shared-filesystem imports and repeated 3+ GB checkpoint reads.

## Resumption

The same campaign can be submitted again. Existing records are skipped, stale leases are recovered, and only pending/retryable instructions run. Workers handle USR1 ten minutes before scheduled termination and requeue themselves. Submission IDs are appended to CAMPAIGN_BASE/submissions.txt.
