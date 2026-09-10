#!/usr/bin/env bash
#SBATCH --job-name=concor-video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append

set -euo pipefail

: "${REPO_ROOT:?}"
: "${VENV_PATH:?}"
: "${CAMPAIGN_ROOT:?}"
: "${SAM31_REPO_ROOT:?}"
: "${SAM31_CHECKPOINT:?}"
: "${SHARD_COUNT:?}"

JOB_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/concor-video-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${JOB_CACHE_ROOT}/model" "${JOB_CACHE_ROOT}/torch" "${JOB_CACHE_ROOT}/triton"
export XDG_CACHE_HOME="${JOB_CACHE_ROOT}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${JOB_CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${JOB_CACHE_ROOT}/cuda"
export PYTHONPYCACHEPREFIX="${JOB_CACHE_ROOT}/pycache"
export PYTHONPATH="${REPO_ROOT}/src:${SAM31_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

LOCAL_CHECKPOINT="${JOB_CACHE_ROOT}/model/sam3.1_multiplex.pt"
cp "${SAM31_CHECKPOINT}" "${LOCAL_CHECKPOINT}.part"
mv "${LOCAL_CHECKPOINT}.part" "${LOCAL_CHECKPOINT}"

flags=(--no-compile --no-warm-up --no-fa3)
if [[ "${COMPILE_MODEL:-0}" == "1" ]]; then
  flags=(--compile --warm-up --no-fa3)
fi
if [[ "${USE_FA3:-0}" == "1" ]]; then
  flags+=(--fa3)
fi

set +e
"${VENV_PATH}/bin/concor-video" process \
  --worklist "${CAMPAIGN_ROOT}/worklist.json" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --checkpoint "${LOCAL_CHECKPOINT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-count "${SHARD_COUNT}" \
  "${flags[@]}"
worker_rc=$?
set -e

if [[ ${worker_rc} -eq 99 ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${worker_rc}"
