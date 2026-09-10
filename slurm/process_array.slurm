#!/usr/bin/env bash
#SBATCH --job-name=concor-video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
set -euo pipefail
: "${REPO_ROOT:?}" "${PYTHON_BIN:?}" "${CAMPAIGN_ROOT:?}" "${SAM31_REPO_ROOT:?}" "${SAM31_CHECKPOINT:?}" "${SHARD_COUNT:?}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"; array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"; restart_count="${SLURM_RESTART_COUNT:-0}"; scratch_base="${SLURM_TMPDIR:-/tmp}"
JOB_CACHE_ROOT="${scratch_base}/concor-video-${array_job_id}-${task_id}-r${restart_count}"
NODE_ASSET_ROOT="/tmp/${USER}-concor-video-assets"
mkdir -p "${JOB_CACHE_ROOT}/"{torch,triton,cuda,xdg,pycache} "${NODE_ASSET_ROOT}"
runtime_root=""
if [[ -n "${PYTHON_RUNTIME_ARCHIVE:-}" ]]; then
  runtime_key="$(stat -c '%s-%Y' "${PYTHON_RUNTIME_ARCHIVE}")"
  runtime_candidate="${NODE_ASSET_ROOT}/runtime-${runtime_key}"
  if (
    flock 9
    find "${NODE_ASSET_ROOT}" -maxdepth 1 -type d -name 'runtime-*.part-*' -exec rm -rf -- {} +
    if [[ ! -s "${runtime_candidate}/.complete" ]]; then
      temp="${runtime_candidate}.part-${SLURM_JOB_ID}-${task_id}"
      mkdir -p "${temp}"
      if ! tar -xzf "${PYTHON_RUNTIME_ARCHIVE}" -C "${temp}"; then
        rm -rf "${temp}"
        exit 1
      fi
      touch "${temp}/.complete"
      rm -rf "${runtime_candidate}"
      mv "${temp}" "${runtime_candidate}"
    fi
  ) 9>"${NODE_ASSET_ROOT}/runtime.lock"; then
    runtime_root="${runtime_candidate}"
  else
    echo "[staging] node-local runtime unavailable; using shared Python environment" >&2
  fi
fi
checkpoint_key="$(stat -c '%s-%Y' "${SAM31_CHECKPOINT}")"
checkpoint_candidate="${NODE_ASSET_ROOT}/sam31-${checkpoint_key}.pt"
LOCAL_CHECKPOINT="${SAM31_CHECKPOINT}"
if (
  flock 9
  find "${NODE_ASSET_ROOT}" -maxdepth 1 -type f -name 'sam31-*.pt.part-*' -delete
  if [[ ! -s "${checkpoint_candidate}" ]]; then
    part="${checkpoint_candidate}.part-${SLURM_JOB_ID}-${task_id}"
    if ! cp --reflink=auto "${SAM31_CHECKPOINT}" "${part}"; then
      rm -f "${part}"
      exit 1
    fi
    mv "${part}" "${checkpoint_candidate}"
  fi
) 9>"${NODE_ASSET_ROOT}/checkpoint.lock"; then
  LOCAL_CHECKPOINT="${checkpoint_candidate}"
else
  echo "[staging] node-local checkpoint unavailable; using shared checkpoint" >&2
fi
export XDG_CACHE_HOME="${JOB_CACHE_ROOT}/xdg" TORCHINDUCTOR_CACHE_DIR="${JOB_CACHE_ROOT}/torch" TRITON_CACHE_DIR="${JOB_CACHE_ROOT}/triton" CUDA_CACHE_PATH="${JOB_CACHE_ROOT}/cuda" PYTHONPYCACHEPREFIX="${JOB_CACHE_ROOT}/pycache"
export PYTHONPATH="${REPO_ROOT}/src:${SAM31_REPO_ROOT}${runtime_root:+:${runtime_root}/site-packages}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${CAMPAIGN_ROOT}/shared-cache/huggingface}"
# Decoded frames are reproducible scratch, not checkpoints. Keep them node-local
# to avoid shared-filesystem metadata traffic and output-file explosion.
FRAME_CACHE_PATH="${JOB_CACHE_ROOT}/frames"
if [[ "${PERSIST_FRAME_CACHE:-0}" == "1" ]]; then
  FRAME_CACHE_PATH="${FRAME_CACHE_ROOT:-${CAMPAIGN_ROOT}/cache}"
fi
mkdir -p "${HF_HOME}" "${FRAME_CACHE_PATH}"
flags=(--no-compile --no-warm-up --no-fa3)
[[ "${COMPILE_MODEL:-0}" == "1" ]] && flags=(--compile --warm-up --no-fa3)
[[ "${USE_FA3:-0}" == "1" ]] && flags+=(--fa3)
set +e
"${PYTHON_BIN}" -m concor_video.cli process --worklist "${CAMPAIGN_ROOT}/worklist.json" --campaign-root "${CAMPAIGN_ROOT}" --frame-cache-root "${FRAME_CACHE_PATH}" --checkpoint "${LOCAL_CHECKPOINT}" --shard-index "${task_id}" --shard-count "${SHARD_COUNT}" --max-error-attempts "${MAX_ERROR_ATTEMPTS:-3}" "${flags[@]}"
worker_rc=$?; set -e
if [[ "${worker_rc}" -eq 99 ]]; then scontrol requeue "${SLURM_JOB_ID}"; exit 0; fi
exit "${worker_rc}"
