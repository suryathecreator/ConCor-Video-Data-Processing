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
NODE_ASSET_ROOT="/tmp/${USER}-concor-video-assets"
mkdir -p "${NODE_ASSET_ROOT}"
# Remove only abandoned atomic parts from this user-owned cache. Locks ensure a
# concurrent task on the same node cannot be writing a matching part.
( flock 9; find "${NODE_ASSET_ROOT}" -maxdepth 1 -type d -name 'runtime-*.part-*' -exec rm -rf -- {} + ) 9>"${NODE_ASSET_ROOT}/runtime.lock"
( flock 9; find "${NODE_ASSET_ROOT}" -maxdepth 1 -type f -name 'sam31-*.pt.part-*' -delete ) 9>"${NODE_ASSET_ROOT}/checkpoint.lock"
scratch_available_kb="$(df -Pk "${scratch_base}" | awk 'NR==2 {print $4}')"
local_stage_ok=1
if (( scratch_available_kb < 5 * 1024 * 1024 )); then local_stage_ok=0; fi
if (( local_stage_ok )); then
  JOB_CACHE_ROOT="${scratch_base}/concor-video-${array_job_id}-${task_id}-r${restart_count}"
else
  JOB_CACHE_ROOT="${CAMPAIGN_ROOT}/fallback-job-cache/${array_job_id}-${task_id}-r${restart_count}"
  echo "[staging] low node-local space (${scratch_available_kb} KiB); using shared mutable scratch" >&2
fi
mkdir -p "${JOB_CACHE_ROOT}/"{torch,triton,cuda,xdg,pycache}
cleanup_job_cache() { rm -rf -- "${JOB_CACHE_ROOT}"; }
trap cleanup_job_cache EXIT
runtime_root=""
if [[ -n "${PYTHON_RUNTIME_ARCHIVE:-}" ]]; then
  runtime_key="$(stat -c '%s-%Y' "${PYTHON_RUNTIME_ARCHIVE}")"
  if (( local_stage_ok )); then
    runtime_candidate="${NODE_ASSET_ROOT}/runtime-${runtime_key}"
    runtime_lock="${NODE_ASSET_ROOT}/runtime.lock"
  else
    # A full node-local /tmp must not silently fall back to the deliberately
    # minimal driver environment. Materialize the immutable runtime once on
    # campaign storage and share it under a cross-node flock instead.
    shared_runtime_root="${CAMPAIGN_ROOT}/shared-cache/python-runtimes"
    mkdir -p "${shared_runtime_root}"
    runtime_candidate="${shared_runtime_root}/runtime-${runtime_key}"
    runtime_lock="${shared_runtime_root}/runtime.lock"
  fi
  if (
    flock 9
    if [[ ! -f "${runtime_candidate}/.complete" ]] || \
       [[ ! -f "${runtime_candidate}/site-packages/pycocotools/__init__.py" ]]; then
      temp="${runtime_candidate}.part-${SLURM_JOB_ID}-${task_id}"
      rm -rf -- "${temp}"
      mkdir -p "${temp}"
      if ! tar -xzf "${PYTHON_RUNTIME_ARCHIVE}" -C "${temp}"; then
        rm -rf "${temp}"
        exit 1
      fi
      [[ -f "${temp}/site-packages/pycocotools/__init__.py" ]] || {
        echo "[staging] runtime archive lacks pycocotools" >&2
        rm -rf "${temp}"
        exit 1
      }
      printf '%s\n' "${runtime_key}" > "${temp}/.complete"
      rm -rf "${runtime_candidate}"
      mv "${temp}" "${runtime_candidate}"
    fi
  ) 9>"${runtime_lock}"; then
    runtime_root="${runtime_candidate}"
  else
    echo "[staging] packaged runtime unavailable; using shared Python environment" >&2
  fi
fi
checkpoint_key="$(stat -c '%s-%Y' "${SAM31_CHECKPOINT}")"
checkpoint_candidate="${NODE_ASSET_ROOT}/sam31-${checkpoint_key}.pt"
LOCAL_CHECKPOINT="${SAM31_CHECKPOINT}"
if (( local_stage_ok )) && (
  flock 9
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
"${PYTHON_BIN}" -m concor_video.cli process --worklist "${WORKLIST_PATH:-${CAMPAIGN_ROOT}/worklist.json}" --campaign-root "${CAMPAIGN_ROOT}" --frame-cache-root "${FRAME_CACHE_PATH}" --checkpoint "${LOCAL_CHECKPOINT}" --shard-index "${task_id}" --shard-count "${SHARD_COUNT}" --max-error-attempts "${MAX_ERROR_ATTEMPTS:-3}" "${flags[@]}"
worker_rc=$?; set -e
if [[ "${worker_rc}" -eq 99 ]]; then scontrol requeue "${SLURM_JOB_ID}"; exit 0; fi
exit "${worker_rc}"
