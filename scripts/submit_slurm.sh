#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
DATASET="${DATASET:?set DATASET=refytvos or DATASET=revos}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the unpacked dataset}"
SPLIT="${SPLIT:-train}"
REF_MODE="${REF_MODE:-full_video}"
REVOS_CATEGORIES="${REVOS_CATEGORIES:-implicit explicit nonexistent}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${REPO_ROOT}/outputs/${DATASET}-${SPLIT}}"
SAM31_REPO_ROOT="${SAM31_REPO_ROOT:-${REPO_ROOT}/external/sam3}"
SAM31_CHECKPOINT="${SAM31_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam3.1/sam3.1_multiplex.pt}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SLURM_PARTITION="${SLURM_PARTITION:?set SLURM_PARTITION}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
GPU_GRES="${GPU_GRES:-gpu:1}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-120G}"
SEED="${SEED:-20260909}"

if (( NUM_WORKERS < 1 )); then
  printf 'NUM_WORKERS must be at least 1\n' >&2
  exit 2
fi
mkdir -p "${CAMPAIGN_ROOT}/slurm_logs"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -s "${CAMPAIGN_ROOT}/worklist.json" ]]; then
  prepare=(
    "${VENV_PATH}/bin/concor-video" build-worklist
    --dataset "${DATASET}"
    --data-root "${DATA_ROOT}"
    --split "${SPLIT}"
    --campaign-root "${CAMPAIGN_ROOT}"
    --seed "${SEED}"
  )
  if [[ "${DATASET}" == "refytvos" ]]; then
    prepare+=(--ref-mode "${REF_MODE}")
    [[ -n "${LIMIT:-}" ]] && prepare+=(--limit "${LIMIT}")
  else
    read -r -a category_values <<< "${REVOS_CATEGORIES//,/ }"
    prepare+=(--revos-categories "${category_values[@]}")
    [[ -n "${LIMIT_PER_CATEGORY:-}" ]] && prepare+=(--limit-per-category "${LIMIT_PER_CATEGORY}")
  fi
  [[ "${ONE_EXPRESSION_PER_VIDEO:-0}" == "1" ]] && prepare+=(--one-expression-per-video)
  "${prepare[@]}"
fi

array_end=$((NUM_WORKERS - 1))
worker_job="$(sbatch --parsable \
  --partition="${SLURM_PARTITION}" \
  --account="${SLURM_ACCOUNT}" \
  --gres="${GPU_GRES}" \
  --time="${TIME_LIMIT}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY}" \
  --array="0-${array_end}%${NUM_WORKERS}" \
  --output="${CAMPAIGN_ROOT}/slurm_logs/worker-%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/slurm_logs/worker-%A_%a.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",VENV_PATH="${VENV_PATH}",CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",SAM31_REPO_ROOT="${SAM31_REPO_ROOT}",SAM31_CHECKPOINT="${SAM31_CHECKPOINT}",SHARD_COUNT="${NUM_WORKERS}" \
  "${REPO_ROOT}/slurm/process_array.slurm")"

export_partition="${EXPORT_PARTITION:-${SLURM_PARTITION}}"
export_job="$(sbatch --parsable \
  --partition="${export_partition}" \
  --account="${SLURM_ACCOUNT}" \
  --time="${EXPORT_TIME_LIMIT:-01:00:00}" \
  --cpus-per-task="${EXPORT_CPUS:-4}" \
  --mem="${EXPORT_MEMORY:-32G}" \
  --dependency="afterany:${worker_job}" \
  --output="${CAMPAIGN_ROOT}/slurm_logs/export-%j.out" \
  --error="${CAMPAIGN_ROOT}/slurm_logs/export-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",VENV_PATH="${VENV_PATH}",CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  "${REPO_ROOT}/slurm/export.slurm")"

printf 'worker_array=%s\nexport=%s\ncampaign=%s\n' \
  "${worker_job}" "${export_job}" "${CAMPAIGN_ROOT}"
