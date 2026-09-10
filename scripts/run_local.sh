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
SAM31_CHECKPOINT="${SAM31_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam3.1/sam3.1_multiplex.pt}"
SEED="${SEED:-20260909}"

mkdir -p "${CAMPAIGN_ROOT}"
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

process=(
  "${VENV_PATH}/bin/concor-video" process
  --worklist "${CAMPAIGN_ROOT}/worklist.json"
  --campaign-root "${CAMPAIGN_ROOT}"
  --checkpoint "${SAM31_CHECKPOINT}"
  --shard-index 0
  --shard-count 1
)
[[ "${COMPILE_MODEL:-0}" == "1" ]] && process+=(--compile --warm-up)
[[ "${USE_FA3:-0}" == "1" ]] && process+=(--fa3)
"${process[@]}"

"${VENV_PATH}/bin/concor-video" export \
  --worklist "${CAMPAIGN_ROOT}/worklist.json" \
  --campaign-root "${CAMPAIGN_ROOT}"
