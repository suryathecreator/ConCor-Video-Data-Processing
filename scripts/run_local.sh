#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASET="${DATASET:?set DATASET=refytvos or DATASET=revos}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the staged dataset}"
SPLIT="${SPLIT:-train}"
REVOS_CATEGORIES="${REVOS_CATEGORIES:-implicit explicit nonexistent}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${REPO_ROOT}/outputs/${DATASET}-${SPLIT}}"
SAM31_CHECKPOINT="${SAM31_CHECKPOINT:-${REPO_ROOT}/checkpoints/sam3.1/sam3.1_multiplex.pt}"
FRAME_CACHE_ROOT="${FRAME_CACHE_ROOT:-${CAMPAIGN_ROOT}/cache}"
SEED="${SEED:-20260909}"
mkdir -p "${CAMPAIGN_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -s "${CAMPAIGN_ROOT}/worklist.json" ]]; then
  prepare=("${PYTHON_BIN}" -m concor_video.cli build-worklist --dataset "${DATASET}" --data-root "${DATA_ROOT}" --split "${SPLIT}" --campaign-root "${CAMPAIGN_ROOT}" --seed "${SEED}")
  if [[ "${DATASET}" == "refytvos" ]]; then [[ -n "${LIMIT:-}" ]] && prepare+=(--limit "${LIMIT}"); else read -r -a category_values <<< "${REVOS_CATEGORIES//,/ }"; prepare+=(--revos-categories "${category_values[@]}"); [[ -n "${LIMIT_PER_CATEGORY:-}" ]] && prepare+=(--limit-per-category "${LIMIT_PER_CATEGORY}"); fi
  [[ "${ONE_EXPRESSION_PER_VIDEO:-0}" == "1" ]] && prepare+=(--one-expression-per-video)
  "${prepare[@]}"
fi
flags=(--no-compile --no-warm-up --no-fa3)
[[ "${COMPILE_MODEL:-0}" == "1" ]] && flags=(--compile --warm-up --no-fa3)
[[ "${USE_FA3:-0}" == "1" ]] && flags+=(--fa3)
"${PYTHON_BIN}" -m concor_video.cli process --worklist "${CAMPAIGN_ROOT}/worklist.json" --campaign-root "${CAMPAIGN_ROOT}" --frame-cache-root "${FRAME_CACHE_ROOT}" --checkpoint "${SAM31_CHECKPOINT}" --shard-index 0 --shard-count 1 "${flags[@]}"
"${PYTHON_BIN}" -m concor_video.cli export --worklist "${CAMPAIGN_ROOT}/worklist.json" --campaign-root "${CAMPAIGN_ROOT}"
