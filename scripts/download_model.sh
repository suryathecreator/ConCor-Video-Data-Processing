#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/sam3.1}"
CONCOR_CACHE_ROOT="${CONCOR_CACHE_ROOT:-${REPO_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CONCOR_CACHE_ROOT}/pip}"
export HF_HOME="${HF_HOME:-${CONCOR_CACHE_ROOT}/huggingface}"

mkdir -p "${CHECKPOINT_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}"
"${VENV_PATH}/bin/python" -m pip install --upgrade huggingface_hub
"${VENV_PATH}/bin/hf" download facebook/sam3.1 sam3.1_multiplex.pt \
  --local-dir "${CHECKPOINT_DIR}"
printf 'checkpoint=%s\n' "${CHECKPOINT_DIR}/sam3.1_multiplex.pt"
