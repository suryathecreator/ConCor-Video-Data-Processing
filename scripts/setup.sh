#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
INSTALL_GPU="${INSTALL_GPU:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SAM31_REPO_ROOT="${SAM31_REPO_ROOT:-${REPO_ROOT}/external/sam3}"
CONCOR_CACHE_ROOT="${CONCOR_CACHE_ROOT:-${REPO_ROOT}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CONCOR_CACHE_ROOT}/pip}"
export HF_HOME="${HF_HOME:-${CONCOR_CACHE_ROOT}/huggingface}"
mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}"

"${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'
"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/python" -m pip install -e "${REPO_ROOT}[test]"
"${VENV_PATH}/bin/python" -m spacy download en_core_web_sm

if [[ "${INSTALL_GPU}" == "1" ]]; then
  if ! "${VENV_PATH}/bin/python" -c 'import torch' >/dev/null 2>&1; then
    "${VENV_PATH}/bin/python" -m pip install torch==2.10.0 torchvision \
      --index-url "${TORCH_INDEX_URL}"
  fi
  if [[ ! -f "${SAM31_REPO_ROOT}/pyproject.toml" ]]; then
    mkdir -p "$(dirname "${SAM31_REPO_ROOT}")"
    git clone https://github.com/facebookresearch/sam3.git "${SAM31_REPO_ROOT}"
  fi
  "${VENV_PATH}/bin/python" -m pip install -e "${SAM31_REPO_ROOT}"
fi

printf 'environment=%s\nsam31_source=%s\n' "${VENV_PATH}" "${SAM31_REPO_ROOT}"
