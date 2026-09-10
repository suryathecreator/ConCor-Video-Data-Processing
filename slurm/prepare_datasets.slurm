#!/usr/bin/env bash
#SBATCH --job-name=concor-video-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
set -euo pipefail
: "${REPO_ROOT:?}" "${PYTHON_BIN:?}" "${DATA_ROOT:?}"
bash "${REPO_ROOT}/scripts/prepare_public_datasets.sh" "${DATA_ROOT}"
