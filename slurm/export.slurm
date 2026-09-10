#!/usr/bin/env bash
#SBATCH --job-name=concor-video-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append

set -euo pipefail

: "${REPO_ROOT:?}"
: "${VENV_PATH:?}"
: "${CAMPAIGN_ROOT:?}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${VENV_PATH}/bin/concor-video" export \
  --worklist "${CAMPAIGN_ROOT}/worklist.json" \
  --campaign-root "${CAMPAIGN_ROOT}"
