#!/usr/bin/env bash
#SBATCH --job-name=concor-video-list
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
set -euo pipefail
: "${REPO_ROOT:?}" "${PYTHON_BIN:?}" "${DATA_ROOT:?}" "${CAMPAIGN_ROOT:?}" "${SCOPE:?}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m concor_video.cli build-release-worklist --scope "${SCOPE}" --refytvos-root "${DATA_ROOT}/ref-youtube-vos" --revos-root "${DATA_ROOT}/ReVOS" --campaign-root "${CAMPAIGN_ROOT}" --seed "${SEED:-20260909}"
