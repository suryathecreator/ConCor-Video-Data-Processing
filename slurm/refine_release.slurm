#!/bin/bash
#SBATCH --job-name=concor-luna-release
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14-00:00:00
#SBATCH --requeue

set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT to the refinement checkout}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Credentials are loaded by the Python wrapper from the ignored .env, never
# placed in scheduler arguments, copied to a job file, or printed in logs.
exec .venv/bin/python scripts/refine_release.py "$@"
