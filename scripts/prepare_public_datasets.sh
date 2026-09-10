#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?usage: prepare_public_datasets.sh DATA_ROOT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
REF_ROOT="${DATA_ROOT}/ref-youtube-vos"; REVOS_ROOT="${DATA_ROOT}/ReVOS"; DOWNLOADS="${DATA_ROOT}/.downloads"
mkdir -p "${REF_ROOT}/archives" "${REVOS_ROOT}" "${DOWNLOADS}"
download() { local url="$1" output="$2"; if [[ ! -s "${output}" ]]; then mkdir -p "$(dirname "${output}")"; curl --fail --location --retry 8 --retry-all-errors --continue-at - --output "${output}.part" "${url}"; mv "${output}.part" "${output}"; fi; }
base="https://huggingface.co/datasets/blue7012/ref-youtube-vos/resolve/main"
download "${base}/valid_data/meta_expressions.zip" "${DOWNLOADS}/ref-meta-expressions.zip"
download "${base}/train_data/train.zip" "${REF_ROOT}/archives/train.zip"
download "${base}/train_data/valid.zip" "${REF_ROOT}/archives/valid.zip"
download "${base}/train_data/test_ytvos.zip" "${REF_ROOT}/archives/test_ytvos.zip"
unzip -tq "${DOWNLOADS}/ref-meta-expressions.zip" >/dev/null
unzip -q -o "${DOWNLOADS}/ref-meta-expressions.zip" -d "${REF_ROOT}"
if [[ ! -s "${REVOS_ROOT}/.staged-complete" ]]; then
  revos_tar="${DOWNLOADS}/ReVOS.tar"; download "https://huggingface.co/datasets/CIPLab-Video/decaf-rvos-revos/resolve/main/ReVOS.tar" "${revos_tar}"; tar -xf "${revos_tar}" -C "${DATA_ROOT}"
  for required in meta_expressions_train_.json meta_expressions_valid_.json mask_dict.json; do [[ -s "${REVOS_ROOT}/${required}" ]] || { echo "missing ${REVOS_ROOT}/${required}" >&2; exit 1; }; done
  [[ -d "${REVOS_ROOT}/JPEGImages" ]] || { echo "missing ReVOS/JPEGImages" >&2; exit 1; }
  touch "${REVOS_ROOT}/.staged-complete"; rm -f "${revos_tar}"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -s "${REVOS_ROOT}/mask_dict.sqlite" ]]; then "${PYTHON_BIN}" -m concor_video.cli index-revos-masks --source "${REVOS_ROOT}/mask_dict.json" --output "${REVOS_ROOT}/mask_dict.sqlite"; fi
printf 'Ref-YouTube-VOS: %s\nReVOS: %s\n' "${REF_ROOT}" "${REVOS_ROOT}"
