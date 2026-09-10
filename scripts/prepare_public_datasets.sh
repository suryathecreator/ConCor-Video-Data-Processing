#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?usage: prepare_public_datasets.sh DATA_ROOT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
REF_ROOT="${DATA_ROOT}/ref-youtube-vos"; REVOS_ROOT="${DATA_ROOT}/ReVOS"; DOWNLOADS="${DATA_ROOT}/.downloads"
mkdir -p "${REF_ROOT}/archives" "${REVOS_ROOT}" "${DOWNLOADS}"
download() {
  local url="$1" output="$2"
  [[ -s "${output}" ]] && return 0
  mkdir -p "$(dirname "${output}")"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --continue=true --max-connection-per-server=4 --split=4 \
      --min-split-size=16M --file-allocation=none --max-tries=0 \
      --retry-wait=5 --allow-overwrite=true --auto-file-renaming=false \
      --dir="$(dirname "${output}")" --out="$(basename "${output}").part" "${url}"
  else
    curl --fail --location --retry 8 --retry-delay 5 --continue-at - \
      --output "${output}.part" "${url}"
  fi
  mv "${output}.part" "${output}"
}
base="https://huggingface.co/datasets/blue7012/ref-youtube-vos/resolve/main"
revos_tar="${DOWNLOADS}/ReVOS.tar"
pids=()
download "${base}/valid_data/meta_expressions.zip" "${DOWNLOADS}/ref-meta-expressions.zip" & pids+=("$!")
download "${base}/train_data/train.zip" "${REF_ROOT}/archives/train.zip" & pids+=("$!")
download "${base}/train_data/valid.zip" "${REF_ROOT}/archives/valid.zip" & pids+=("$!")
download "${base}/train_data/test_ytvos.zip" "${REF_ROOT}/archives/test_ytvos.zip" & pids+=("$!")
if [[ ! -s "${REVOS_ROOT}/.staged-complete" ]]; then
  download "https://huggingface.co/datasets/CIPLab-Video/decaf-rvos-revos/resolve/main/ReVOS.tar" "${revos_tar}" & pids+=("$!")
fi
for pid in "${pids[@]}"; do wait "${pid}"; done
for archive in "${DOWNLOADS}/ref-meta-expressions.zip" "${REF_ROOT}/archives/train.zip" "${REF_ROOT}/archives/valid.zip" "${REF_ROOT}/archives/test_ytvos.zip"; do
  unzip -tq "${archive}" >/dev/null
done
unzip -q -o "${DOWNLOADS}/ref-meta-expressions.zip" -d "${REF_ROOT}"
if [[ ! -s "${REVOS_ROOT}/.staged-complete" ]]; then
  tar -xf "${revos_tar}" -C "${DATA_ROOT}"
  for required in meta_expressions_train_.json meta_expressions_valid_.json mask_dict.json; do [[ -s "${REVOS_ROOT}/${required}" ]] || { echo "missing ${REVOS_ROOT}/${required}" >&2; exit 1; }; done
  [[ -d "${REVOS_ROOT}/JPEGImages" ]] || { echo "missing ReVOS/JPEGImages" >&2; exit 1; }
  touch "${REVOS_ROOT}/.staged-complete"; rm -f "${revos_tar}"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -s "${REVOS_ROOT}/mask_dict.sqlite" ]]; then "${PYTHON_BIN}" -m concor_video.cli index-revos-masks --source "${REVOS_ROOT}/mask_dict.json" --output "${REVOS_ROOT}/mask_dict.sqlite"; fi
printf 'Ref-YouTube-VOS: %s\nReVOS: %s\n' "${REF_ROOT}" "${REVOS_ROOT}"
