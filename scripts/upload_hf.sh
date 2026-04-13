#!/usr/bin/env bash
set -euo pipefail

# Fast uploader launcher for VLN trajectory data.
# Usage:
#   HF_TOKEN=xxx bash VLN/scripts/upload_hf.sh
#   HF_TOKEN=xxx bash VLN/scripts/upload_hf.sh --part-size-gb 10 --upload-workers 32

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_SCRIPT="${PROJECT_ROOT}/tool/upload_hf.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Missing python script: ${PY_SCRIPT}"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" ]]; then
  echo "[ERROR] Please export HF_TOKEN or HUGGINGFACE_TOKEN first."
  exit 2
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

echo "[INFO] Installing/Updating required Python packages..."
python -m pip install -U "hf-transfer" "tqdm" >/dev/null

SOURCE_ROOT="/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/trajectory_data"
REPO_ID="Lelouchrx/vln_data"
SUBSETS="R2R,RxR,ScaleVLN,EnvDrop"


# Auto-tune workers for higher throughput.
CPU_CORES="$(nproc)"
PACK_WORKERS="${PACK_WORKERS:-4}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-$((CPU_CORES * 2))}"
if (( UPLOAD_WORKERS < 8 )); then
  UPLOAD_WORKERS=8
fi

echo "[INFO] Starting fast upload..."
python "${PY_SCRIPT}" \
  --source-root "${SOURCE_ROOT}" \
  --repo-id "${REPO_ID}" \
  --repo-type dataset \
  --subsets "${SUBSETS}" \
  --work-dir "/tmp/vln_hf_upload" \
  --path-in-repo "trajectory_data_chunks" \
  --part-size-gb "${PART_SIZE_GB:-8}" \
  --compress-level "${COMPRESS_LEVEL:-1}" \
  --pack-workers "${PACK_WORKERS}" \
  --upload-workers "${UPLOAD_WORKERS}" \
  --max-retries "${MAX_RETRIES:-1000}" \
  "$@"

echo "[DONE] Upload finished."
echo "[TIP] Verify by download:"
echo "      hf download Lelouchrx/vln_data --repo-type=dataset"
