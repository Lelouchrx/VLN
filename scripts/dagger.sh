#!/bin/bash
umask 000
set -euo pipefail
set -x

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# R2R
R2R_DATA_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/r2r/train/train.json.gz
R2R_GT_ANNOTATIONS_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/r2r/train/train_gt.json.gz

# RxR
RXR_DATA_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/rxr/train/train_guide_en.json.gz
RXR_GT_ANNOTATIONS_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/rxr/train/train_guide_gt.json.gz

# # EnvDrop (optional)
# ENVDROP_DATA_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/r2r/envdrop/envdrop.json.gz
# ENVDROP_GT_ANNOTATIONS_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/r2r/envdrop/envdrop_gt.json.gz


DAGGER_UPDATE_SIZE=20000 # Max size of the dataset to be collected
DAGGER_COMMIT_FREQ=1 # dump data every DAGGER_COMMIT_FREQ updates
DAGGER_P=0 # allow model inference
DAGGER_DATA_IT=3 # not used if DAGGER_P=0
PARALLEL_ENVS=8

MID_RUN_NAME="Qwen3VL_8B_R2R_RxR_EnvDrop_swift"

CHECKPOINT="${MID_RUN_NAME}"
echo "CHECKPOINT: ${CHECKPOINT}"

VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8001/v1}
VLLM_MODEL_NAME=${VLLM_MODEL_NAME:-qwen3vl}

export CUDA_VISIBLE_DEVICES=6,7

run_dagger_collect() {
    local dagger_dataset="$1"
    local dagger_data_path="$2"
    local dagger_gt_annotations_path="$3"
    local master_port=$((RANDOM % 101 + 20000))
    local dagger_output_path=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/dagger_data/${dagger_dataset}

    mkdir -p "${dagger_output_path}"

    torchrun --nproc_per_node=2 --master_port="${master_port}" vln/dagger.py \
        --vllm_base_url ${VLLM_BASE_URL} \
        --vllm_model_name ${VLLM_MODEL_NAME} \
        --vllm_max_workers ${PARALLEL_ENVS} \
        --parallel_envs ${PARALLEL_ENVS} \
        --dagger_dataset "${dagger_dataset}" \
        --dagger_data_path "${dagger_data_path}" \
        --dagger_update_size ${DAGGER_UPDATE_SIZE} \
        --dagger_commit_freq ${DAGGER_COMMIT_FREQ} \
        --dagger_p ${DAGGER_P} \
        --dagger_data_it ${DAGGER_DATA_IT} \
        --dagger_output_path "${dagger_output_path}" \
        --output_path "${dagger_output_path}" \
        --dagger_gt_annotations_path "${dagger_gt_annotations_path}"
        # --dagger-save-model-trace
        # --no-collect_images
}

# 先 R2R，完成后 RxR
run_dagger_collect R2R "${R2R_DATA_PATH}" "${R2R_GT_ANNOTATIONS_PATH}"
run_dagger_collect RxR "${RXR_DATA_PATH}" "${RXR_GT_ANNOTATIONS_PATH}"
