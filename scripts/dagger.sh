#!/bin/bash
umask 000
set -euo pipefail
set -x

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))


DAGGER_DATASET=R2R
DAGGER_DATA_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/r2r/train/train.json.gz
DAGGER_GT_ANNOTATIONS_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/r2r/train/train_gt.json.gz

# DAGGER_DATASET=RxR
# DAGGER_DATA_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/rxr/train/train_guide.json.gz
# DAGGER_GT_ANNOTATIONS_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/rxr/train/train_guide_gt.json.gz

# DAGGER_DATASET=EnvDrop
# DAGGER_DATA_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/r2r/envdrop/envdrop.json.gz
# DAGGER_GT_ANNOTATIONS_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/datasets/r2r/envdrop/envdrop_gt.json.gz


DAGGER_UPDATE_SIZE=16000 # Max size of the dataset to be collected
DAGGER_COMMIT_FREQ=1 # dump data every DAGGER_COMMIT_FREQ updates
DAGGER_P=0 # allow model inference
DAGGER_DATA_IT=3 # not used if DAGGER_P=0
PARALLEL_ENVS=8

MID_RUN_NAME="Qwen3VL_2B_R2R_RxR_swift"

CHECKPOINT="${MID_RUN_NAME}"
echo "CHECKPOINT: ${CHECKPOINT}"

DAGGER_OUTPUT_PATH=/mnt/share172/cs22-hongly/DATACENTER/VLN_DATA/dagger_data/${DAGGER_DATASET}
VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8002/v1}
VLLM_MODEL_NAME=${VLLM_MODEL_NAME:-qwen3vl}

mkdir -p ${DAGGER_OUTPUT_PATH}

export CUDA_VISIBLE_DEVICES=4,5

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} vln/dagger.py \
    --vllm_base_url ${VLLM_BASE_URL} \
    --vllm_model_name ${VLLM_MODEL_NAME} \
    --vllm_max_workers ${PARALLEL_ENVS} \
    --parallel_envs ${PARALLEL_ENVS} \
    --dagger_dataset ${DAGGER_DATASET} \
    --dagger_data_path ${DAGGER_DATA_PATH} \
    --dagger_update_size ${DAGGER_UPDATE_SIZE} \
    --dagger_commit_freq ${DAGGER_COMMIT_FREQ} \
    --dagger_p ${DAGGER_P} \
    --dagger_data_it ${DAGGER_DATA_IT} \
    --dagger_output_path "${DAGGER_OUTPUT_PATH}" \
    --output_path "${DAGGER_OUTPUT_PATH}" \
    --dagger_gt_annotations_path ${DAGGER_GT_ANNOTATIONS_PATH} \
    # --dagger-save-model-trace
    # --no-collect_images