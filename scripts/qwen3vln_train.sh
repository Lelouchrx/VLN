#!/bin/bash
set -e

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

eval "$(conda shell.bash hook)"
conda activate streamvln

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

CHECKPOINT="/media/mldadmin/home/s125mdg38_06/StreamVLN/checkpoints/checkpoints/Qwen3-VL-4B-Instruct"
OUTPUT_DIR="/media/mldadmin/home/s125mdg38_06/StreamVLN/checkpoints/checkpoints/Qwen3-VL-4B-Instruct_TUNE"
CACHE_DIR="./cache"
DATASETS="data/trajectory_data/R2R,data/trajectory_data/RxR"

mkdir -p "$OUTPUT_DIR" logs

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  vln/qwenvl/train/train_qwen.py \
  --model_name_or_path $CHECKPOINT \
  --tune_mm_llm False \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --tune_mm_llm_high True \
  --dataset_use $DATASETS \
  --output_dir $OUTPUT_DIR \
  --cache_dir $CACHE_DIR \
  --bf16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --mm_projector_lr 1e-5 \
  --model_max_length 32768 \
  --max_pixels $((256*28*28)) \
  --min_pixels $((16*28*28)) \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --save_steps 200 \
  --save_total_limit 1 \
  --gradient_checkpointing \
  --dataloader_num_workers 0 \
  --dataloader_persistent_workers False \
  --num_frames 8 \
  --seed 42 \
  --report_to none \
  --deepspeed scripts/zero3.json \
  > ${OUTPUT_DIR}/train_qwen.log 2>&1
