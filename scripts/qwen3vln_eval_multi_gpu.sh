export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

# 激活conda环境
source /media/mldadmin/home/s125mdg38_06/miniconda3/bin/activate streamvln
# Fix libtinfo.so.6 library conflict by prioritizing system libraries
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MASTER_PORT=$((RANDOM % 101 + 20000))

# 4B LoRA S4 训练产出
CHECKPOINT="/media/mldadmin/home/s125mdg38_06/StreamVLN/checkpoints/checkpoints/Qwen3-VL-4B-Instruct_LORA_S4"
# CHECKPOINT="/media/mldadmin/home/s125mdg38_06/StreamVLN/checkpoints/checkpoints/Qwen3-VL-8B-Instruct_TUNE/checkpoint-1600"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT vln/qwen3vln_eval.py --model_path $CHECKPOINT --habitat_config_path config/vln_r2r.yaml --eval_split val_unseen --output_path ./results/qwen3vln_eval_4b_lora_s4 --num_frames 1 --num_future_steps 4
