export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

source /media/mldadmin/home/s125mdg38_06/miniconda3/bin/activate streamvln
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="/media/mldadmin/home/s125mdg38_06/StreamVLN/checkpoints/checkpoints/Qwen3-VL-8B-Instruct_TUNE/checkpoint-1600"
# CHECKPOINT="/media/mldadmin/home/s125mdg38_06/StreamVLN/checkpoints/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT vln/qwen3vln_eval.py --model_path $CHECKPOINT --habitat_config_path config/vln_r2r.yaml --eval_split val_unseen --output_path ./results/qwen3vln_eval --num_history 4
