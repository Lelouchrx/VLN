export CUDA_VISIBLE_DEVICES=4,5
# VLN_ROOT="${VLN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONTEXT_LENGTH=51200

BASE_MODEL="/home/cs22-hongly/VLN/checkpoints/Qwen3-VL-2B-Instruct"
ADAPTER="/home/cs22-hongly/VLN/checkpoints/Qwen3-VL-2B_lora_random_10k"

VLLM_USE_MODELSCOPE=false vllm serve "$BASE_MODEL" \
    --served-model-name qwen3vl \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules qwen3vl="$ADAPTER" \
    --max-model-len $CONTEXT_LENGTH \
    --gpu-memory-utilization 0.7 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8001
