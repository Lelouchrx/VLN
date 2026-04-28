#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

#R2R
CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="/home/cs22-hongly/VLN/results/navida3"

#RxR
# CONFIG_PATH="config/vln_rxr.yaml"
# SAVE_PATH="eval_log/navida_rxr" 

CHUNKS=16 # which is also the number of simulators to launch during evaluation

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.176.62.172:8001/v1"

CUDA_VISIBLE_DEVICES=5 python vln/eval_vllm.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --forward-distance 25 \
    --turn-angle 15 \
    --max-action-history 200 \
    --num-generations 1 \
    --do-sample False \
    --use-model-defaults False \
    --result-path $SAVE_PATH
    
# python src/eval/analyze_results.py \
#     --path $SAVE_PATH
