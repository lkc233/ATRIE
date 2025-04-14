#!/bin/bash

# 获取当前日期时间，格式为YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H%M%S")

# 创建带有时间戳的logs文件夹
mkdir -p logs/pred_$timestamp

# Define variables for the parameters
INPUT_DIR="data/docs/v1/test"
OUTPUT_DIR="results/pred/v1/Qwen2.5-72B-Instruct"
REF_DIR="results/summary/124000/Qwen2.5-72B-Instruct"
SOURCES="local_qwen"
MODEL="Qwen/Qwen2.5-14B-Instruct"

# Define arrays for input_type and corresponding mode
# INPUT_TYPES=("只给案情" "直接生成" "司法解释" "本院认为" "概括案情和本院认为" "参考答案" "CoT")
# MODES=("a" "b" "c" "d" "e" "gt")
# INPUT_TYPES=("只给案情" "直接生成" "司法解释" "本院认为" "概括案情和本院认为" "参考答案" "CoT")
# MODES=("a" "b" "c" "d" "e" "gt" "cot")
# INPUT_TYPES=("只给案情" "直接生成" "司法解释" "本院认为" "概括案情和本院认为" "CoT" "理由")
# MODES=("a" "b" "c" "d" "e" "cot" "f")
# INPUT_TYPES=("参考答案")
# MODES=("gt")
INPUT_TYPES=("参考答案")
MODES=("gt")

# Iterate over both arrays and execute the command
for i in "${!INPUT_TYPES[@]}"
do
    INPUT_TYPE=${INPUT_TYPES[$i]}
    MODE=${MODES[$i]}
    
    # Execute the command and redirect output to respective log file
    python -u src/judgement_pred/judgement_pred.py \
        --input_type "$INPUT_TYPE" \
        --mode "$MODE" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --ref_dir "$REF_DIR" \
        --sources "$SOURCES" \
        --model "$MODEL" \
        --response_num 1 \
        > logs/pred_$timestamp/pred_$MODE.log 2>&1
done
# Wait for all the commands to finish
wait