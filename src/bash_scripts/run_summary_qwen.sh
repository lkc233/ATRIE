# 获取当前日期时间，格式为YYYYMMDD_HHMMSS
timestamp=$(date +"%Y%m%d_%H%M%S")

# 创建带有时间戳的logs文件夹
mkdir -p logs/summary_$timestamp


INPUT_FOLDER="data/docs/v1/train"
OUTPUT_FOLDER="results/summary/"
SOURCE="local_qwen"
MODEL="Qwen/Qwen2.5-72B-Instruct"
MAX_LENGTH=124000
DEMO_PATH="few-shot-demo/example.txt"

# Define an array of input_type values to iterate over
# INPUT_TYPES=("直接生成" "本院认为" "概括案情和本院认为" "理由")
INPUT_TYPES=("理由")
# INPUT_TYPES=("理由")

# Iterate over each input_type value and execute the Python script
for INPUT_TYPE in "${INPUT_TYPES[@]}"
do
    echo "Running for INPUT_TYPE: $INPUT_TYPE"
    python -u src/generate_summary/generate_summary_v2.py \
        --input_type "$INPUT_TYPE" \
        --input_folder "$INPUT_FOLDER" \
        --output_folder "$OUTPUT_FOLDER" \
        --source "$SOURCE" \
        --model "$MODEL" \
        --max_length "$MAX_LENGTH" \
        --demo_path "$DEMO_PATH" \
        > logs/summary_$timestamp/summary_$INPUT_TYPE.txt 2>&1 
done

wait
