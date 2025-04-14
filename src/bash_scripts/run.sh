set -x
python src/data_preparation/mongodb_data_filter.py
# 30min
bash src/bash_scripts/start_qwen_7b_servers.sh
sleep 2m
python src/data_preparation/llm_data_filter.py
# 10h
bash src/bash_scripts/stop_qwen_7b_servers.sh
sleep 15
bash src/bash_scripts/start_qwen_72b_servers.sh
sleep 3m
python src/data_preparation/llm_data_filter.py --model Qwen/Qwen2.5-72B-Instruct --input_folder data/docs/v0/Qwen2.5-7B-Instruct_filtered
# 20h
python src/data_preparation/data_split.py
python src/generate_interpretation/extract_reason.py --input_dir data/docs/v1/test --output_dir data/docs/v1/test/reason
python src/generate_interpretation/extract_reason.py --input_dir data/docs/v1/train --output_dir data/docs/v1/train/reason
bash src/bash_scripts/stop_qwen_72b_servers.sh
sleep 15
bash src/bash_scripts/vllm_Qwen72B_server_7280_long.sh
sleep 3m
python src/generate_interpretation/generate_interpretation.py
tmux kill-session -t vllm_Qwen72B_server_7280_long
sleep 15
bash src/bash_scripts/start_qwen_72b_servers.sh
sleep 3m
python src/judgement_pred/judgement_pred.py --response_num 3
bash src/bash_scripts/stop_qwen_72b_servers.sh
sleep 15
bash src/bash_scripts/start_qwen_14b_servers.sh
sleep 2m
python src/judgement_pred/judgement_pred.py --input_type 理由 --model Qwen/Qwen2.5-14B-Instruct
python src/judgement_pred/judgement_pred.py --input_type 直接生成 --model Qwen/Qwen2.5-14B-Instruct
bash src/bash_scripts/stop_qwen_14b_servers.sh
sleep 15
bash src/bash_scripts/start_qwen_72b_servers.sh
sleep 3m
python src/evaluate/evaluate.py
python src/evaluate/evaluate.py --output_dir results/pred/v1/Qwen2.5-72B-Instruct/Qwen2.5-14B-Instruct/直接生成/evaluation --input_type 直接生成
bash src/bash_scripts/stop_qwen_72b_servers.sh
