SESSION_NAME="vllm_Qwen72B_server_7279"

# 如果会话已存在，则杀死会话
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Killing it..."
    tmux kill-session -t "$SESSION_NAME"
fi

# 创建新会话
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p logs/vllm_Qwen72B_server_7279

# 启动 tmux 会话并激活 conda 环境后再运行 vllm 命令
tmux new -d -s "$SESSION_NAME" 
tmux send-keys -t "$SESSION_NAME" "mamba activate law3 && \
     CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen2.5-72B-Instruct \
         --tensor-parallel-size 4 \
         --gpu_memory_utilization 0.9 \
         --host 0.0.0.0 \
         --port 7279 \
         --disable-custom-all-reduce \
         --disable-log-requests \
         --max-model-len 25200 \
         --trust-remote-code \
         > logs/vllm_Qwen72B_server_7279/$timestamp.log 2>&1" C-m

echo "Session $SESSION_NAME created successfully."