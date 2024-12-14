# 64 A800-80GB GPUs
# Qwen2-32B, 12 tasks
echo "====================================================== 32B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 32B 12 16384 8 8 1 7 16384 exp_task12
echo "====================================================== 32B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 32B, 64 GPUs Task-Sequential begin ======================================================"
echo "====================================================== split and dump individual task configs begin ======================================================"
python3 scripts/split_and_dump_task_configs.py --config_path trainer_config/exp_task12.json --split_num 12
echo "====================================================== split and dump individual task configs end ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 32B 1 4096 16 2 2 7 4096 exp_task12_0
bash scripts/llama_lora_multi_task.sh 32B 1 16384 8 8 1 7 16384 exp_task12_1
bash scripts/llama_lora_multi_task.sh 32B 1 8192 16 4 1 7 8192 exp_task12_2
bash scripts/llama_lora_multi_task.sh 32B 1 16384 8 8 1 7 16384 exp_task12_4
bash scripts/llama_lora_multi_task.sh 32B 1 8192 16 4 1 7 8192 exp_task12_3
bash scripts/llama_lora_multi_task.sh 32B 1 8192 16 4 1 7 8192 exp_task12_5
bash scripts/llama_lora_multi_task.sh 32B 1 8192 16 4 1 7 8192 exp_task12_6
bash scripts/llama_lora_multi_task.sh 32B 1 2048 16 2 2 7 16384 exp_task12_8
bash scripts/llama_lora_multi_task.sh 32B 1 16384 8 8 1 7 16384 exp_task12_9
bash scripts/llama_lora_multi_task.sh 32B 1 8192 16 4 1 7 8192 exp_task12_7
bash scripts/llama_lora_multi_task.sh 32B 1 16384 8 8 1 7 16384 exp_task12_10
bash scripts/llama_lora_multi_task.sh 32B 1 16384 8 8 1 7 16384 exp_task12_11
echo "====================================================== 32B, 64 GPUs Task-Sequential end ======================================================"
echo "====================================================== 32B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=TRUE
bash scripts/llama_lora_multi_task.sh 32B 12 2048,4096,8192,16384 4,4,2,1 1,2,4,8 6,2,1,2 16 16384 exp_task12
echo "====================================================== 32B, 64 GPUs LobRA end ======================================================"
