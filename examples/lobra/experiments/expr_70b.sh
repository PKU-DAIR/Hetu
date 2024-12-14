# 64 A800-80GB GPUs
# Llama2-70B, 12 tasks
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 12 16384 4 16 1 7 16384 exp_task12
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Sequential begin ======================================================"
echo "====================================================== split and dump individual task configs begin ======================================================"
python3 scripts/split_and_dump_task_configs.py --config_path trainer_config/exp_task12.json --split_num 12
echo "====================================================== split and dump individual task configs end ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 1 4096 8 4 2 7 4096 exp_task12_0
bash scripts/llama_lora_multi_task.sh 70B 1 16384 4 16 1 7 16384 exp_task12_1
bash scripts/llama_lora_multi_task.sh 70B 1 8192 8 8 1 7 8192 exp_task12_2
bash scripts/llama_lora_multi_task.sh 70B 1 16384 4 16 1 7 8192 exp_task12_3
bash scripts/llama_lora_multi_task.sh 70B 1 8192 8 8 1 7 16384 exp_task12_4
bash scripts/llama_lora_multi_task.sh 70B 1 8192 8 8 1 7 8192 exp_task12_5
bash scripts/llama_lora_multi_task.sh 70B 1 8192 8 8 1 7 8192 exp_task12_6
bash scripts/llama_lora_multi_task.sh 70B 1 2048 8 4 2 7 16384 exp_task12_8
bash scripts/llama_lora_multi_task.sh 70B 1 16384 4 16 1 7 16384 exp_task12_9
bash scripts/llama_lora_multi_task.sh 70B 1 8192 8 8 1 7 8192 exp_task12_7
bash scripts/llama_lora_multi_task.sh 70B 1 16384 4 16 1 7 16384 exp_task12_10
bash scripts/llama_lora_multi_task.sh 70B 1 16384 4 16 1 7 16384 exp_task12_11
echo "====================================================== 70B, 64 GPUs Task-Sequential end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=TRUE
bash scripts/llama_lora_multi_task.sh 70B 12 2048,4096,8192,16384 4,1,1,1 2,4,8,16 4,2,1,1 16 16384 exp_task12
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"

echo "====================================================== 70B, 16 GPUs LobRA begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 16384 1 16 1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 16 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 32 GPUs LobRA begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 2048,4096,16384 1,1,1 2,4,16 4,2,1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 32 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 48 GPUs LobRA begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 2048,4096,8192,16384 2,1,1,1 2,4,8,16 4,2,1,1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 48 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 2048,4096,8192,16384 3,1,2,1 2,4,8,16 4,2,1,1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 16 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 16384 1 16 1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 16 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 32 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 16384 2 16 1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 32 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 48 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 16384 3 16 1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 48 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 16384 4 16 1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"

echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 2048,4096,8192,16384 3,1,2,1 2,4,8,16 4,2,1,1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 8 2048,4096,8192,16384 3,2,1,1 2,4,8,16 4,2,1,1 7 16384 exp_scalability_task8
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 12 2048,4096,8192,16384 3,2,1,1 2,4,8,16 4,2,1,1 7 16384 exp_scalability_task12
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 16 2048,4096,8192,16384 3,2,1,1 2,4,8,16 4,2,1,1 7 16384 exp_scalability_task16
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 4 16384 4 16 1 7 16384 exp_scalability_task4
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 8 16384 4 16 1 7 16384 exp_scalability_task8
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 12 16384 4 16 1 7 16384 exp_scalability_task12
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 70B 16 16384 4 16 1 7 16384 exp_scalability_task16
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"

