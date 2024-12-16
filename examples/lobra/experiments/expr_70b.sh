# 64 A800-80GB GPUs
# Llama2-70B, 12 tasks
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 12 16384 exp_task12 70B_homo_fuse
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Sequential begin ======================================================"
echo "====================================================== split and dump individual task configs begin ======================================================"
python3 scripts/split_and_dump_task_configs.py --config_path trainer_config/exp_task12.json --split_num 12
echo "====================================================== split and dump individual task configs end ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 4096 exp_task12_0 70B_homo_seq_2
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 16384 exp_task12_1 70B_homo_seq_4
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 8192 exp_task12_2 70B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 16384 exp_task12_3 70B_homo_seq_4
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 8192 exp_task12_4 70B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 8192 exp_task12_5 70B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 8192 exp_task12_6 70B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 2048 exp_task12_8 70B_homo_seq_1
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 16384 exp_task12_9 70B_homo_seq_4
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 8192 exp_task12_7 70B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 16384 exp_task12_10 70B_homo_seq_4
bash scripts/llama_lora_multi_task.sh 70B 64 7 1 16384 exp_task12_11 70B_homo_seq_4
echo "====================================================== 70B, 64 GPUs Task-Sequential end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=ON
bash scripts/llama_lora_multi_task.sh 70B 64 16 12 16384 exp_task12 70B_hetero_fuse
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"

echo "====================================================== 70B, 16 GPUs LobRA begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 16 7 4 16384 exp_scalability_task4 70B_hetero_gpu_scalability_1
echo "====================================================== 70B, 16 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 32 GPUs LobRA begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 32 7 4 16384 exp_scalability_task4 70B_hetero_gpu_scalability_2
echo "====================================================== 70B, 32 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 48 GPUs LobRA begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 48 7 4 16384 exp_scalability_task4 70B_hetero_gpu_scalability_3
echo "====================================================== 70B, 48 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 4 16384 exp_scalability_task4 70B_hetero_gpu_scalability_4
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 16 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 16 7 4 16384 exp_scalability_task4 70B_homo_gpu_scalability_1
echo "====================================================== 70B, 16 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 32 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 32 7 4 16384 exp_scalability_task4 70B_homo_gpu_scalability_2
echo "====================================================== 70B, 32 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 48 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 48 7 4 16384 exp_scalability_task4 70B_homo_gpu_scalability_3
echo "====================================================== 70B, 48 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 4 16384 exp_scalability_task4 70B_homo_gpu_scalability_4
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"

echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 4 16384 exp_scalability_task4 70B_hetero_task_scalability_1
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 8 16384 exp_scalability_task8 70B_hetero_task_scalability_2
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 12 16384 exp_scalability_task12 70B_hetero_task_scalability_2
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 16 16384 exp_scalability_task16 70B_hetero_task_scalability_2
echo "====================================================== 70B, 64 GPUs LobRA end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 4 16384 exp_scalability_task4 70B_homo_task_scalability
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 8 16384 exp_scalability_task8 70B_homo_task_scalability
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 12 16384 exp_scalability_task12 70B_homo_task_scalability
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 70B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 70B 64 7 16 16384 exp_scalability_task16 70B_homo_task_scalability
echo "====================================================== 70B, 64 GPUs Task-Fused end ======================================================"

