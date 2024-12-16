# 64 A800-80GB GPUs
# Qwen2-32B, 12 tasks
echo "====================================================== 32B, 64 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 32B 64 7 12 16384 exp_task12 32B_homo_fuse
echo "====================================================== 32B, 64 GPUs Task-Fused end ======================================================"
echo "====================================================== 32B, 64 GPUs Task-Sequential begin ======================================================"
echo "====================================================== split and dump individual task configs begin ======================================================"
python3 scripts/split_and_dump_task_configs.py --config_path trainer_config/exp_task12.json --split_num 12
echo "====================================================== split and dump individual task configs end ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 4096 exp_task12_0 32B_homo_seq_2
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 16384 exp_task12_1 32B_homo_seq_4
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 8192 exp_task12_2 32B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 16384 exp_task12_4 32B_homo_seq_4
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 8192 exp_task12_3 32B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 8192 exp_task12_5 32B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 8192 exp_task12_6 32B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 2048 exp_task12_8 32B_homo_seq_1
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 16384 exp_task12_9 32B_homo_seq_4
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 8192 exp_task12_7 32B_homo_seq_3
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 16384 exp_task12_10 32B_homo_seq_4
bash scripts/llama_lora_multi_task.sh 32B 64 7 1 16384 exp_task12_11 32B_homo_seq_4
echo "====================================================== 32B, 64 GPUs Task-Sequential end ======================================================"
echo "====================================================== 32B, 64 GPUs LobRA begin ======================================================"
export DP_BUCKET=ON
bash scripts/llama_lora_multi_task.sh 32B 64 16 12 16384 exp_task12 32B_hetero_fuse
echo "====================================================== 32B, 64 GPUs LobRA end ======================================================"
