# 16 A100-40GB GPUs
# Llama2-7B, 6 tasks
echo "====================================================== 7B, 16 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 7B 6 16384 2 8 1 7 16384 exp_task6
echo "====================================================== 7B, 16 GPUs Task-Fused end ======================================================"
echo "====================================================== 7B, 16 GPUs Task-Sequential begin ======================================================"
echo "====================================================== split and dump individual task configs begin ======================================================"
python3 scripts/split_and_dump_task_configs.py --config_path trainer_config/exp_task6.json --split_num 6
echo "====================================================== split and dump individual task configs end ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 7B 1 8192 4 4 1 7 8192 exp_task6_0
bash scripts/llama_lora_multi_task.sh 7B 1 16384 2 8 1 7 16384 exp_task6_1
bash scripts/llama_lora_multi_task.sh 7B 1 16384 2 8 1 7 16384 exp_task6_2
bash scripts/llama_lora_multi_task.sh 7B 1 8192 4 4 1 7 8192 exp_task6_3
bash scripts/llama_lora_multi_task.sh 7B 1 16384 2 8 1 7 16384 exp_task6_4
bash scripts/llama_lora_multi_task.sh 7B 1 16384 2 8 1 7 16384 exp_task6_5
echo "====================================================== 7B, 16 GPUs Task-Sequential end ======================================================"
echo "====================================================== 7B, 16 GPUs LobRA begin ======================================================"
export DP_BUCKET=TRUE
bash scripts/llama_lora_multi_task.sh 7B 6 2048,4096,16384 6,1,1 1,2,8 1,1,1 16 16384 exp_task6
echo "====================================================== 7B, 16 GPUs LobRA end ======================================================"


echo "====================================================== 7B, 16 GPUs Homo Task-Fused begin ======================================================"
export DP_BUCKET=FALSE
bash scripts/llama_lora_multi_task.sh 7B 6 16384 2 8 1 7 16384 exp_task6
echo "====================================================== 7B, 16 GPUs Homo Task-Fused end ======================================================"
echo "====================================================== 7B, 16 GPUs Len-based w/o Dynamic Bucketing begin ======================================================"
export DP_BUCKET=FALSE
export HETU_DATA_DISPATCH=GROUP
unset HETU_DATA_DISPATCH
bash scripts/llama_lora_multi_task.sh 7B 6 2048,4096,16384 6,1,1 1,2,8 1,1,1 16 16384 exp_task6
echo "====================================================== 7B, 16 GPUs Len-based w/o Dynamic Bucketing end ======================================================"
echo "====================================================== 7B, 16 GPUs Balanced w/o Dynamic Bucketing begin ======================================================"
export DP_BUCKET=FALSE
export HETU_DATA_DISPATCH=BALANCE
unset HETU_DATA_DISPATCH
bash scripts/llama_lora_multi_task.sh 7B 6 2048,4096,16384 6,1,1 1,2,8 1,1,1 16 16384 exp_task6
echo "====================================================== 7B, 16 GPUs Balanced w/o Dynamic Bucketing end ======================================================"
echo "====================================================== 7B, 16 GPUs Balanced w/ Dynamic Bucketing begin ======================================================"
export DP_BUCKET=TRUE
export HETU_DATA_DISPATCH=BALANCE
unset HETU_DATA_DISPATCH
bash scripts/llama_lora_multi_task.sh 7B 6 2048,4096,16384 6,1,1 1,2,8 1,1,1 16 16384 exp_task6
echo "====================================================== 7B, 16 GPUs Balanced w/ Dynamic Bucketing end ======================================================"
