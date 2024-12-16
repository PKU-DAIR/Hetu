# 16 A100-40GB GPUs
# Llama2-7B, 6 tasks
# end to end
echo "====================================================== 7B, 16 GPUs Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 7B 16 7 6 16384 exp_task6 7B_homo_fuse
echo "====================================================== 7B, 16 GPUs Task-Fused end ======================================================"
echo "====================================================== 7B, 16 GPUs Task-Sequential begin ======================================================"
echo "====================================================== split and dump individual task configs begin ======================================================"
python3 scripts/split_and_dump_task_configs.py --config_path trainer_config/exp_task6.json --split_num 6
echo "====================================================== split and dump individual task configs end ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 7B 16 7 1 8192 exp_task6_0 7B_homo_seq_1
bash scripts/llama_lora_multi_task.sh 7B 16 7 1 16384 exp_task6_1 7B_homo_seq_2
bash scripts/llama_lora_multi_task.sh 7B 16 7 1 16384 exp_task6_2 7B_homo_seq_2
bash scripts/llama_lora_multi_task.sh 7B 16 7 1 8192 exp_task6_3 7B_homo_seq_1
bash scripts/llama_lora_multi_task.sh 7B 16 7 1 16384 exp_task6_4 7B_homo_seq_2
bash scripts/llama_lora_multi_task.sh 7B 16 7 1 16384 exp_task6_5 7B_homo_seq_2
echo "====================================================== 7B, 16 GPUs Task-Sequential end ======================================================"
echo "====================================================== 7B, 16 GPUs LobRA begin ======================================================"
export DP_BUCKET=ON
bash scripts/llama_lora_multi_task.sh 7B 16 16 6 exp_task6 7B_hetero_fuse
echo "====================================================== 7B, 16 GPUs LobRA end ======================================================"

# ablation study && case study
export EXPR_CASE_STUDY=ON
echo "====================================================== 7B, 16 GPUs Homo Task-Fused begin ======================================================"
export DP_BUCKET=OFF
bash scripts/llama_lora_multi_task.sh 7B 16 7 6 16384 exp_task6 7B_homo_fuse
echo "====================================================== 7B, 16 GPUs Homo Task-Fused end ======================================================"
echo "====================================================== 7B, 16 GPUs Len-based w/o Dynamic Bucketing begin ======================================================"
export DP_BUCKET=OFF
export EXPR_DATA_DISPATCH=GROUP
bash scripts/llama_lora_multi_task.sh 7B 16 16 6 16384 exp_task6 7B_hetero_fuse
echo "====================================================== 7B, 16 GPUs Len-based w/o Dynamic Bucketing end ======================================================"
unset EXPR_DATA_DISPATCH
echo "====================================================== 7B, 16 GPUs Balanced w/o Dynamic Bucketing begin ======================================================"
export DP_BUCKET=OFF
export EXPR_DATA_DISPATCH=BALANCE
bash scripts/llama_lora_multi_task.sh 7B 16 16 6 16384 exp_task6 7B_hetero_fuse
echo "====================================================== 7B, 16 GPUs Balanced w/o Dynamic Bucketing end ======================================================"
unset EXPR_DATA_DISPATCH
echo "====================================================== 7B, 16 GPUs Balanced w/ Dynamic Bucketing begin ======================================================"
export DP_BUCKET=ON
export EXPR_DATA_DISPATCH=BALANCE
bash scripts/llama_lora_multi_task.sh 7B 16 16 6 16384 exp_task6 7B_hetero_fuse
echo "====================================================== 7B, 16 GPUs Balanced w/ Dynamic Bucketing end ======================================================"
unset EXPR_DATA_DISPATCH
unset EXPR_CASE_STUDY

# effectiveness of planning
echo "====================================================== 7B, 16 GPUs Balanced w/ Dynamic Bucketing begin ======================================================"
export EXPR_EFFECTIVENESS=ON
export EXPR_DATA_DISPATCH=BALANCE
bash scripts/llama_lora_multi_task.sh 7B 16 16 6 16384 exp_task6 7B_hetero_fuse
export BUCKET_PLAN=PROFILE
bash scripts/deploy_strategy_plan.sh 7B 16 16 6 16384 exp_task6 7B_hetero_fuse
echo "====================================================== 7B, 16 GPUs Balanced w/ Dynamic Bucketing end ======================================================"
unset EXPR_EFFECTIVENESS
unset EXPR_DATA_DISPATCH
unset BUCKET_PLAN

# sensitivity of bucket num
export EXPR_SENSITIVITY=ON
for bucket_num in {4..32}
do
    echo "====================================================== 7B, 16 GPUs Balanced w/ Dynamic Bucketing ${bucket_num} begin ======================================================"
    export DP_BUCKET=ON
    export EXPR_DATA_DISPATCH=BALANCE
    bash scripts/llama_lora_multi_task.sh 7B 16 ${bucket_num} 6 16384 exp_task6 7B_hetero_fuse
    echo "====================================================== 7B, 16 GPUs Balanced w/ Dynamic Bucketing ${bucket_num} end ======================================================"
    unset EXPR_DATA_DISPATCH
    unset DP_BUCKET
done
unset EXPR_SENSITIVITY
