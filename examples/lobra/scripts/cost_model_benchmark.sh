MODEL_SIZE=${1:-'7B'}
TPS=${2:-2}
HOST_FILE=${3:-'scripts/hostfile'}

export NCCL_DEBUG=VERSION
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_EVENT_TIMING=TRUE

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

NUM_LAYERS=3
if [ "${MODEL_SIZE}" = "7B" ]; then
    HIDDEN_SIZE=4096
	FFN_HIDDEN_SIZE=11008
    NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "32B" ]; then
    HIDDEN_SIZE=6656
	FFN_HIDDEN_SIZE=17920
    NUM_HEADS=64
elif [ "${MODEL_SIZE}" = "70B" ]; then
    HIDDEN_SIZE=8192
	FFN_HIDDEN_SIZE=28672
    NUM_HEADS=64
else
    echo the model should be 7b/32b/70b for test.
    exit 0
fi

TRAINER_CONFIG_PATH=trainer_config/example.json
PROFILE_MEMORY_PATH=exp_result/profile/memory/max_tokens_llama_${MODEL_SIZE}_1tasks.csv
PROFILE_PATH=exp_result/profile/cost_model/profile_time_llama_${MODEL_SIZE}.csv
VALIDATION_PATH=exp_result/profile/cost_model/validation_time_llama_${MODEL_SIZE}.csv
LOG_FILE_PATH=logs/cost_model_llama_${MODEL_SIZE}/ds_parallel_${NUM_GPUS}_tp${TP}

IFS=',' read -r -a tps <<< "$TPS"

for i in $(seq 0 $(( ${#tps[@]} - 1 ))); do
    TP=${tps[$i]}
    NUM_GPUS=${tps[$i]}
    PROFILE_STEPS=100
    WARMUP_STEPS=15
    mpirun --allow-run-as-root -mca orte_abort_on_non_zero_status 1 -np ${NUM_GPUS} \
        --hostfile ${HOST_FILE} \
        -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX -x NCCL_DEBUG \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE \
        -x HETU_INTERNAL_LOG_LEVEL -x HETU_EVENT_TIMING \
        --output-filename ${LOG_FILE_PATH} --merge-stderr-to-stdout \
        python3 scripts/cost_model_benchmark.py \
        --trainer_config_path $TRAINER_CONFIG_PATH \
        --profile_path $PROFILE_PATH \
        --profile_memory_path $PROFILE_MEMORY_PATH \
        --validation_path $VALIDATION_PATH \
        --vocab_size 30592 \
        --hidden_size $HIDDEN_SIZE \
        --ffn_hidden_size $FFN_HIDDEN_SIZE \
        --num_attention_heads $NUM_HEADS \
        --tp $TP \
        --num_layers $NUM_LAYERS \
        --lr 1e-4 \
        --seq_len_range $SEQ_LEN \
        --profile_steps $PROFILE_STEPS \
        --warmup_steps $WARMUP_STEPS \
        --dropout_prob 0 \
        --bf16 \
        --use_flash_attn \
        --sequence_parallel
done