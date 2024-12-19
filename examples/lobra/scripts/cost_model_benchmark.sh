MODEL_SIZE=${1:-'7B'}
TPS=${2:-2}
SERVER_ADDR=${3:-"${IP_1}"}
SERVER_PORT=${4:-"23333"}
HOST_FILE=${5:-'scripts/hostfile'}
ENV_FILE=${6:-'scripts/env.sh'}

# export HETU_EVENT_TIMING=TRUE # NOTE

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
PROFILE_MEMORY_PATH=exp_result/memory/max_tokens_llama_${MODEL_SIZE}_1tasks.csv
PROFILE_PATH=exp_result/cost_model/profile_time_llama_${MODEL_SIZE}.csv
VALIDATION_PATH=exp_result/cost_model/validation_time_llama_${MODEL_SIZE}.csv
LOG_FILE_PATH=logs/cost_model_llama_${MODEL_SIZE}/ds_parallel_${NUM_GPUS}_tp${TP}
mkdir -p ${LOG_FILE_PATH}
echo logs will save to ${LOG_FILE_PATH}...

IFS=',' read -r -a tps <<< "$TPS"

for i in $(seq 0 $(( ${#tps[@]} - 1 ))); do
    TP=${tps[$i]}
    NUM_GPUS=${tps[$i]}
    PROFILE_STEPS=100
    WARMUP_STEPS=15
    CMD="python3 -u scripts/cost_model_benchmark.py \
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
        --server_addr $SERVER_ADDR \
        --server_port $SERVER_PORT \
        --dropout_prob 0 \
        --bf16 \
        --use_flash_attn \
        --sequence_parallel"
    source ${ENV_FILE}
    if [ ${NUM_GPUS} -gt 8 ]; then
    python3 ../../python/hetu/rpc/pssh_start.py \
        --hosts ${HOST_FILE} \
        --command "$CMD" \
        --server_port ${SERVER_PORT} \
        --ngpus ${NUM_GPUS} \
        --envs ${ENV_FILE} \
        --log_path ${LOG_FILE_PATH}
    else
    python3 ../../python/hetu/rpc/pssh_start.py \
        --command "$CMD" \
        --server_port ${SERVER_PORT} \
        --ngpus ${NUM_GPUS} \
        --envs ${ENV_FILE} \
        --log_path ${LOG_FILE_PATH}
    fi
done