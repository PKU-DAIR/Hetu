MODEL_SIZE=${1:-'7B'}
NUM_GPUS=${2:-16}
BUCKET_NUM=${3:-16}
TRAIN_TASK_NUM=${4:-6}
MAX_SEQ_LENGTH=${5:-16384}
SERVER_ADDR=${6:-"${IP_1}"}
SERVER_PORT=${7:-"23333"}
TRAINER_CONFIG=${8:-'example'}
STRATEGY_CONFIG=${9:-'strategy_example'}
HOST_FILE=${10:-'scripts/hostfile.yaml'}
ENV_FILE=${11:-'scripts/env.sh'}

if [ "${MODEL_SIZE}" = "7B" ]; then
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
	FFN_HIDDEN_SIZE=11008
    NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "32B" ]; then
    NUM_LAYERS=60
    HIDDEN_SIZE=6656
	FFN_HIDDEN_SIZE=17920
    NUM_HEADS=64
elif [ "${MODEL_SIZE}" = "70B" ]; then
    NUM_LAYERS=80
    HIDDEN_SIZE=8192
	FFN_HIDDEN_SIZE=28672
    NUM_HEADS=64
else
    echo the model should be 7b/32b/70b for test.
    exit 0
fi

ROOT_FOLDER=data
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt
PROFILE_PATH=exp_result/cost_model/profile_time_llama_${MODEL_SIZE}.csv
PROFILE_MEMORY_PATH=exp_result/memory/max_tokens_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks.csv
TRAINER_CONFIG_PATH=trainer_config/${TRAINER_CONFIG}.json
STRATEGY_CONFIG_PATH=strategy_config/${STRATEGY_CONFIG}.yaml
LOG_FILE_PATH=logs/run_multi_task_llama/ds_parallel_${NUM_GPUS}_${STRATEGY_CONFIG}_${TRAINER_CONFIG}
mkdir -p ${LOG_FILE_PATH}
echo logs will save to ${LOG_FILE_PATH}...

# generate strategy config by solve deployment plan
if [ ! -f "$STRATEGY_CONFIG_PATH" ]; then
    bash scripts/deploy_strategy_plan.sh \
    ${MODEL_SIZE} ${NUM_GPUS} ${BUCKET_NUM} \
    ${TRAIN_TASK_NUM} ${MAX_SEQ_LENGTH} \
    ${TRAINER_CONFIG} ${STRATEGY_CONFIG}
fi

CMD="python3 -u scripts/llama_lora_multi_task.py \
    --strategy_config_path $STRATEGY_CONFIG_PATH \
    --trainer_config_path $TRAINER_CONFIG_PATH \
    --profile_path $PROFILE_PATH \
    --profile_memory_path $PROFILE_MEMORY_PATH \
    --train_task_num $TRAIN_TASK_NUM \
    --ngpus $NUM_GPUS \
    --vocab_file $VOCAB_FILE \
    --merge_file $MERGE_FILE \
    --vocab_size 30592 \
    --hidden_size $HIDDEN_SIZE \
    --ffn_hidden_size $FFN_HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_attention_heads $NUM_HEADS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --min_seq_length 256 \
    --bucket_num $BUCKET_NUM \
    --server_addr $SERVER_ADDR \
    --server_port $SERVER_PORT \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --hidden_act relu \
    --dropout_prob 0.1 \
    --bf16 \
    --use_flash_attn \
    --sequence_parallel \
    --split_scheme"

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
