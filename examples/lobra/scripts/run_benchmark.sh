NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
TRAIN_TASK_NUM=${4:-1}
SEQ_LEN=${5:-1024}
MICRO_BATCH_SIZE=${6:-2}
NUM_MICRO_BATCHES=${7:-1}
DP=${8:-2}
TP=${9:-2}
PP=${10:-2}
SERVER_ADDR=${11:-"${IP_1}"}
SERVER_PORT=${12:-"23333"}
HOST_FILE=${13:-'scripts/hostfile.yaml'}
ENV_FILE=${14:-'scripts/env.sh'}
SAVE_PATH=${15:-"null"}
TRAINER_CONFIG_PATH=${16:-}
EXP_NAME=${17:-}

FFN_HIDDEN_SIZE=$(($HIDDEN_SIZE * 4))
if [ $NUM_LAYERS -eq 32 ] && [ $HIDDEN_SIZE -eq 4096 ] && [ $NUM_HEADS -eq 32 ]; then
    FFN_HIDDEN_SIZE=11008
    MODEL_SIZE=7B
elif [ $NUM_LAYERS -eq 40 ] && [ $HIDDEN_SIZE -eq 5120 ] && [ $NUM_HEADS -eq 40 ]; then
    FFN_HIDDEN_SIZE=13824
    MODEL_SIZE=13B
elif [ $NUM_LAYERS -eq 80 ] && [ $HIDDEN_SIZE -eq 8192 ] && [ $NUM_HEADS -eq 64 ]; then
    FFN_HIDDEN_SIZE=28672
    MODEL_SIZE=70B
else
    MODEL_SIZE=UNKNOWN
fi

if [ -z $TRAINER_CONFIG_PATH ]; then
    TRAINER_CONFIG_PATH=trainer_config/exp_task${TRAIN_TASK_NUM}.json
fi

NUM_GPUS=$(($DP * $TP * $PP))

WARMUP_STEPS=10
PROFILE_STEPS=0
# no profile
if [ $SAVE_PATH == "null" ]; then
    WARMUP_STEPS=15
    PROFILE_STEPS=10
fi

LOG_FILE_PATH=logs/${EXP_NAME}_llama_${MODEL_SIZE}/ds_parallel_task${TRAIN_TASK_NUM}_gpu${NUM_GPUS}_dp${DP}_tp${TP}_pp${PP}
mkdir -p ${LOG_FILE_PATH}
echo logs will save to ${LOG_FILE_PATH}...

CMD="python3 -u scripts/benchmark.py \
    --trainer_config_path $TRAINER_CONFIG_PATH \
    --save_path $SAVE_PATH \
    --vocab_size 30592 \
    --hidden_size $HIDDEN_SIZE \
    --ffn_hidden_size $FFN_HIDDEN_SIZE \
    --num_attention_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --seq_length $SEQ_LEN \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --num_micro_batches $NUM_MICRO_BATCHES \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --train_task_num $TRAIN_TASK_NUM \
    --profile_steps $PROFILE_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --server_addr $SERVER_ADDR \
    --server_port $SERVER_PORT \
    --lr 1e-4 \
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
