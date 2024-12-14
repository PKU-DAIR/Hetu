# NCCL_DEBUG=info
MODEL_SIZE=${1:-'7B'}
NUM_GPUS_LIMIT=${2:-1}
TRAIN_TASK_NUM=${3:-1}
MAX_SEQ_LENGTH=${4:-16384}
TRAINER_CONFIG=${5:-"example"}

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

TRAINER_CONFIG=trainer_config/${TRAINER_CONFIG}.json
PROFILE_MEMORY_PATH=exp_result/profile/memory/max_tokens_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks.csv

python3 scripts/profile_memory.py \
    --trainer_config_path $TRAINER_CONFIG_PATH \
    --profile_memory_path $PROFILE_MEMORY_PATH \
    --hidden_size $HIDDEN_SIZE \
    --num_attention_heads $NUM_HEADS \
    --train_task_num $TRAIN_TASK_NUM \
    --num_layers $NUM_LAYERS \
    --num_gpus_limit $NUM_GPUS_LIMIT \
    --seq_len_limit $MAX_SEQ_LENGTH \
    --train_task_num $TRAIN_TASK_NUM
