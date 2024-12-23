MODEL_SIZE=${1:-'7B'}
NUM_GPUS=${2:-16}
BUCKET_NUM=${3:-16}
TRAIN_TASK_NUM=${4:-6}
MAX_SEQ_LENGTH=${5:-16384}
TRAINER_CONFIG=${6:-'example'}
STRATEGY_CONFIG=${7:-'strategy_example'}

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

export EXPR_CUSTOM_DISTRIBUTION=OFF
export EXPR_EFFECTIVENESS=OFF

if [ -z ${EXPR_SCHEME_PROPOSAL} ]; then
    export EXPR_SCHEME_PROPOSAL=ON
fi

if [ -z ${EXPR_DEPLOY_PATTERN} ]; then
    export EXPR_DEPLOY_PATTERN=PRUNE
fi

if [ -z ${BUCKET_PLAN} ]; then
    export BUCKET_PLAN=DYNAMIC
fi

ROOT_FOLDER=data
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt
TRAINER_CONFIG_PATH=trainer_config/${TRAINER_CONFIG}.json
STRATEGY_CONFIG_PATH=strategy_config/${STRATEGY_CONFIG}.yaml
PROFILE_PATH=exp_result/cost_model/profile_time_llama_${MODEL_SIZE}.csv
MEMORY_PROFILE_PATH=exp_result/memory/max_tokens_llama_${MODEL_SIZE}_${TRAIN_TASK_NUM}tasks.csv

python3 scripts/deploy_strategy_plan.py \
--trainer_config_path $TRAINER_CONFIG_PATH \
--profile_path $PROFILE_PATH \
--profile_memory_path $MEMORY_PROFILE_PATH \
--strategy_config_path $STRATEGY_CONFIG_PATH \
--train_task_num $TRAIN_TASK_NUM \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--max_seq_length $MAX_SEQ_LENGTH \
--min_seq_length 256 \
--ngpus $NUM_GPUS \
--bucket_num $BUCKET_NUM \
--dropout_prob 0 \
--use_flash_attn \
--sequence_parallel

