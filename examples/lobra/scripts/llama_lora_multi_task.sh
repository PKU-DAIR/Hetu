MODEL_SIZE=${1:-'7B'}
NUM_GPUS=${2:-16}
BUCKET_NUM=${3:-16}
TRAIN_TASK_NUM=${4:-6}
MAX_SEQ_LENGTH=${5:-16384}
TRAINER_CONFIG=${6:-'trainer_example'}
STRATEGY_CONFIG=${7:-''}
HOST_FILE=${8:-'scripts/hostfile'}

# env
PATH="/home/pkuhetu/envs/miniconda3/envs/hetu-py/bin:${PATH}"
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python_refactor:${HETU_HOME}/build/lib:${PYTHONPATH}"

export HETU_MEMORY_PROFILE=WARN
export HETU_MEMORY_LOG_FILE=""
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_EVENT_TIMING=OFF

export NCCL_DEBUG=VERSION
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

if [ -z $DP_BUCKET ]; then
    export DP_BUCKET=ON
fi

if [ -z $EXPR_DATA_DISPATCH ]; then
    export EXPR_DATA_DISPATCH=BALANCE
fi

if [ $DP_BUCKET = "ON" ] || [ $BUCKET_NUM = 16 ]; then
    export HETU_MAX_SPLIT_SIZE_MB=10240
    export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=0
else 
    export HETU_MAX_SPLIT_SIZE_MB=200
    export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20
fi

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

export EXPR_SENSITIVITY=OFF
export EXPR_EFFECTIVENESS=OFF
export EXPR_CUSTOM_DISTRIBUTION=OFF
export EXPR_SIMULATE=OFF

ROOT_FOLDER=data
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt
PROFILE_PATH=exp_result/profile/cost_model/profile_time_llama_${MODEL_SIZE}.csv
TRAINER_CONFIG_PATH=trainer_config/${TRAINER_CONFIG}.json
STRATEGY_CONFIG_PATH=strategy_config/${STRATEGY_CONFIG}.yaml
LOG_FILE_PATH=logs/run_multi_task_llama/ds_parallel_${NUM_GPUS}_${STRATEGY_CONFIG}_${TRAINER_CONFIG}

# generate strategy config by solve deployment plan
if [ ! -f "$STRATEGY_CONFIG_PATH" ]; then
    bash scripts/deploy_strategy_plan.sh \
    ${MODEL_SIZE} ${NUM_GPUS} ${BUCKET_NUM} \
    ${TRAIN_TASK_NUM} ${MAX_SEQ_LENGTH} \
    ${TRAINER_CONFIG} ${STRATEGY_CONFIG}
fi

mpirun --allow-run-as-root -mca orte_abort_on_non_zero_status 1 -np ${NUM_GPUS} \
    --hostfile ${HOST_FILE} \
    -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX -x NCCL_DEBUG \
    -x HETU_EVENT_TIMING -x DP_BUCKET \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL \
    -x EXPR_CUSTOM_DISTRIBUTION -x EXPR_DATA_DISPATCH -x EXPR_SENSITIVITY -x EXPR_EFFECTIVENESS -x EXPR_SIMULATE \
    -x HETU_MEMORY_PROFILE \
    -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
    --output-filename ${LOG_FILE_PATH} --merge-stderr-to-stdout \
    python3 scripts/llama_lora_multi_task.py \
    --strategy_config_path $STRATEGY_CONFIG_PATH \
    --trainer_config_path $TRAINER_CONFIG_PATH \
    --profile_path $PROFILE_PATH \
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
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --hidden_act relu \
    --dropout_prob 0.1 \
    --bf16 \
    --use_flash_attn \
    --sequence_parallel \
    --split_scheme
