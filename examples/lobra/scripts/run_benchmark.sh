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
SAVE_PATH=${11:-"null"}
TRAINER_CONFIG_PATH=${12:-}
EXP_NAME=${13:-}

# env
PATH="/home/pkuhetu/envs/miniconda3/envs/hetu-py/bin:${PATH}"
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python_refactor:${HETU_HOME}/build/lib:${PYTHONPATH}"

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_MEMORY_PROFILE=WARN
export HETU_MEMORY_LOG_FILE=""
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_EVENT_TIMING=FALSE

export NCCL_DEBUG=VERSION
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

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

mpirun --allow-run-as-root -mca orte_abort_on_non_zero_status 1 -np ${NUM_GPUS} \
    --hostfile ${HOST_FILE} \
    -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
    -x CUDA_HOME -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x NCCL_DEBUG \
    -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x NCCL_IB_GID_INDEX=3 \
    -x HETU_MEMORY_PROFILE -x HETU_MEMORY_LOG_FILE \
    --output-filename logs/${EXP_NAME}_llama_${MODEL_SIZE}/ds_parallel_task${TRAIN_TASK_NUM}_gpu${NUM_GPUS}_dp${DP}_tp${TP}_pp${PP} --merge-stderr-to-stdout \
    python3 scripts/benchmark.py \
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
    --lr 1e-4 \
    --dropout_prob 0 \
    --bf16 \
    --use_flash_attn \
    --sequence_parallel