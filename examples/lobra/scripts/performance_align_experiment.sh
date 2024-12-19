MODEL_SIZE=${1:-'7B'}
NUM_GPUS_LIST=${2:-'1,2,4,8,16'}
SERVER_ADDR=${3:-"${IP_1}"}
SERVER_PORT=${4:-"23333"}
HOST_FILE=${5:-'scripts/hostfile.yaml'}
ENV_FILE=${6:-'scripts/env.sh'}

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

TRAINER_CONFIG_PATH=trainer_config/example_fused.json
MEMORY_PROFILE_PATH=exp_result/memory/max_tokens_llama_${MODEL_SIZE}_1tasks.csv
SAVE_PATH=exp_result/performance_align/performance_align_llama_${MODEL_SIZE}_1tasks.csv
RAW_PATH=$SAVE_PATH.raw
GLOBAL_BATCH_SIZE=64

IFS=',' read -r -a PP_LIST <<< $(python3 -c "import math; print(','.join([str(i) for i in range(1, $NUM_LAYERS + 1) if $NUM_LAYERS % i == 0]))")
IFS=',' read -r -a NUM_GPUS_LIST <<< $NUM_GPUS_LIST

for TP in 1 2 4 8; do
    for PP in ${PP_LIST[@]}; do
        for NUM_GPUS in ${NUM_GPUS_LIST[@]}
        do
            if [ $(($TP * $PP)) -gt $NUM_GPUS ]; then
                continue
            fi
            DP=$(($NUM_GPUS / ($TP * $PP)))
            for SEQ_LEN in 2048 4096 8192 16384; do
                MICRO_BATCH_SIZE=1
                NUM_MICRO_BATCHES=$(($GLOBAL_BATCH_SIZE / ($DP * $MICRO_BATCH_SIZE)))
                bash scripts/run_benchmark.sh \
                $NUM_LAYERS $HIDDEN_SIZE $NUM_HEADS 1 \
                $SEQ_LEN $MICRO_BATCH_SIZE $NUM_MICRO_BATCHES \
                $DP $TP $PP $SERVER_ADDR $SERVER_PORT $HOST_FILE $ENV_FILE \
                $RAW_PATH $TRAINER_CONFIG_PATH performance_align
            done
        done
    done
done

# filter required keys
python3 utils/csv_filter.py \
    --input_path $RAW_PATH \
    --output_path $SAVE_PATH \
    --filter_column dp tp pp mbs seq_len num_micro_batches e2e_time