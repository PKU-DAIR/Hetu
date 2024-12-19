MODEL_SIZE=${1:-'7B'}
SERVER_ADDR=${2:-"${IP_1}"}
SERVER_PORT=${3:-"23333"}
HOST_FILE=${4:-'scripts/hostfile.yaml'}
ENV_FILE=${5:-'scripts/env.sh'}

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

TRAINER_CONFIG_PATH=trainer_config/exp_task1.json
MEMORY_PROFILE_PATH=exp_result/memory/max_tokens_llama_${MODEL_SIZE}_1tasks.csv
SAVE_PATH=exp_result/throughput/throughput_per_gpu_llama_${MODEL_SIZE}_1tasks.csv
RAW_PATH=$SAVE_PATH.raw
SEQ_LEN_RANGE=(256 512 1024 2048 4096 8192 16384)
NUM_MICRO_BATCHES=64

# 从MEMORY_PROFILE_PATH读取，每一行形如tp,pp,sp,max_tokens
for line in `tail -n +2 ${MEMORY_PROFILE_PATH}`
do
    DP=1
    TP=`echo $line | awk -F ',' '{print $1}'`
    PP=`echo $line | awk -F ',' '{print $2}'`
    MAX_TOKENS=`echo $line | awk -F ',' '{print $3}'`
    MAX_TOKENS=${MAX_TOKENS%$'\r'}
    echo "dp: $DP, tp: $TP, pp: $PP, max_tokens: $MAX_TOKENS"
    for SEQ_LEN in ${SEQ_LEN_RANGE[@]}
    do
        if [ $SEQ_LEN -gt $MAX_TOKENS ]; then
            break
        fi
        MICRO_BATCH_SIZE=$(expr $MAX_TOKENS / $SEQ_LEN)
        bash scripts/run_benchmark.sh \
            $NUM_LAYERS $HIDDEN_SIZE $NUM_HEADS 1 \
            $SEQ_LEN $MICRO_BATCH_SIZE $NUM_MICRO_BATCHES \
            $DP $TP $PP $SERVER_ADDR $SERVER_PORT $HOST_FILE $ENV_FILE \
            $RAW_PATH $TRAINER_CONFIG_PATH throughput_experiment
    done
done

# filter required keys
python3 utils/csv_filter.py \
    --input_path $RAW_PATH \
    --output_path $SAVE_PATH \
    --filter_column dp tp pp mbs seq_len throughput_per_gpu
