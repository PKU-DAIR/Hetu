NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-128}
GLOBAL_BATCH_SIZE=${5:-8}
MICRO_BATCH_SIZE=${6:-2}
DP=${7:-2}
TP=${8:-4}
PP=${9:-2}
SERVER_ADDR=${10:-"127.0.0.1"}
SERVER_PORT=${11:-"23457"}
NUM_GPUS=$(( $DP * $TP *$PP ))

echo dp=${DP}, tp=${TP}, pp=${PP}, num_gpus=${NUM_GPUS} 

if [[ ${NUM_LAYERS} -eq 32 && ${HIDDEN_SIZE} -eq 4096 && ${NUM_HEADS} -eq 32 ]]; then
	MODEL_SIZE=7b
	echo use gpt 7b model...
elif [[ ${NUM_LAYERS} -eq 40 && ${HIDDEN_SIZE} -eq 5120 && ${NUM_HEADS} -eq 40 ]]; then
	MODEL_SIZE=13b
	echo use gpt 13b model...
else
	echo the model should be 7b or 13b for test.
	exit 0
fi

if [ ${SEQ_LEN} -lt 1024 ]; then
	SEQ=$SEQ_LEN
else
	SEQ=$(( ${SEQ_LEN} / 1024 ))k
fi
echo use seq_len = ${SEQ}

DS_PARALLEL_CONFIG=ds_parallel_config/gpus${NUM_GPUS}/${MODEL_SIZE}/dp${DP}_tp${TP}_pp${PP}.json
if [ ! -f ${DS_PARALLEL_CONFIG} ]; then
	python3 ds_parallel_config/generate_gpt_3d_config.py --model_size ${MODEL_SIZE} --dp ${DP} --tp ${TP} --pp ${PP} --zero
	echo generate ${DS_PARALLEL_CONFIG}...
else
	echo use ${DS_PARALLEL_CONFIG}...
fi

LOG_FOLDER=logs/gpus${NUM_GPUS}_${MODEL_SIZE}_${SEQ}
mkdir -p ${LOG_FOLDER}
LOG_FILE=${LOG_FOLDER}/gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_dp${DP}_tp${TP}_pp${PP}.log
echo log will save to ${LOG_FILE}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

# ds_parallel_config can use ds_parallel_config/generate_gpt_3d_config.py for auto-generate
export NCCL_DEBUG=VERSION
export HETU_INTERNAL_LOG_LEVEL=INFO
export PATH="/home/pkuhetu/envs/miniconda3/envs/hetu-grpc/bin:${PATH}"
export HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../../" && pwd )"
export LD_LIBRARY_PATH="${HETU_HOME}/build/lib"
export PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib"

echo hetu_home=${HETU_HOME}
export HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
export LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib:${PYTHONPATH}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=en,eth,em,bond0
export GLOO_SOCKET_IFNAME=en,eth,em,bond0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_11
export NCCL_NET_GDR_READ=1
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0

CMDM="python3 train_hetu_gpt_ds_parallel.py \
--ds_parallel_config ${DS_PARALLEL_CONFIG} \
--global_batch_size ${GLOBAL_BATCH_SIZE} \
--micro_batch_size ${MICRO_BATCH_SIZE} \
--json_file ${JSON_FILE} \
--json_key ${JSON_KEY} \
--vocab_file ${VOCAB_FILE} \
--merge_file ${MERGE_FILE} \
--vocab_size 30592 \
--hidden_size ${HIDDEN_SIZE} \
--num_hidden_layers ${NUM_LAYERS} \
--num_attention_heads ${NUM_HEADS} \
--seq_length ${SEQ_LEN} \
--epochs 1 \
--steps 10 \
--lr 1e-6 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--use_multi_node \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS} \
2>&1 | tee ${LOG_FILE}"

python ./scripts/pssh_train_hetu.py --command "$CMDM" --server_port ${SERVER_PORT} --ngpus ${NUM_GPUS} --hosts ./scripts/host.yaml