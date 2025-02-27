MODEL_SIZE=${1:-'7b'}
SEQ_LEN=${2:-4096}
if [ "${MODEL_SIZE}" = "7b" ]; then
        NUM_LAYERS=32
        HIDDEN_SIZE=4096
	FFN_HIDDEN_SIZE=11008
        NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "13b" ]; then
        NUM_LAYERS=40
        HIDDEN_SIZE=5120
	FFN_HIDDEN_SIZE=13824
        NUM_HEADS=40
elif [ "${MODEL_SIZE}" = "32b" ]; then
        # actually 30b = 12*num_layers*hidden_size^2
        NUM_LAYERS=60
        HIDDEN_SIZE=6656
	FFN_HIDDEN_SIZE=17920
        NUM_HEADS=64
else
        echo the model should be 7b/13b/30b for test.
        exit 0
fi

DP=${3:-2}
TP=${4:-4}
PP=${5:-2}
GLOBAL_BATCH_SIZE=${6:-32}
MICRO_BATCH_SIZE=${7:-2}
HOSTFILE=${8:-'hostfile_51_52'}
STEPS=${9:-11}

NNODES=$(cat ${HOSTFILE} | wc -l)
NUM_GPUS_PER_NODE=$( cat $HOSTFILE | head -n 1 | awk -F 'slots=' '{print $2}' )
WORLD_SIZE=$(( ${NNODES} * ${NUM_GPUS_PER_NODE} ))
NUM_GPUS=$(( $DP * $TP *$PP ))
if [ ${NUM_GPUS} -ne ${WORLD_SIZE} ]; then
	echo world size ${WORLD_SIZE} is not equal to dp ${DP} x tp ${TP} x pp ${PP}!
	exit 0
fi

echo use llama model ${MODEL_SIZE}, seq_len=${SEQ_LEN}, dp=${DP}, tp=${TP}, pp=${PP}, num_gpus=${NUM_GPUS}, hostfile=${HOSTFILE} 


if [ ${SEQ_LEN} -lt 1024 ]; then
	SEQ=$SEQ_LEN
else
	SEQ=$(( ${SEQ_LEN} / 1024 ))k
fi

DS_PARALLEL_CONFIG=ds_parallel_config/gpus${NUM_GPUS}/${MODEL_SIZE}/dp${DP}_tp${TP}_pp${PP}.json
if [ ! -f ${DS_PARALLEL_CONFIG} ]; then
	python3 ds_parallel_config/generate_gpt_3d_config.py --model_size ${MODEL_SIZE} --num_gpus ${NUM_GPUS} --dp ${DP} --tp ${TP} --pp ${PP} --zero
	echo generate ${DS_PARALLEL_CONFIG}...
else
	echo use ${DS_PARALLEL_CONFIG}...
fi

LOG_FOLDER=logs_llama/gpus${NUM_GPUS}_${MODEL_SIZE}_${SEQ}
mkdir -p ${LOG_FOLDER}
LOG_FILE=${LOG_FOLDER}/gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_dp${DP}_tp${TP}_pp${PP}.log
echo log will save to ${LOG_FILE}...

#ROOT_FOLDER=/data/nolan/develop/bak/ht/hot_switch/gh/Megatron-LM/data
ROOT_FOLDER=/jizhicfs/hymiezhao/hetu-gh/Hetu-dev/examples/nlp/gpt/data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

export PATH=/jizhicfs/hymiezhao/miniconda3/envs/hetu-gh2/bin:$PATH
export HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../../" && pwd )"
#export HETU_HOME=/jizhicfs/hymiezhao/hetu-gh/Hetu-dev-llama
export LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${HETU_HOME}/build/hetu/third_party/cutlass/install:${LD_LIBRARY_PATH}"
export PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib:${PYTHONPATH}"

echo use HETU_HOME = $HETU_HOME

source /jizhicfs/hymiezhao/hetu-gh/Hetu-dev/init.sh

#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_NVLS_ENABLE=0
#export NCCL_SOCKET_IFNAME=bond1
#export GLOO_SOCKET_IFNAME=bond1
#export NCCL_IB_DISABLE=0
#export NCCL_IB_HCA=mlx5
#export NCCL_NET_GDR_READ=1
#export NCCL_IB_GID_INDEX=3
#export NCCL_NET_GDR_LEVEL=2
#export NCCL_IB_QPS_PER_CONNECTION=4
#export NCCL_IB_TC=160
#export NCCL_IB_TIMEOUT=22
#export NCCL_PXN_DISABLE=0
#export NCCL_SOCKET_NTHREADS=8

#export UCX_TLS=dc_x,self,sm

export NCCL_DEBUG=VERSION
export HETU_INTERNAL_LOG_LEVEL=WARN

sleep 5
mpirun -np ${NUM_GPUS} --hostfile ${HOSTFILE} \
-v --allow-run-as-root --bind-to none --map-by slot \
--mca btl_tcp_if_include bond1 -x NCCL_SOCKET_IFNAME=bond1 \
--mca oob_tcp_if_include bond1 \
-x UCX_NET_DEVICES=bond1 -x NCCL_IB_DISABLE=0 -x NCCL_IB_GID_INDEX=3 \
-x NCCL_IB_CUDA_SUPPORT=1  -x NCCL_DEBUG=VERSION \
-x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_NVLS_ENABLE=0 -x NCCL_NET_GDR_READ=1 -x NCCL_SOCKET_NTHREADS=8 \
-x NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 \
-x NCCL_COLLNET_ENABLE=0  -x SHARP_COLL_ENABLE_SAT=0 -x NCCL_NET_GDR_LEVEL=2 -x NCCL_IB_QPS_PER_CONNECTION=4 \
-x NCCL_IB_TC=160 -x NCCL_PXN_DISABLE=0 \
-x HETU_INTERNAL_LOG_LEVEL -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
--output-filename logs/ds_parallel --merge-stderr-to-stdout \
python train_hetu_llama_ds_parallel.py \
--ds_parallel_config $DS_PARALLEL_CONFIG \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--seq_length $SEQ_LEN \
--epochs 1 \
--steps $STEPS \
--lr 1e-6 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--use_multi_node \
2>&1 | tee ${LOG_FILE}

bash kill.sh
sleep 5
