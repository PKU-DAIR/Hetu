#!/bin/bash
MAX_NUM_LAYERS=${1:-80}
HIDDEN_SIZE=${2:-8192}
NUM_HEADS=${3:-64}
SEQ_LEN=${4:-4096}
GLOBAL_BATCH_SIZE=${5:-64}
MICRO_BATCH_SIZE=${6:-1}
DP=${7:-2}
TP=${8:-8}
PP=${9:-4}
HOSTFILE=${10:-'hostfile01234567'}
FFN_HIDDEN_SIZE=${11:-49152}

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

export PATH=/jizhicfs/hymiezhao/miniconda3/envs/hetu-py/bin:$PATH
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib:${PYTHONPATH}"

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_MEMORY_PROFILE=MICRO_BATCH

export HETU_MAX_SPLIT_SIZE_MB=0
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=0

num=$PP

NNODES=$(cat ${HOSTFILE} | wc -l)
NUM_GPUS_PER_NODE=$( cat $HOSTFILE | head -n 1 | awk -F 'slots=' '{print $2}' )
WORLD_SIZE=$(( ${NNODES} * ${NUM_GPUS_PER_NODE} ))

while [ $num -le $MAX_NUM_LAYERS ]; do

    echo "current layer number: $num"
    NUM_LAYERS=$num
    file="experiments/memory/dp${DP}tp${TP}pp${PP}/${NUM_LAYERS}/homo_memory"
    
    for i in $(seq 0 $WORLD_SIZE); do
        dir=$(dirname "${file}_${i}.txt")
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
        fi
        if [ ! -f "${file}_${i}.txt" ]; then
            touch "${file}_${i}.txt"
        fi
        echo -n > "${file}_${i}.txt"
    done

    # 生成hetero脚本
    python ./ds_parallel_config/generate_gpt_3d_config.py \
        --num_layers $NUM_LAYERS \
        --num_gpus $WORLD_SIZE \
        --dp $DP \
        --tp $TP \
        --pp $PP \
        --zero

    # 运行
    mpirun --allow-run-as-root -np ${WORLD_SIZE} --hostfile ${HOSTFILE} \
        --bind-to none --map-by slot \
        --mca btl_tcp_if_include bond1 -x NCCL_SOCKET_IFNAME=bond1 \
        --mca oob_tcp_if_include bond1 \
        -x UCX_NET_DEVICES=bond1 -x NCCL_IB_DISABLE=0 -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_CUDA_SUPPORT=1  -x NCCL_DEBUG=VERSION \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 -x NCCL_NVLS_ENABLE=0 -x NCCL_NET_GDR_READ=1 -x NCCL_SOCKET_NTHREADS=8 \
        -x NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 \
        -x NCCL_COLLNET_ENABLE=0  -x SHARP_COLL_ENABLE_SAT=0 -x NCCL_NET_GDR_LEVEL=2 -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_IB_TC=160 -x NCCL_PXN_DISABLE=0 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
    python lhy_hetero_pack_or_pad.py \
    --num_strategy=1 \
    --ds_parallel_config ds_parallel_config/homo/dp${DP}_tp${TP}_pp${PP}.json \
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
    --epochs 4 \
    --steps 40 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --hidden_act relu \
    --dropout_prob 0.1 \
    --bf16 \
    --use_flash_attn \
    --use_two_node \
    --hetero_stage_gpus $TP \
    --run_memory_experiment \
    --memory_file $file \
    --tencent

    num=$((num + PP))
done