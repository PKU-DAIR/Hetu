#!/bin/bash

NUM_LAYERS=${1:-60}
HIDDEN_SIZE=${2:-6656}
NUM_HEADS=${3:-64}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-512}
MICRO_BATCH_SIZE=${6:-1}
DP=${7:-4}
TP=${8:-4}
PP=${9:-2}
HOSTFILE=${10:-'hostfile0123'}
HETERO_LAYER=${11:-2}
NUM_MICRO_BATCH=${12:-512} # should equal to GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE
FFN_HIDDEN_SIZE=${13:-17920}

NNODES=$(cat ${HOSTFILE} | wc -l)
NUM_GPUS_PER_NODE=$( cat $HOSTFILE | head -n 1 | awk -F 'slots=' '{print $2}' )
WORLD_SIZE=$(( ${NNODES} * ${NUM_GPUS_PER_NODE} ))

file="experiments/straggler/dp4tp2pp4dp${DP}tp${TP}pp${PP}/hetero_batch"

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
export HETU_INTERNAL_LOG_LEVEL=WARN
export HETU_STRAGGLER=EXP

export HETU_MAX_SPLIT_SIZE_MB=200
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20

OTHER_LAYER=$(((NUM_LAYERS - HETERO_LAYER) / (PP - 1)))

# 生成hetero脚本
python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus $WORLD_SIZE \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $HETERO_LAYER,$OTHER_LAYER,30,30,30,30,30,30 \
    --file_name "exp.json"

for i in $(seq 128 $NUM_MICRO_BATCH); do
    hetero_num_micro_batch=$((NUM_MICRO_BATCH - (DP - 1) * i))
    if [ $hetero_num_micro_batch -lt $PP ]; then
        continue
    fi
    # 运行
    echo "split num micro batch $hetero_num_micro_batch begin..."
    for j in $(seq 0 $WORLD_SIZE); do
        echo -e "\nsplit straggler num micro batch $hetero_num_micro_batch:" >> "${file}_${j}.txt"
    done
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
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_pack_or_pad.py \
            --num_strategy=2 \
            --ds_parallel_config ds_parallel_config/hetero/exp.json,ds_parallel_config/hetero/exp.json \
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
            --run_straggler_experiment \
            --hetero_stage_gpus $TP \
            --hetero_pipeline \
            --hetero_data \
            --normal_micro_batches $i \
            --straggler_file $file \
            --tencent
done
