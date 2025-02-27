#!/bin/bash

# bash scripts/hot_switch_multi.sh 7b two bf16 greedy host01

MODEL_SIZE=${1:-'32b'}
SWITCH=${2:-0}
HOSTFILE=${3:-'hostfile0123'}
SEQ_LEN=${4:-4096}
GLOBAL_BATCH_SIZE=${5:-64}
MICRO_BATCH_SIZE=${6:-1}
DP=${7:-4}
TP=${7:-2}
PP=${7:-4}
HETERO=true

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
elif [ "${MODEL_SIZE}" = "30b" ]; then
    # actually 30b = 12*num_layers*hidden_size^2
    NUM_LAYERS=60
    HIDDEN_SIZE=6528 #6672
    FFN_HIDDEN_SIZE=17920
    NUM_HEADS=48 # should be divided by tp32... so 48 will cause error!!!
elif [ "${MODEL_SIZE}" = "32b" ]; then
    NUM_LAYERS=60
    HIDDEN_SIZE=6656 #6672
    # HIDDEN_SIZE=512
    # FFN_HIDDEN_SIZE=2752
    FFN_HIDDEN_SIZE=17920
    NUM_HEADS=64
elif [ "${MODEL_SIZE}" = "70b" ]; then
    NUM_LAYERS=80
    HIDDEN_SIZE=8192 #6672
    FFN_HIDDEN_SIZE=28672
    NUM_HEADS=64
else
    echo the model should be 7b/13b/30b for test.
    exit 0
fi

NNODES=$(cat ${HOSTFILE} | wc -l)
NUM_GPUS_PER_NODE=$( cat $HOSTFILE | head -n 1 | awk -F 'slots=' '{print $2}' )
WORLD_SIZE=$(( ${NNODES} * ${NUM_GPUS_PER_NODE} ))

# before
BEFORE_LAYERS_NUM_LIST="16,16,8,24"
BEFORE_MICRO_BATCH_NUM_LIST="[32,32]"
BEFORE_UNUSED_RANK="[0,1]"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:4,5:5,6:6,7:7,8:0,9:1,10:10,11:11,12:12,13:13,14:14,15:15}"
# BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

# BEFORE_LAYERS_NUM_LIST="15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15"
BEFORE_LAYERS_NUM_LIST="7,17,18,18,7,17,18,18,7,17,18,18,7,17,18,18"
BEFORE_STAGES_NUM_LIST="[4,4,4,4]"
BEFORE_MICRO_BATCH_NUM_LIST="[14,16,17,17]"
BEFORE_UNUSED_RANK="[]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:8,3:9,4:16,5:17,6:24,7:25,8:2,9:3,10:10,11:11,12:18,13:19,14:26,15:27,16:4,17:5,18:12,19:13,20:20,21:21,22:28,23:29,24:6,25:7,26:14,27:15,28:22,29:23,30:30,31:31}"

BEFORE_LAYERS_NUM_LIST="11,8,19,22,15,15,15,15,15,15,15,15,15,15,15,15"
BEFORE_STAGES_NUM_LIST="[4,4,4,4]"
BEFORE_MICRO_BATCH_NUM_LIST="[16,16,16,16]"
BEFORE_UNUSED_RANK="[3]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:7,1:0,2:8,3:9,4:16,5:17,6:24,7:25,8:1,9:2,10:10,11:11,12:18,13:19,14:26,15:27,16:3,17:4,18:12,19:13,20:20,21:21,22:28,23:29,24:5,25:6,26:14,27:15,28:22,29:23,30:30,31:31}"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:7,1:0,2:8,3:9,4:16,5:17,6:24,7:25,8:2,9:1,10:10,11:11,12:18,13:19,14:26,15:27,16:4,17:3,18:12,19:13,20:20,21:21,22:28,23:29,24:6,25:5,26:14,27:15,28:22,29:23,30:30,31:31}"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:15,1:8,2:7,3:0,4:16,5:17,6:24,7:25,8:10,9:9,10:2,11:1,12:18,13:19,14:26,15:27,16:12,17:11,18:4,19:3,20:20,21:21,22:28,23:29,24:14,25:13,26:6,27:5,28:22,29:23,30:30,31:31}"
# BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:8,3:9,4:16,5:17,6:24,7:25,8:2,9:3,10:10,11:11,12:18,13:19,14:26,15:27,16:4,17:5,18:12,19:13,20:20,21:21,22:28,23:29,24:6,25:7,26:14,27:15,28:22,29:23,30:30,31:31}"
# BEFORE_RANK_TO_DEVICE_MAPPING="{0:2,1:1,2:10,3:9,4:16,5:17,6:26,7:27,8:4,9:3,10:12,11:11,12:18,13:19,14:28,15:29,16:6,17:7,18:8,19:13,20:20,21:21,22:30,23:31,24:0,25:5,26:14,27:15,28:22,29:23,30:24,31:25}"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus $WORLD_SIZE \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $BEFORE_LAYERS_NUM_LIST \
    --hetero_stages $BEFORE_STAGES_NUM_LIST \
    --rank_to_device_mapping $BEFORE_RANK_TO_DEVICE_MAPPING \
    --unused_rank $BEFORE_UNUSED_RANK \
    --file_name "before.json"

# after
AFTER_LAYERS_NUM_LIST="20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20"
AFTER_STAGES_NUM_LIST="[4,4,4,4]"
AFTER_MICRO_BATCH_NUM_LIST="[16,16,16,16]"
AFTER_UNUSED_RANK="[]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,28:28,29:29,30:30,31:31,32:32,33:33,34:34,35:35,36:36,37:37,38:38,39:39,40:40,41:41,42:42,43:43,44:44,45:45,46:46,47:47,48:48,49:49,50:50,51:51,52:52,53:53,54:54,55:55,56:56,57:57,58:58,59:59,60:60,61:61,62:62,63:63}"

AFTER_LAYERS_NUM_LIST="12,7,19,22,7,17,18,18,7,17,18,18,7,17,18,18"
AFTER_STAGES_NUM_LIST="[4,4,4,4]"
AFTER_MICRO_BATCH_NUM_LIST="[11,18,18,17]"
AFTER_UNUSED_RANK="[]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:7,1:0,2:8,3:9,4:16,5:17,6:24,7:25,8:1,9:2,10:10,11:11,12:18,13:19,14:26,15:27,16:3,17:4,18:12,19:13,20:20,21:21,22:28,23:29,24:5,25:6,26:14,27:15,28:22,29:23,30:30,31:31}"
AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:8,3:9,4:16,5:17,6:24,7:25,8:2,9:3,10:10,11:11,12:18,13:19,14:26,15:27,16:4,17:5,18:12,19:13,20:20,21:21,22:28,23:29,24:6,25:7,26:14,27:15,28:22,29:23,30:30,31:31}"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:7,1:8,2:9,3:0,4:16,5:17,6:24,7:25,8:1,9:2,10:10,11:11,12:18,13:19,14:26,15:27,16:3,17:4,18:12,19:13,20:20,21:21,22:28,23:29,24:5,25:6,26:14,27:15,28:22,29:23,30:30,31:31}"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:15,1:8,2:7,3:0,4:16,5:17,6:24,7:25,8:10,9:9,10:2,11:1,12:18,13:19,14:26,15:27,16:12,17:11,18:4,19:3,20:20,21:21,22:28,23:29,24:14,25:13,26:6,27:5,28:22,29:23,30:30,31:31}"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:8,3:9,4:16,5:17,6:24,7:25,8:2,9:3,10:10,11:11,12:18,13:19,14:26,15:27,16:4,17:5,18:12,19:13,20:20,21:21,22:28,23:29,24:6,25:7,26:14,27:15,28:22,29:23,30:30,31:31}"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:2,1:1,2:10,3:9,4:16,5:17,6:26,7:27,8:4,9:3,10:12,11:11,12:18,13:19,14:28,15:29,16:6,17:7,18:8,19:13,20:20,21:21,22:30,23:31,24:0,25:5,26:14,27:15,28:22,29:23,30:24,31:25}"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus $WORLD_SIZE \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $AFTER_LAYERS_NUM_LIST \
    --hetero_stages $AFTER_STAGES_NUM_LIST \
    --rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
    --unused_rank $AFTER_UNUSED_RANK \
    --file_name "after.json"

python ./ds_parallel_config/generate_gpt_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus $WORLD_SIZE \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero 

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

export PATH=/jizhicfs/hymiezhao/miniconda3/envs/hetu-py/bin:$PATH
export HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../../" && pwd )"
export LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${HETU_HOME}/build/hetu/third_party/cutlass/install:${LD_LIBRARY_PATH}"
export PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib:${PYTHONPATH}"

echo HETU_HOME = $HETU_HOME

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=WARN
export HETU_STRAGGLER=ANALYSIS
export HETU_MEMORY_PROFILE=INFO

export HETU_MAX_SPLIT_SIZE_MB=0
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=0

export NCCL_DEBUG=WARN

file="straggler_exp/${MODEL_SIZE}_gpus${WORLD_SIZE}_result.txt"
echo will write result into ${file}...
dir=$(dirname "$file")
if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
fi
if [ ! -f "$file" ]; then
    touch "$file"
fi
echo -n > "$file"

if [ "${SWITCH}" = 1 ]; then
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
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_switch.py \
        --num_strategy=2 \
        --ds_parallel_config ds_parallel_config/hetero/before.json,ds_parallel_config/hetero/after.json \
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
        --switch $SWITCH \
        --hetero_stage_gpus $TP \
        --hetero_pipeline \
        --hetero_data \
        --before_hetero_stages $BEFORE_STAGES_NUM_LIST \
        --before_micro_batch_num_list $BEFORE_MICRO_BATCH_NUM_LIST \
        --before_rank_to_device_mapping $BEFORE_RANK_TO_DEVICE_MAPPING \
        --before_unused_rank $BEFORE_UNUSED_RANK \
        --after_hetero_stages $AFTER_STAGES_NUM_LIST \
        --after_micro_batch_num_list $AFTER_MICRO_BATCH_NUM_LIST \
        --after_rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
        --after_unused_rank $AFTER_UNUSED_RANK \
        --tencent
else
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
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_pack_or_pad.py \
        --num_strategy=2 \
        --ds_parallel_config ds_parallel_config/homo/dp${DP}_tp${TP}_pp${PP}.json,ds_parallel_config/hetero/after.json \
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
        --switch $SWITCH \
        --hetero_stage_gpus $TP \
        --hetero_stages $AFTER_STAGES_NUM_LIST \
        --hetero_pipeline \
        --hetero_data \
        --micro_batch_num_list $AFTER_MICRO_BATCH_NUM_LIST \
        --rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
        --unused_rank $AFTER_UNUSED_RANK \
        --tencent
fi