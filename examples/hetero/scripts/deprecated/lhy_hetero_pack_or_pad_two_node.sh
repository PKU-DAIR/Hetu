NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-512}
# HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-256}
MICRO_BATCH_SIZE=${6:-4}
FFN_HIDDEN_SIZE=${7:-2752}
# FFN_HIDDEN_SIZE=${7:-11008}

SWITCH=0

# setting 1
DP=4
TP=2
PP=2
HETERO=true
LAYERS_NUM_LIST="16,16,16,16,16,16,16,16"
MICRO_BATCH_NUM_LIST="[16,16,16,16]"
UNUSED_RANK="[0]"
RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:4,5:5,6:6,7:7,8:0,9:1,10:10,11:11,12:12,13:13,14:14,15:15}"
# RANK_TO_DEVICE_MAPPING="{0:10,1:11,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:0,11:1,12:12,13:13,14:14,15:15}"
# RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:10,3:11,4:4,5:5,6:6,7:7,8:0,9:1,10:2,11:3,12:12,13:13,14:14,15:15}"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"


# setting 2
DP=2
TP=2
PP=4
HETERO=true
LAYERS_NUM_LIST="1,10,11,10,8,8,8,8"
MICRO_BATCH_NUM_LIST="[32,32]"
UNUSED_RANK="[]"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:14,7:15,8:8,9:9,10:10,11:11,12:12,13:13,14:6,15:7}"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:10,7:11,8:8,9:9,10:6,11:7,12:12,13:13,14:14,15:15}"
# RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:10,3:11,4:4,5:5,6:6,7:7,8:8,9:9,10:2,11:3,12:12,13:13,14:14,15:15}"
# RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:14,3:15,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:2,15:3}"
# RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:10,3:11,4:4,5:5,6:14,7:15,8:8,9:9,10:2,11:3,12:12,13:13,14:6,15:7}"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

# setting 3
DP=2
TP=4
PP=2
HETERO=true
LAYERS_NUM_LIST="2,30,20,12"
LAYERS_NUM_LIST="16,16,16,16"
STAGES_NUM_LIST="[2,2]"
MICRO_BATCH_NUM_LIST="[32,32]"
UNUSED_RANK="[]"
# RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

'''
# setting 4

DP=2
TP=2
PP=4
HETERO=true
LAYERS_NUM_LIST="8,8,8,8,8,8,8,8"
MICRO_BATCH_NUM_LIST="[32,32]"
UNUSED_RANK="[]"
# RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

DP=4
TP=2
PP=2
HETERO=true
LAYERS_NUM_LIST="16,16,16,16,16,16,16,16"
STAGES_NUM_LIST="[2,2,2,2]"
MICRO_BATCH_NUM_LIST="[16,16,16,16]"
UNUSED_RANK="[0]"
# RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"
'''

'''
# setting 5
DP=4
TP=2
PP=2
SWITCH=0
HETERO=false
UNUSED_RANK="[]"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"
'''

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

PATH="/home/pkuhetu/envs/miniconda3/envs/hetu-py/bin:${PATH}"
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib:${PYTHONPATH}"

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=TIME
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_STRAGGLER=ANALYSIS
export HETU_MEMORY_PROFILE=WARN

export HETU_MAX_SPLIT_SIZE_MB=200
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

if [ "${HETERO}" = false ]; then
    python ./ds_parallel_config/generate_gpt_3d_config.py \
        --num_layers $NUM_LAYERS \
        --num_gpus 16 \
        --dp $DP \
        --tp $TP \
        --pp $PP \
        --zero
    mpirun --allow-run-as-root -np 16 \
        -H job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0:8,job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0:8 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
        -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
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
        --rank_to_device_mapping $RANK_TO_DEVICE_MAPPING \
        --unused_rank $UNUSED_RANK
else
    python ./ds_parallel_config/generate_gpt_3d_config.py \
        --num_layers $NUM_LAYERS \
        --num_gpus 16 \
        --dp $DP \
        --tp $TP \
        --pp $PP \
        --zero
    python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
        --num_layers $NUM_LAYERS \
        --num_gpus 16 \
        --dp $DP \
        --tp $TP \
        --pp $PP \
        --zero \
        --hetero_layers $LAYERS_NUM_LIST \
        --hetero_stages $STAGES_NUM_LIST \
        --rank_to_device_mapping $RANK_TO_DEVICE_MAPPING \
        --unused_rank $UNUSED_RANK
    mpirun --allow-run-as-root -np 16 \
        -H job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0:8,job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0:8 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
        -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_pack_or_pad.py \
        --num_strategy=2 \
        --ds_parallel_config ds_parallel_config/homo/dp${DP}_tp${TP}_pp${PP}.json,ds_parallel_config/hetero/dp${DP}_tp${TP}_pp${PP}.json \
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
        --hetero_stages $STAGES_NUM_LIST \
        --hetero_pipeline \
        --hetero_data \
        --micro_batch_num_list $MICRO_BATCH_NUM_LIST \
        --rank_to_device_mapping $RANK_TO_DEVICE_MAPPING \
        --unused_rank $UNUSED_RANK
fi
