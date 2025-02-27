NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-64}
MICRO_BATCH_SIZE=${6:-1}
FFN_HIDDEN_SIZE=${7:-11008}

SWITCH=1

DP=2
TP=2
PP=4
HETERO=true

# before
BEFORE_LAYERS_NUM_LIST="4,4,4,9,11,8,8,8,8"
BEFORE_STAGES_NUM_LIST="[5,4]"
BEFORE_MICRO_BATCH_NUM_LIST="[30,34]"
BEFORE_UNUSED_RANK="[1,3]"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:8,1:16,2:2,3:17,4:4,5:5,6:6,7:7,8:0,9:1,10:10,11:11,12:12,13:13,14:14,15:15,16:9,17:3}"
# BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus 16 \
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
AFTER_LAYERS_NUM_LIST="8,8,8,8,8,8,8,8"
AFTER_STAGES_NUM_LIST="[4,4]"
AFTER_MICRO_BATCH_NUM_LIST="[32,32]"
AFTER_UNUSED_RANK="[]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
AFTER_RANK_TO_DEVICE_MAPPING="{0:3,1:2,2:1,3:0,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"

'''
AFTER_LAYERS_NUM_LIST="32,0,20,12"
AFTER_STAGES_NUM_LIST="[2,2]"
AFTER_MICRO_BATCH_NUM_LIST="[20,44]"
AFTER_UNUSED_RANK="[0,1,6,7,8,9,10,11]"
# AFTER_RANK_TO_DEVICE_MAPPING="{0:8,1:9,2:2,3:3,4:10,5:11,6:6,7:7,8:0,9:1,10:4,11:5,12:12,13:13,14:14,15:15}"
AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"
'''

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
    --num_layers $NUM_LAYERS \
    --num_gpus 16 \
    --dp $DP \
    --tp $TP \
    --pp $PP \
    --zero \
    --hetero_layers $AFTER_LAYERS_NUM_LIST \
    --hetero_stages $AFTER_STAGES_NUM_LIST \
    --rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
    --unused_rank $AFTER_UNUSED_RANK \
    --file_name "after.json"

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
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=WARN
export HETU_STRAGGLER=ANALYSIS
export HETU_MEMORY_PROFILE=WARN

export HETU_MAX_SPLIT_SIZE_MB=200
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

if [ "${SWITCH}" = 1 ]; then
    mpirun --allow-run-as-root -np 16 \
        -H job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0:8,job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0:8 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
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
        --after_unused_rank $AFTER_UNUSED_RANK
else
    mpirun --allow-run-as-root -np 16 \
        -H job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0:8,job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0:8 \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
        -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
        -x HETU_MAX_SPLIT_SIZE_MB -x HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB \
        -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL -x HETU_STRAGGLER -x HETU_MEMORY_PROFILE \
        --output-filename logs/ds_parallel --merge-stderr-to-stdout \
        python lhy_hetero_pack_or_pad.py \
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
        --micro_batch_num_list $AFTER_MICRO_BATCH_NUM_LIST \
        --rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
        --unused_rank $AFTER_UNUSED_RANK
fi
