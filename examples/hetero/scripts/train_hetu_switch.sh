# 2024.7.28: need to support cp
echo "*** this script is currently deprecated ***"
exit 1

NUM_LAYERS=${1:-32}
# HIDDEN_SIZE=${2:-4096}
HIDDEN_SIZE=${2:-512}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-256}
MICRO_BATCH_SIZE=${6:-4}
# FFN_HIDDEN_SIZE=${7:-11008}
FFN_HIDDEN_SIZE=${7:-1376}
# SERVER_ADDR="${IP_1}"
SERVER_ADDR="${IP_2}" # worker-0
# SERVER_ADDR="127.0.0.1" # 216
SERVER_PORT=${9:-"23457"}
HOST_FILE_PATH=${10:-"./scripts/host.yaml"}
ENV_FILE_PATH=${11:-"./scripts/env_A100.sh"}

# before
NUM_GPUS=8
BEFORE_DP=2
BEFORE_TP=2
BEFORE_PP=2
BEFORE_LAYERS_NUM_LIST="16,16,16,16"
BEFORE_STAGES_NUM_LIST="[2,2]"
BEFORE_MICRO_BATCH_NUM_LIST="[32,32]"
BEFORE_UNUSED_RANK="[4]"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}"
BEFORE_RECOMPUTE_LAYERS="[[15,16],[]]"

# before
NUM_GPUS=16
BEFORE_DP=4
BEFORE_TP=2
BEFORE_PP=2
BEFORE_LAYERS_NUM_LIST="16,16,16,16,16,16,16,16"
BEFORE_STAGES_NUM_LIST="[2,2,2,2]"
BEFORE_MICRO_BATCH_NUM_LIST="[16,16,16,16]"
BEFORE_UNUSED_RANK="[]"
BEFORE_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}"
BEFORE_RECOMPUTE_LAYERS="[[15,16],[],[0,1],[30,31]]"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $BEFORE_DP \
	--tp $BEFORE_TP \
	--pp $BEFORE_PP \
	--zero \
	--hetero_layers $BEFORE_LAYERS_NUM_LIST \
	--hetero_stages $BEFORE_STAGES_NUM_LIST \
	--rank_to_device_mapping $BEFORE_RANK_TO_DEVICE_MAPPING \
	--unused_rank $BEFORE_UNUSED_RANK \
	--recompute_layers $BEFORE_RECOMPUTE_LAYERS \
	--file_name "before.json"

# after
NUM_GPUS=8
AFTER_DP=2
AFTER_TP=1
AFTER_PP=4
AFTER_LAYERS_NUM_LIST="1,10,11,10,8,8,8,8"
AFTER_STAGES_NUM_LIST="[4,4]"
AFTER_MICRO_BATCH_NUM_LIST="[28,36]"
AFTER_UNUSED_RANK="[]"
AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}"
AFTER_RECOMPUTE_LAYERS="[[0,1],[30,31]]"
	
# after
NUM_GPUS=16
AFTER_DP=2
AFTER_TP=2
AFTER_PP=4
AFTER_LAYERS_NUM_LIST="1,10,11,10,8,8,8,8"
AFTER_STAGES_NUM_LIST="[4,4]"
AFTER_MICRO_BATCH_NUM_LIST="[28,36]"
AFTER_UNUSED_RANK="[1]"
AFTER_RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:14,7:15,8:8,9:9,10:10,11:11,12:12,13:13,14:6,15:7}"
AFTER_RECOMPUTE_LAYERS="[[0,1],[30,31]]"

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $AFTER_DP \
	--tp $AFTER_TP \
	--pp $AFTER_PP \
	--zero \
	--hetero_layers $AFTER_LAYERS_NUM_LIST \
	--hetero_stages $AFTER_STAGES_NUM_LIST \
	--rank_to_device_mapping $AFTER_RANK_TO_DEVICE_MAPPING \
	--unused_rank $AFTER_UNUSED_RANK \
	--recompute_layers $AFTER_RECOMPUTE_LAYERS \
	--file_name "after.json"

if [[ ${NUM_LAYERS} -eq 32 && ${HIDDEN_SIZE} -eq 4096 && ${NUM_HEADS} -eq 32 ]]; then
	MODEL_SIZE=7b
	echo use gpt 7b model...
elif [[ ${NUM_LAYERS} -eq 40 && ${HIDDEN_SIZE} -eq 5120 && ${NUM_HEADS} -eq 40 ]]; then
	MODEL_SIZE=13b
	echo use gpt 13b model...
else
	MODEL_SIZE=unknown-size
	echo use gpt unknown-size model...
fi

if [ ${SEQ_LEN} -lt 1024 ]; then
	SEQ=$SEQ_LEN
else
	SEQ=$(( ${SEQ_LEN} / 1024 ))k
fi
echo use seq_len = ${SEQ}

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/hot-switch
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

CMD="python3 -u train_hetu_switch.py \
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
--hetero_pipeline \
--hetero_data \
--before_hetero_stage_gpus ${BEFORE_TP} \
--before_hetero_stages \"${BEFORE_STAGES_NUM_LIST}\" \
--before_micro_batch_num_list \"${BEFORE_MICRO_BATCH_NUM_LIST}\" \
--before_rank_to_device_mapping \"${BEFORE_RANK_TO_DEVICE_MAPPING}\" \
--before_unused_rank \"${BEFORE_UNUSED_RANK}\" \
--after_hetero_stage_gpus ${AFTER_TP} \
--after_hetero_stages \"${AFTER_STAGES_NUM_LIST}\" \
--after_micro_batch_num_list \"${AFTER_MICRO_BATCH_NUM_LIST}\" \
--after_rank_to_device_mapping \"${AFTER_RANK_TO_DEVICE_MAPPING}\" \
--after_unused_rank \"${AFTER_UNUSED_RANK}\" \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS}"

source ${ENV_FILE_PATH}
if [ ${NUM_GPUS} -gt 8 ]; then
python3 ../../python/hetu/rpc/pssh_start.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
else
python3 ../../python/hetu/rpc/pssh_start.py \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
fi