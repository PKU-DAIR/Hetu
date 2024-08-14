NUM_LAYERS=${1:-32}
# HIDDEN_SIZE=${2:-4096}
HIDDEN_SIZE=${2:-256}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-256}
MICRO_BATCH_SIZE=${6:-4}
# FFN_HIDDEN_SIZE=${7:-11008}
FFN_HIDDEN_SIZE=${7:-2752}
# SERVER_ADDR=${8:-"172.24.19.127"} # master-0
SERVER_ADDR=${8:-"172.24.71.238"} # worker-0
# SERVER_ADDR=${8:-"127.0.0.1"} # 216
SERVER_PORT=${9:-"23459"}
HOST_FILE_PATH=${10:-"./scripts/host.yaml"}
ENV_FILE_PATH=${11:-"./scripts/env_A100.sh"}

CASE=1
if [[ ${CASE} -eq 1 ]]; then
	# 单机同构
	# setting 1
	NUM_GPUS=8
	DP=1
	CP=4
	TP=2
	PP=1
	HETERO=false
	RECOMPUTE_LAYERS="[0,1]"
elif [[ ${CASE} -eq 2 ]]; then
	# 单机异构
	# setting 2
	NUM_GPUS=8
	DP=2
	CP_LIST="[1,3]"
	TP=1
	PP=2
	HETERO=true
	LAYERS_NUM_LIST="17,15,8,24,16,16,16,16"
	STAGES_NUM_LIST="[2,2,2,2]"
	MICRO_BATCH_NUM_LIST="[34,30]"
	UNUSED_RANK="[]"
	RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}"
	RECOMPUTE_LAYERS="[[],[],[30],[]]"
	SEQ_LEN_LIST="[1024, 341, 341, 342]"
elif [[ ${CASE} -eq 3 ]]; then
	# 多机同构
	# setting 3
	NUM_GPUS=16
	DP=2
	CP=2
	TP=2
	PP=2
	HETERO=false
	RECOMPUTE_LAYERS="[15,16]"
elif [[ ${CASE} -eq 4 ]]; then	
	# 多机异构
	# setting 4
	NUM_GPUS=16
	DP=2
	CP_LIST="[1,3]"
	TP=2
	PP=2
	HETERO=true
	LAYERS_NUM_LIST="16,16,12,20,16,16,16,16"
	STAGES_NUM_LIST="[2,2,2,2]"
	MICRO_BATCH_NUM_LIST="[28,36]"
	UNUSED_RANK="[4]"
	RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:14,7:15,8:8,9:9,10:10,11:11,12:12,13:13,14:6,15:7}"
	RECOMPUTE_LAYERS="[[9,10,11],[30,31],[],[]]"
	SEQ_LEN_LIST="[1024, 341, 341, 342]"
else
    echo unknown CASE
	exit 1
fi

if [ "${HETERO}" = false ]; then
	CP_LIST="["
	for ((i=1; i<=DP; i++)); do
		if [ $i -ne 1 ]; then
			CP_LIST="$CP_LIST,"
		fi
		CP_LIST="$CP_LIST$CP"
	done
	CP_LIST="$CP_LIST]"
fi

trimmed_list=${CP_LIST:1:-1}
IFS=',' read -r -a array <<< "$trimmed_list"
DCP=0
for element in "${array[@]}"; do
  DCP=$((DCP + element))
done

echo dcp=${DCP}, tp=${TP}, pp=${PP}, num_gpus=${NUM_GPUS} 

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
LOG_FOLDER=logs/gpus${NUM_GPUS}_${MODEL_SIZE}_seq${SEQ}_gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_dcp${DCP}_tp${TP}_pp${PP}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

if [ "${HETERO}" = false ]; then

python ./ds_parallel_config/generate_gpt_3d_config.py \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp $CP \
	--tp $TP \
	--pp $PP \
	--zero \
	--recompute_layers $RECOMPUTE_LAYERS

CMD="python3 -u train_hetu.py \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/homo/dcp${DCP}_tp${TP}_pp${PP}.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 4 \
--steps 40 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS} \
--cp_list \"${CP_LIST}\""

fi

if [ "${HETERO}" = true ]; then

python ./ds_parallel_config/generate_gpt_hetero_3d_config.py \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp_list $CP_LIST \
	--tp $TP \
	--pp $PP \
	--zero \
	--hetero_layers $LAYERS_NUM_LIST \
	--hetero_stages $STAGES_NUM_LIST \
	--rank_to_device_mapping $RANK_TO_DEVICE_MAPPING \
	--unused_rank $UNUSED_RANK \
	--recompute_layers $RECOMPUTE_LAYERS

CMD="python3 -u train_hetu.py \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/hetero/dcp${DCP}_tp${TP}_pp${PP}.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 4 \
--steps 40 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS} \
--cp_list \"${CP_LIST}\" \
--hetero \
--seq_len_list \"${SEQ_LEN_LIST}\" \
--hetero_stage_gpus ${TP} \
--hetero_stages \"${STAGES_NUM_LIST}\" \
--micro_batch_num_list \"${MICRO_BATCH_NUM_LIST}\" \
--rank_to_device_mapping \"${RANK_TO_DEVICE_MAPPING}\" \
--unused_rank \"${UNUSED_RANK}\""

fi

source ${ENV_FILE_PATH}
if [ ${NUM_GPUS} -gt 8 ]; then
python3 ../../../python_refactor/hetu/rpc/pssh_start.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
else
python3 ../../../python_refactor/hetu/rpc/pssh_start.py \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
fi