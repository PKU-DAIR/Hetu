# NUM_LAYERS=${1:-32}
NUM_LAYERS=${1:-2}
HIDDEN_SIZE=${2:-4096}
HIDDEN_SIZE=${2:-256}
# HIDDEN_SIZE=${2:-16}
NUM_HEADS=${3:-32}
# NUM_HEADS=${3:-2}
SEQ_LEN=${4:-1024}
GLOBAL_BATCH_SIZE=${5:-16}
FFN_HIDDEN_SIZE=${7:-11008}
FFN_HIDDEN_SIZE=${7:-2752}
# FFN_HIDDEN_SIZE=${7:-16}
SERVER_ADDR="${IP_1}"
# SERVER_ADDR="${IP_2}" # worker-0
# SERVER_ADDR="127.0.0.1"
SERVER_PORT=${8:-"23456"}
HOST_FILE_PATH=${9:-"${ENV_PATH}/host_single.yaml"}
ENV_FILE_PATH=${10:-"${ENV_PATH}/env_A100.sh"}

CASE=3
if [[ ${CASE} -eq 0 ]]; then
	HETERO=false
	NUM_GPUS=8
	TP=2
	PP=1
	DP=2
	CP=2
	RECOMPUTE_GRANULARITY="null"
	RECOMPUTE_METHOD="null"
	RECOMPUTE_NUM_LAYERS="null"
	RECOMPUTE_LAYER_IDXS="null"
elif [[ ${CASE} -eq 1 ]]; then
	HETERO=false
	NUM_GPUS=3
	TP=1
	PP=1
	DP=1
	CP=3
	RECOMPUTE_GRANULARITY="null"
	RECOMPUTE_METHOD="null"
	RECOMPUTE_NUM_LAYERS="null"
	RECOMPUTE_LAYER_IDXS="null"
elif [[ ${CASE} -eq 2 ]]; then
	HETERO=true
	NUM_GPUS=6
	DP=1
	CP_LIST="[3]"
	TP=2
	LAYERS_NUM_LIST="[[2],[2],[2]]"
	UNUSED_RANK="[0]"
	RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5}"
	RECOMPUTE_GRANULARITY="['selective','selective','selective']"
	RECOMPUTE_METHOD="null"
	RECOMPUTE_NUM_LAYERS="null"
	RECOMPUTE_LAYER_IDXS_LIST="[[1,],[1,],[1,]]"
elif [[ ${CASE} -eq 3 ]]; then
	# 注意需要保证一个CP组中的recompute相关的设置全部一致
	HETERO=true
	NUM_GPUS=8
	DP=2
	CP_LIST="[1,3]"
	TP=2
	LAYERS_NUM_LIST="[[2],[2],[2],[2]]"
	UNUSED_RANK="[1,3]"
	RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:7,3:6,4:4,5:5,6:3,7:2}"
	RECOMPUTE_GRANULARITY="['selective','full','full','full']"
	RECOMPUTE_METHOD="null"
	RECOMPUTE_NUM_LAYERS="null"
	RECOMPUTE_LAYER_IDXS_LIST="[[0,1],[1,],[1,],[1,]]"
else
    echo unknown CASE
	exit 1
fi

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/case${CASE}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/combined_data.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

# homo
if [ "${HETERO}" = false ]; then

CP_LIST="["
for ((i=1; i<=DP; i++)); do
	if [ $i -ne 1 ]; then
		CP_LIST="$CP_LIST,"
	fi
	CP_LIST="$CP_LIST$CP"
done
CP_LIST="$CP_LIST]"

python -m hetu.models.llama.generate_llama_4d_config \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp $CP \
	--tp $TP \
	--pp $PP \
	--zero \
	--recompute_granularity $RECOMPUTE_GRANULARITY \
	--recompute_method $RECOMPUTE_METHOD \
	--recompute_num_layers $RECOMPUTE_NUM_LAYERS \
	--recompute_layer_idxs $RECOMPUTE_LAYER_IDXS

CMD="python3 -u train_hetu_packing.py \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/llama_homo/dp${DP}_cp${CP}_tp${TP}_pp${PP}.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 50304 \
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
--gpus_per_stage ${TP}"

# hetero
else

python -m hetu.models.llama.generate_llama_hetero_4d_config \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp_list $CP_LIST \
	--tp $TP \
	--zero \
	--hetero_layers $LAYERS_NUM_LIST \
	--rank_to_device_mapping $RANK_TO_DEVICE_MAPPING \
	--unused_rank $UNUSED_RANK \
	--recompute_granularity $RECOMPUTE_GRANULARITY \
	--recompute_method $RECOMPUTE_METHOD \
	--recompute_num_layers $RECOMPUTE_NUM_LAYERS \
	--recompute_layer_idxs_list $RECOMPUTE_LAYER_IDXS_LIST \
	--file_name "hetero_config.json"

CMD="python3 -u train_hetu_packing.py \
--num_strategy=1 \
--ds_parallel_config ds_parallel_config/llama_hetero/hetero_config.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 50304 \
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
--gpus_per_stage ${TP} \
--hetero_layers \"${LAYERS_NUM_LIST}\" \
--rank_to_device_mapping \"${RANK_TO_DEVICE_MAPPING}\" \
--unused_rank \"${UNUSED_RANK}\""

fi

source ${ENV_FILE_PATH}
python3 -m hetu.rpc.pssh_start \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}