NUM_LAYERS=${1:-32}
# HIDDEN_SIZE=${2:-4096}
HIDDEN_SIZE=${2:-256}
# FFN_HIDDEN_SIZE=${3:-11008}
FFN_HIDDEN_SIZE=${3:-2752}
NUM_HEADS=${4:-32}
GLOBAL_BATCH_SIZE=${5:-128}
MAX_SEQ_LEN=${6:-8}
IMAGE_SIZE=${7:-224}
SERVER_ADDR=${7:-"127.0.0.1"} # 216
SERVER_PORT=${8:-"23333"}
HOST_FILE_PATH=${9:-"./scripts/host.yaml"}
ENV_FILE_PATH=${10:-"./scripts/env_4090.sh"}

CASE=1
if [[ ${CASE} -eq 1 ]]; then
	# homo + greedy packing with static shape
	NUM_GPUS=1
	MULTI_TP_PP_LIST="[[(1, 1)], ]"
	BATCHING_METHOD=0
elif [[ ${CASE} -eq 2 ]]; then	
    # homo + greedy packing with dynamic shape
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], ]"
	BATCHING_METHOD=3
elif [[ ${CASE} -eq 3 ]]; then	
    # homo + hydraulis packing
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], ]"
	BATCHING_METHOD=4
elif [[ ${CASE} -eq 4 ]]; then	
    # hetero + hydraulis packing
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (1, 8)], ]"
	BATCHING_METHOD=4
elif [[ ${CASE} -eq 5 ]]; then	
    # hetero + hydraulis packing
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (1, 4), (1, 4)], ]"
	BATCHING_METHOD=4
else
    echo unknown CASE
	exit 1
fi

echo num_gpus=${NUM_GPUS}, global_batch_size = ${GLOBAL_BATCH_SIZE}, max_seq_len = ${MAX_SEQ_LEN}

if [[ ${NUM_LAYERS} -eq 32 && ${HIDDEN_SIZE} -eq 4096 && ${NUM_HEADS} -eq 32 ]]; then
	MODEL_SIZE=7b
	echo use llama 7b model...
elif [[ ${NUM_LAYERS} -eq 40 && ${HIDDEN_SIZE} -eq 5120 && ${NUM_HEADS} -eq 40 ]]; then
	MODEL_SIZE=13b
	echo use llama 13b model...
else
	MODEL_SIZE=-unknown-size
	echo use llama unknown-size model...
fi

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/case${CASE}/llama${MODEL_SIZE}_gpus${NUM_GPUS}_gbs${GLOBAL_BATCH_SIZE}_msl${MAX_SEQ_LEN}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=/home/gehao/njw1123/precision_alignment/python_refactor/elastic/engine/data
JSON_FILE=${ROOT_FOLDER}/wikipedia_zea-llama_text_document
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

if [ ! -d "ds_parallel_config" ]; then
  mkdir "ds_parallel_config"
fi

# 数据配置参数
DATA_CONFIG="\
--data_path $JSON_FILE \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--max_seq_len $MAX_SEQ_LEN \
--batching_method $BATCHING_METHOD \
--patch_size 2 \
--in_channels 3"


LLM_MODEL_CONFIG="\
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--hidden_act relu \
--dropout_prob 0.1 \
--use_flash_attn"


# 训练配置参数
TRAINING_CONFIG="\
--global_batch_size $GLOBAL_BATCH_SIZE \
--epochs 4 \
--steps 1 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--bf16"

# 分布式运行时配置参数
DISTRIBUTED_CONFIG="\
--multi_tp_pp_list \"${MULTI_TP_PP_LIST}\" \
--ngpus ${NUM_GPUS} \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT}"

# 最终拼接命令
CMD="python3 -u train_mmdit.py ${DATA_CONFIG} ${LLM_MODEL_CONFIG} ${TRAINING_CONFIG} ${DISTRIBUTED_CONFIG}"

echo CMD: $CMD
echo 

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