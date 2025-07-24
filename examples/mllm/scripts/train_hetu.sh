#!/bin/bash

MODEL_SIZE='3b'
# MODEL_SIZE = '7b'
GLOBAL_BATCH_SIZE=${5:-4}
TEXT_MAX_SEQ_LEN=${6:-9216}
VISION_MAX_SEQ_LEN=${7:-8192}
IMAGE_SIZE=${8:-224}
SERVER_ADDR=${7:-"${IP_1}"} # master-0
# SERVER_ADDR=${7:-"${IP_2}"} # worker-0
# SERVER_ADDR=${7:-"127.0.0.1"} # 216
SERVER_PORT=${8:-"23456"}
HOST_FILE_PATH=${9:-"${ENV_PATH}/host_master.yaml"}
# HOST_FILE_PATH=${9:-"${ENV_PATH}/host.yaml"}
ENV_FILE_PATH=${10:-"${ENV_PATH}/env_A100.sh"}

NUM_GPUS=6
VISION_MULTI_TP_PP_LIST="[[(1, 1), (1, 1)],]"
LLM_MULTI_TP_PP_LIST="[[(2, 1),(2, 1)],]"
BATCHING_METHOD=3



# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/case${CASE}/llama${MODEL_SIZE}_gpus${NUM_GPUS}_gbs${GLOBAL_BATCH_SIZE}_msl${TEXT_MAX_SEQ_LEN}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=/home/pkuhetu/njw1123/hetu_mm/python/hetu/engine/data
JSON_FILE=${ROOT_FOLDER}/wikipedia_zea-llama_text_document
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/gpt2-vocab.json
MERGE_FILE=${ROOT_FOLDER}/gpt2-merges.txt

if [ ! -d "ds_parallel_config" ]; then
  mkdir "ds_parallel_config"
fi

# 数据配置参数
DATA_CONFIG="\
--data_path $JSON_FILE \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vision_max_seqlen $VISION_MAX_SEQ_LEN \
--text_max_seqlen $TEXT_MAX_SEQ_LEN \
--batching_method $BATCHING_METHOD"


# 模型配置参数
VISION_MODEL_CONFIG="\
--patch_size 14 \
--temporal_patch_size 2 \
--vision_embed_dim 1280 \
--vision_mlp_dim 3420 \
--vision_num_heads 1 \
--vision_num_layers 24 \
--vision_dropout 0.0 \
--in_channels 3"


if [[ ${MODEL_SIZE} == "7b" ]]; then
    # 7B
    LLM_MODEL_CONFIG="\
    --vocab_size 50304 \
    --hidden_size  3584 \
    --ffn_hidden_size  18944 \
    --num_hidden_layers  28 \
    --num_attention_heads  28\
    --hidden_act relu \
    --dropout_prob 0.0 \
    --use_flash_attn"
elif [[ ${MODEL_SIZE} == "3b" ]]; then
    # 3B
    LLM_MODEL_CONFIG="\
    --vocab_size 50304 \
    --hidden_size  2560 \
    --ffn_hidden_size  10240 \
    --num_hidden_layers  1 \
    --num_attention_heads  32 \
    --hidden_act relu \
    --dropout_prob 0.0 \
    --use_flash_attn"
else
    echo "Error: MODEL_SIZE must be either '3b' or '7b'"
    exit 1
fi

# 训练配置参数
TRAINING_CONFIG="\
--global_batch_size $GLOBAL_BATCH_SIZE \
--epochs 1 \
--steps 5 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--bf16 \
--torch_profile 1"
# 分布式运行时配置参数
DISTRIBUTED_CONFIG="\
--vision_multi_tp_pp_list \"${VISION_MULTI_TP_PP_LIST}\" \
--llm_multi_tp_pp_list \"${LLM_MULTI_TP_PP_LIST}\" \
--ngpus ${NUM_GPUS} \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT}"

# 最终拼接命令
CMD="python3 -u train_mllm.py ${DATA_CONFIG} ${VISION_MODEL_CONFIG} ${LLM_MODEL_CONFIG} ${TRAINING_CONFIG} ${DISTRIBUTED_CONFIG}"

echo CMD: $CMD

source ${ENV_FILE_PATH}
python3 ../../python/hetu/rpc/pssh_start.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
