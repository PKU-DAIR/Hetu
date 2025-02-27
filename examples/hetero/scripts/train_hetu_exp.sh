TP=${1:-2}
PP=${2:-2}
EXP_FILE=${3:-"./experiments/scale/tp2_pp2.txt"}
DP=${4:-1}

NUM_LAYERS=60
HIDDEN_SIZE=6656
NUM_HEADS=64
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
FFN_HIDDEN_SIZE=11008
# SERVER_ADDR="${IP_1}"
SERVER_ADDR="${IP_2}" # worker-0
# SERVER_ADDR="127.0.0.1" # 216
SERVER_PORT="23462"
HOST_FILE_PATH="./hostfiles/host0123.yaml"
ENV_FILE_PATH="./scripts/env_A800.sh"

NUM_GPUS=$(expr $TP \* $PP \* $DP)
CP=1
DCP=${DP}
CP_LIST="["
for ((i=1; i<=DP; i++)); do
	if [ $i -ne 1 ]; then
		CP_LIST="$CP_LIST,"
	fi
	CP_LIST="$CP_LIST$CP"
done
CP_LIST="$CP_LIST]"
RECOMPUTE_LAYERS="[]"

echo run exp: tp=${TP}, pp=${PP}, num_gpus=${NUM_GPUS} 
# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/exp_tp${TP}_pp${PP}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

python -m hetu.models.gpt.generate_gpt_4d_config.py \
	--num_layers $NUM_LAYERS \
	--num_gpus $NUM_GPUS \
	--dp $DP \
	--cp $CP \
	--tp $TP \
	--pp $PP \
	--zero \
	--recompute_layers $RECOMPUTE_LAYERS

EXP_DIR=$(dirname "$EXP_FILE")
if [ ! -d "$EXP_DIR" ]; then
  mkdir -p "$EXP_DIR"
fi
if [ ! -e "$EXP_FILE" ]; then
	> "$EXP_FILE"
fi

if (( TP == 8 && PP == 4 )); then
	START_SEQ=8448
elif (( TP == 8 && PP == 2 )); then
	START_SEQ=128
else
	START_SEQ=128
fi

for i in $(seq ${START_SEQ} 128 32768); do

content=$(<"$EXP_FILE")
length=${#content}
# 检查倒数第二个字符是否是冒号
if [[ "${content:length-2:1}" == ":" ]]; then
	echo "run exp: already OOM"
    break
fi
if [[ "${content:length-1:1}" == ":" ]]; then
	echo "run exp: already OOM"
    break
fi
echo "run exp: seq_len = ${i}"
echo "seq len = ${i}:" >> "$EXP_FILE"
SEQ_LEN=${i}
CMD="python3 -u train_hetu_exp.py \
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
--cp_list \"${CP_LIST}\" \
--exp_file ${EXP_FILE}"

source ${ENV_FILE_PATH}
if [ ${NUM_GPUS} -gt 8 ]; then
python3 ../../python/hetu/rpc/pssh_start_exp.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
else
python3 ../../python/hetu/rpc/pssh_start_exp.py \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
fi

done
