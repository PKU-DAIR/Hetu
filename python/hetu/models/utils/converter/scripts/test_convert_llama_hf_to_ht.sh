SERVER_ADDR="127.0.0.1"
SERVER_PORT="23338"
NUM_GPUS=1
HOST_FILE_PATH=${ENV_PATH}/host_single.yaml
ENV_FILE_PATH=${ENV_PATH}/env_A100.sh
SRC_MODEL_PATH=/path/to/hf/llama
DST_MODEL_PATH=/path/to/ht/llama
LOG_FOLDER=logs
mkdir -p ${LOG_FOLDER}

CMD="python3 convert_llama_hf_to_ht.py \
--input_name_or_path ${SRC_MODEL_PATH} \
--output_path ${DST_MODEL_PATH} \
--precision bfloat16 \
--sharded_store \
--test_loading \
--num_gpus ${NUM_GPUS} \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT}"

source ${ENV_FILE_PATH}
python3 -m hetu.rpc.pssh_start \
--hosts ${HOST_FILE_PATH} \
--command "$CMD" \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS} \
--envs ${ENV_FILE_PATH} \
--log_path ${LOG_FOLDER}