MODEL_PATH=${1:-""}

SERVER_ADDR="127.0.0.1"
SERVER_PORT="23462"
NUM_GPUS=1
HOST_FILE_PATH=${ENV_PATH}/host_single.yaml
ENV_FILE_PATH=${ENV_PATH}/env_A100.sh
LOG_FOLDER=logs
mkdir -p ${LOG_FOLDER}

CMD="HETU_PRE_ALLOCATE_SIZE_MB=0 python3 test_ckpt_load.py \
--num_gpus ${NUM_GPUS} \
--num_layers 32 \
--model_path ${MODEL_PATH} \
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