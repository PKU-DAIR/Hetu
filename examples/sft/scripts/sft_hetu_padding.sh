CONFIG_NAME=${1:-"test_llama_7b_sft_pad_homo_tp"}

PWD="$(pwd)"
CASE=0

if [[ ${CASE} -eq 0 ]]; then
    CONFIG_NAME="test_llama_7b_sft_pad_homo_tp"
elif [[ ${CASE} -eq 1 ]]; then
    CONFIG_NAME="test_llama_7b_sft_pad_homo_cp"
elif [[ ${CASE} -eq 2 ]]; then
    CONFIG_NAME="test_llama_7b_sft_pad_hetero"
elif [[ ${CASE} -ne -1 ]]; then
    echo unknown CASE
    exit 1
fi

if [[ ${CASE} -eq 0 ]] || [[ ${CASE} -eq 1 ]]; then
    python3 -m hetu.models.llama.generate_llama_4d_config \
        --config-path=${PWD}/config \
        --config-name=${CONFIG_NAME}
elif [[ ${CASE} -eq 2 ]]; then
    python3 -m hetu.models.llama.generate_llama_hetero_4d_config \
        --config-path=${PWD}/config \
        --config-name=${CONFIG_NAME}
fi

CMD="python3 -u sft_hetu.py \
--config-path=config --config-name=${CONFIG_NAME}"

python3 -m hetu.rpc.pssh_start_config \
    --config-dir=${PWD}/config \
    --config-name=${CONFIG_NAME} \
    "rpc.command='$CMD'"