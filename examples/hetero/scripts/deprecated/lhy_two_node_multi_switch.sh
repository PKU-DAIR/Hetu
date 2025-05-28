# NCCL_DEBUG=info
NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-512}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-128}
GLOBAL_BATCH_SIZE=${5:-16}
NUM_MICRO_BATCHES=${6:-4}

PATH="/home/pkuhetu/envs/miniconda3/envs/hetu/bin:${PATH}"
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib:${PYTHONPATH}"

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=INFO
export HETU_INTERNAL_LOG_LEVEL=INFO

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3

mpirun --allow-run-as-root -np 16 \
-H job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0:8,job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0:8 \
-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
-x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
-x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL \
--output-filename logs/ds_parallel --merge-stderr-to-stdout \
python lhy_two_node_multi_switch.py \
--num_strategy=8 \
--ds_parallel_config ds_parallel_config/two_node/dp2_tp2_pp4.json,ds_parallel_config/two_node/dp4_tp2_pp2.json,ds_parallel_config/two_node/dp2_tp4_pp2.json,ds_parallel_config/two_node/dp16.json,ds_parallel_config/two_node/dp8_tp2.json,ds_parallel_config/two_node/dp4_tp4.json,ds_parallel_config/two_node/dp2_tp8.json,ds_parallel_config/two_node/tp16.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--num_micro_batches $NUM_MICRO_BATCHES \
--dataset wikicorpus_en \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--seq_length $SEQ_LEN \
--epochs 20 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--use_two_node