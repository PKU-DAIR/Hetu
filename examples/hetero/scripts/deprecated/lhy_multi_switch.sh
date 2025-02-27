# NCCL_DEBUG=info
NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-2048}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-1024}
# todo: should pass a gbs/mbs list
GLOBAL_BATCH_SIZE=${5:-16}
NUM_MICRO_BATCHES=${6:-2}

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

export NCCL_DEBUG=VERSION
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=NVLINK
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_MAX_SPLIT_SIZE_MB=200
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20
mpirun --allow-run-as-root -np 8 \
--output-filename logs/ds_parallel --merge-stderr-to-stdout \
python3 lhy_multi_switch.py \
--num_strategy=5 \
--ds_parallel_config ds_parallel_config/dp2_tp2_pp2.json,ds_parallel_config/dp8.json,ds_parallel_config/dp4_tp2.json,ds_parallel_config/dp2_tp4.json,ds_parallel_config/tp8.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--num_micro_batches $NUM_MICRO_BATCHES \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
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

