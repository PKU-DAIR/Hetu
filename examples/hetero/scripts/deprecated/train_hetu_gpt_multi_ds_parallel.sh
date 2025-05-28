# NCCL_DEBUG=info
NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-2560}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-128}
GLOBAL_BATCH_SIZE=${5:-16}
NUM_MICRO_BATCHES=${6:-2}

HETU_INTERNAL_LOG_LEVEL=INFO mpirun --allow-run-as-root -np 8 \
--output-filename logs/ds_parallel --merge-stderr-to-stdout \
python train_hetu_gpt_multi_ds_parallel.py \
--num_strategy=4 \
--ds_parallel_config ds_parallel_config/dp2_tp2_pp2.json,ds_parallel_config/dp8.json,ds_parallel_config/dp4_tp2.json,ds_parallel_config/tp8.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--num_micro_batches $NUM_MICRO_BATCHES \
--dataset wikicorpus_en \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--seq_length $SEQ_LEN \
--epochs 20 \
--lr 1e-6 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
