# NCCL_DEBUG=info
NUM_LAYERS=${1:-12}
HIDDEN_SIZE=${2:-768}
NUM_HEADS=${3:-12}
SEQ_LEN=${4:-1024}

export HETU_INTERNAL_LOG_LEVEL=INFO
mpirun --allow-run-as-root -np 8 python lhy_train_hetu_gpt_3d_parallel.py \
--global_batch_size 8 \
--num_micro_batches 2 \
--dp 2 \
--vocab_size 50257 \
--hidden_size $HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--seq_length $SEQ_LEN \
--lr 0.01 \
--dropout_prob 0.1