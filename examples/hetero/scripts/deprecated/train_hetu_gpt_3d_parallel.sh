# NCCL_DEBUG=info
NUM_LAYERS=${1:-24}
HIDDEN_SIZE=${2:-768}
NUM_HEADS=${3:-12}
SEQ_LEN=${4:-128}

mpirun --allow-run-as-root -np 8 \
--output-filename logs/3d_parallel --merge-stderr-to-stdout \
python train_hetu_gpt_3d_parallel.py \
--global_batch_size 8 \
--num_micro_batches 2 \
--dp 2 \
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
# --bf16 \