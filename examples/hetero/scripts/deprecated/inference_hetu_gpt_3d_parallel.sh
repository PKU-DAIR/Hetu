export HETU_INTERNAL_LOG_LEVEL=INFO
mpirun --allow-run-as-root -np 8 python inference_hetu_gpt_3d_parallel.py \
--global_batch_size 4 \
--num_micro_batches 2 \
--dp 2 \
--vocab_size 50257 \
--hidden_size 768 \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--seq_length 1024 \
--lr 1e-5 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1