LAUNCHER="python3"

TRAINER="train.py"

${LAUNCHER} ${TRAINER} \
--gpu_id 0 \
--global_train_batch_size 1 \
--model_size gpt-0.3b \
--set_model_config_manually 0 \
--set_layernum_manually 0 \
--vocab_size 50257 \
--hidden_size 1024 \
--num_hidden_layers 24 \
--num_attention_heads 16 \
--seq_length 1024 \
--epochs 10 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--dropout_prob 0.1 \
--check_loss 0 \
--profile 1