#!/bin/bash

# 主要更改这三个属性
TWO_NODE=true
USE_BF16=true
export HETU_SWITCH_ALGORITHM=NEW_GREEDY # 可更改成FCFS或MULTI_NODE_ROUND_ROBIN作为baseline

NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-128}
GLOBAL_BATCH_SIZE=${5:-16}
NUM_MICRO_BATCHES=${6:-2}

PATH="/home/pkuhetu/envs/miniconda3/envs/hetu-py/bin:${PATH}"
HETU_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HETU_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HETU_HOME}/python:${HETU_HOME}/build/lib:${PYTHONPATH}"

file="experiments/hetero/result.txt"
dir=$(dirname "$file")
if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
fi
if [ ! -f "$file" ]; then
    touch "$file"
fi
echo -n > "$file"

export HETU_SWITCH_PROFILE=TIME
export HETU_INTERNAL_LOG_LEVEL=WARN

if [ "${TWO_NODE}" = true ]; then
    for i in $(seq 0 6); do
        for j in $(seq 0 6); do
            if [ $i -eq $j ]; then
                continue 
            fi
            echo "switch from $i to $j begin..."
            echo -e "\nswitch from $i to $j: " >> "$file"
            if [ "${USE_BF16}" = true ]; then
                mpirun --allow-run-as-root -np 16 \
                -H job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0:8,job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0:8 \
                -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
                -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL \
                --output-filename logs/ds_parallel --merge-stderr-to-stdout \
                python lhy_two_node_multi_switch.py \
                --num_strategy=7 \
                --ds_parallel_config ds_parallel_config/two_node/dp4_tp2_pp2.json,ds_parallel_config/two_node/dp2_tp4_pp2.json,ds_parallel_config/two_node/dp16.json,ds_parallel_config/two_node/dp8_tp2.json,ds_parallel_config/two_node/dp4_tp4.json,ds_parallel_config/two_node/dp2_tp8.json,ds_parallel_config/two_node/tp16.json \
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
                --use_two_node \
                --from_strategy "$i" \
                --to_strategy "$j" \
                --switch_file "$file"
            else
                mpirun --allow-run-as-root -np 16 \
                -H job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0:8,job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0:8 \
                -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
                -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL \
                --output-filename logs/ds_parallel --merge-stderr-to-stdout \
                python lhy_two_node_multi_switch.py \
                --num_strategy=7 \
                --ds_parallel_config ds_parallel_config/two_node/dp4_tp2_pp2.json,ds_parallel_config/two_node/dp2_tp4_pp2.json,ds_parallel_config/two_node/dp16.json,ds_parallel_config/two_node/dp8_tp2.json,ds_parallel_config/two_node/dp4_tp4.json,ds_parallel_config/two_node/dp2_tp8.json,ds_parallel_config/two_node/tp16.json \
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
                --use_flash_attn \
                --use_two_node \
                --from_strategy "$i" \
                --to_strategy "$j" \
                --switch_file "$file"
            fi
        done
    done
else
    for i in $(seq 0 4); do
        for j in $(seq 0 4); do
            if [ $i -eq $j ]; then
                continue 
            fi
            echo "switch from $i to $j begin..."
            echo -e "\nswitch from $i to $j: " >> "$file"
            if [ "${USE_BF16}" = true ]; then
                mpirun --allow-run-as-root -np 8 \
                -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
                -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL \
                --output-filename logs/ds_parallel --merge-stderr-to-stdout \
                python lhy_two_node_multi_switch.py \
                --num_strategy=5 \
                --ds_parallel_config ds_parallel_config/dp2_tp2_pp2.json,ds_parallel_config/dp8.json,ds_parallel_config/dp4_tp2.json,ds_parallel_config/dp2_tp4.json,ds_parallel_config/tp8.json \
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
                --from_strategy "$i" \
                --to_strategy "$j" \
                --switch_file "$file"
            else
                mpirun --allow-run-as-root -np 8 \
                -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
                -x HETU_SWITCH_ALGORITHM -x HETU_SWITCH_PROFILE -x HETU_INTERNAL_LOG_LEVEL \
                --output-filename logs/ds_parallel --merge-stderr-to-stdout \
                python lhy_two_node_multi_switch.py \
                --num_strategy=5 \
                --ds_parallel_config ds_parallel_config/dp2_tp2_pp2.json,ds_parallel_config/dp8.json,ds_parallel_config/dp4_tp2.json,ds_parallel_config/dp2_tp4.json,ds_parallel_config/tp8.json \
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
                --use_flash_attn \
                --from_strategy "$i" \
                --to_strategy "$j" \
                --switch_file "$file"
            fi
        done
    done
fi
    

        