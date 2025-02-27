#!/bin/bash  
  
# 设置ROOT路径  
ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/combined_data.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

python preprocess_data.py \
--input $JSON_FILE \
--output-prefix web \
--tokenizer-type GPT2BPETokenizer   \
--vocab-file $VOCAB_FILE \
--merge-file $MERGE_FILE  \
--json-keys $JSON_KEY \
--workers 16 \