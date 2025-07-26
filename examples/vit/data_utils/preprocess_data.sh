#!/bin/bash  
  
# 设置ROOT路径  
ROOT_FOLDER=/home/gehao/njw1123/merge_all/Hetu-dev/tests/refactor/ci_test/data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt


python /home/gehao/njw1123/precision_alignment/examples/hydraulis/data_utils/preprocess_data.py \
--input $JSON_FILE \
--output-prefix refinedweb0 \
--tokenizer-type GPT2BPETokenizer   \
--vocab-file $VOCAB_FILE \
--merge-file $MERGE_FILE  \
--json-keys $JSON_KEY \
--workers 16 \