mkdir -p data
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O data/vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O data/merges.txt
python3 scripts/create_finetune_datasets.py