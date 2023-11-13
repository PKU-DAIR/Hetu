# Use hetu 3d-parallel to train gpt

1. mkdir `data` folder under `examples/nlp/gpt`, copy data in this folder like `data/wikicorpus_en_training_0.hdf5`

2. run `bash scripts/train_hetu_gpt_parallel.sh` to use 2d parallel (dp=2, tp=2, 4 gpus)

3. run `bash scripts/train_hetu_gpt_3d_parallel.sh` to use 3d parallel (dp=2, tp=2, pp=2, 8 gpus)

# Use hetu 2d-parallel to inference gpt

1. mkdir `data` folder under `examples/nlp/gpt`, copy data in this folder like `data/wikicorpus_en_training_0.hdf5`

2. run `bash scripts/inference_hetu_gpt_parallel.sh` to use 2d parallel (dp=2, tp=2, 4 gpus)

# Important: use hetu 3d-parallel to train gpt and examine precision

1. use `conda activate hetu-py`

2. copy checkpoint folder (currently located on daim216 at `/home/gehao/lhy/Hetu-dev/examples/nlp/gpt/checkpoint`) to your own folder

3. run `bash scripts/lhy_train_hetu_gpt_3d_parallel.sh` to use 3d parallel (dp=2, tp=2, pp=2, 8 gpus) and train one round

4. run `python lhy_train_pytorch_gpt.py` to use pytorch and train one round

5. you could compare the averge loss in step 3 with the single loss in step 4 (they actually should exactly be 3.5993 and 3.0424, whose averge is 3.32 and equals to pytorch)

6. you could compare the model weight by running `python examine_ckpt.py`, you may add more assertion or change the learning rate for double check.